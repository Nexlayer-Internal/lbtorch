#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>
#include <torch/utils/checkpoint.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <tuple>
#include <vector>

// Mirrors the local helper macro test/cpp/api/autograd.cpp defines for
// itself (there's no shared one in support.h).
#define ASSERT_TENSOR_EQ(a, b) ASSERT_TRUE(torch::allclose((a), (b)))

using namespace torch::nn;
using torch::Tensor;
using torch::utils::checkpoint;

namespace {

// A stack of Linear layers (+ ReLU after each) sharing one hidden dimension,
// so arbitrary contiguous sub-ranges can be grouped into one checkpointed
// segment.
struct LayerStack {
  LayerStack(int64_t num_layers, int64_t dim) {
    for (const auto i : c10::irange(num_layers)) {
      (void)i;
      layers.push_back(Linear(dim, dim));
    }
  }

  std::vector<Tensor> parameters() const {
    std::vector<Tensor> params;
    for (const auto& layer : layers) {
      auto p = layer->parameters();
      params.insert(params.end(), p.begin(), p.end());
    }
    return params;
  }

  void zero_grad() {
    for (auto& layer : layers) {
      layer->zero_grad();
    }
  }

  // Applies layers [begin, end) to `x`, with ReLU after every layer.
  // Not `const`: torch::nn::ModuleHolder::operator->() has a const overload
  // that returns a const Contained*, and Module::forward() is not const, so
  // a const method here would fail to call `layers[i]->forward(x)`.
  Tensor apply_range(Tensor x, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      x = torch::relu(layers[i]->forward(x));
    }
    return x;
  }

  std::vector<Linear> layers;
};

} // namespace

// (a) Numerical correctness: a multi-layer network's forward+backward,
// wrapping some layers in torch::utils::checkpoint(), must produce exactly
// the same gradients as running the same layers without checkpointing.
TEST(CheckpointTest, MatchesNonCheckpointedGradients) {
  torch::manual_seed(0);
  LayerStack stack(6, 16);

  auto run = [&](bool use_checkpoint) {
    stack.zero_grad();
    torch::manual_seed(1);
    auto x = torch::randn({4, 16}, torch::requires_grad());

    Tensor h = x;
    if (use_checkpoint) {
      // Two checkpointed segments of three layers each -- exercises both
      // "checkpoint feeding into checkpoint" and a non-trivial segment size.
      h = checkpoint(
          [&](std::vector<Tensor> ins) -> std::vector<Tensor> {
            return {stack.apply_range(ins[0], 0, 3)};
          },
          {h})[0];
      h = checkpoint(
          [&](std::vector<Tensor> ins) -> std::vector<Tensor> {
            return {stack.apply_range(ins[0], 3, 6)};
          },
          {h})[0];
    } else {
      h = stack.apply_range(h, 0, 6);
    }

    auto loss = h.pow(2).sum();
    loss.backward();

    std::vector<Tensor> grads;
    grads.push_back(x.grad().clone());
    for (const auto& p : stack.parameters()) {
      // EXPECT, not ASSERT: this lambda returns std::vector<Tensor>, and
      // gtest's ASSERT_* macros expand to a bare `return;` on failure, which
      // does not compile in a non-void-returning function/lambda.
      EXPECT_TRUE(p.grad().defined());
      grads.push_back(p.grad().clone());
    }
    return grads;
  };

  auto baseline_grads = run(/*use_checkpoint=*/false);
  auto checkpoint_grads = run(/*use_checkpoint=*/true);

  ASSERT_EQ(baseline_grads.size(), checkpoint_grads.size());
  for (const auto i : c10::irange(baseline_grads.size())) {
    ASSERT_TENSOR_EQ(baseline_grads[i], checkpoint_grads[i]);
  }
}

// Basic plumbing: multiple inputs, multiple outputs, and a segment that
// doesn't use every input (so one returned gradient is legitimately
// undefined).
TEST(CheckpointTest, MultipleInputsAndOutputs) {
  torch::manual_seed(0);
  auto a = torch::randn({3, 3}, torch::requires_grad());
  auto b = torch::randn({3, 3}, torch::requires_grad());
  auto unused = torch::randn({3, 3}, torch::requires_grad());

  auto outputs = checkpoint(
      [](std::vector<Tensor> ins) -> std::vector<Tensor> {
        return {ins[0] + ins[1], ins[0] * ins[1]};
      },
      {a, b});
  ASSERT_EQ(outputs.size(), 2u);

  auto loss = outputs[0].sum() + outputs[1].sum();
  loss.backward();

  ASSERT_TENSOR_EQ(a.grad(), torch::ones({3, 3}) + b);
  ASSERT_TENSOR_EQ(b.grad(), torch::ones({3, 3}) + a);
  ASSERT_FALSE(unused.grad().defined());
}

// `preserve_rng_state=false` must still produce correct results for a
// deterministic (RNG-free) function -- it only opts out of the save/restore
// bookkeeping, not correctness for non-random functions.
TEST(CheckpointTest, PreserveRngStateFalseIsCorrectForDeterministicFn) {
  auto x = torch::randn({4, 4}, torch::requires_grad());
  auto out = checkpoint(
      [](std::vector<Tensor> ins) -> std::vector<Tensor> {
        return {ins[0].pow(2)};
      },
      {x},
      /*preserve_rng_state=*/false)[0];
  out.sum().backward();
  ASSERT_TENSOR_EQ(x.grad(), 2 * x.detach());
}

// (b) RNG correctness, direct test: the whole point of save/restore is that
// the recompute's dropout draw must reproduce the exact mask the original
// forward pass used. We capture the output of `fn` on *every* call
// (the first, no_grad forward, and the second, recompute-in-backward call)
// and assert they're bit-for-bit identical. If RNG save/restore were
// broken, the second call would draw from a different point in the RNG
// stream and (overwhelming odds, p=0.5 over 1024 elements) produce a
// different mask here -- silently wrong gradients, not a crash, which is
// exactly the failure mode a less direct test could miss.
TEST(CheckpointTest, DropoutRecomputeReproducesForwardMask) {
  torch::manual_seed(42);
  Linear fc(32, 32);
  Dropout drop(0.5);
  fc->train();
  drop->train();

  std::vector<Tensor> fn_call_outputs;
  auto segment_fn = [&](std::vector<Tensor> ins) -> std::vector<Tensor> {
    auto h = torch::relu(fc->forward(ins[0]));
    h = drop->forward(h);
    fn_call_outputs.push_back(h.detach().clone());
    return {h};
  };

  auto x = torch::randn({8, 32}, torch::requires_grad());
  auto out = checkpoint(segment_fn, {x})[0];
  ASSERT_EQ(fn_call_outputs.size(), 1u); // only the no_grad forward ran so far

  out.sum().backward();
  ASSERT_EQ(fn_call_outputs.size(), 2u); // backward triggered the recompute

  ASSERT_TENSOR_EQ(fn_call_outputs[0], fn_call_outputs[1]);
}

// (b) RNG correctness, end-to-end test: with the global RNG seeded
// identically, a checkpointed forward+backward through a dropout layer must
// produce *exactly* the same gradients as the same network run without
// checkpointing at all. This only holds if the recompute inside backward()
// replays the same dropout mask the original forward pass drew -- i.e. it
// is a direct correctness check on the RNG save/restore logic, not just an
// internal self-consistency check.
TEST(CheckpointTest, DropoutGradientMatchesNonCheckpointedBaseline) {
  Linear fc1(32, 32);
  Linear fc2(32, 32);
  Dropout drop(0.5);
  fc1->train();
  fc2->train();
  drop->train();

  auto forward = [&](Tensor x) {
    auto h = torch::relu(fc1->forward(x));
    h = drop->forward(h);
    h = fc2->forward(h);
    return h;
  };

  auto run = [&](bool use_checkpoint) {
    fc1->zero_grad();
    fc2->zero_grad();
    torch::manual_seed(7);
    auto x = torch::randn({8, 32}, torch::requires_grad());

    Tensor h;
    if (use_checkpoint) {
      h = checkpoint(
          [&](std::vector<Tensor> ins) -> std::vector<Tensor> {
            return {forward(ins[0])};
          },
          {x})[0];
    } else {
      h = forward(x);
    }
    h.pow(2).sum().backward();

    return std::make_tuple(
        x.grad().clone(), fc1->weight.grad().clone(), fc2->weight.grad().clone());
  };

  auto [x_grad_baseline, w1_grad_baseline, w2_grad_baseline] = run(false);
  auto [x_grad_checkpoint, w1_grad_checkpoint, w2_grad_checkpoint] = run(true);

  ASSERT_TENSOR_EQ(x_grad_baseline, x_grad_checkpoint);
  ASSERT_TENSOR_EQ(w1_grad_baseline, w1_grad_checkpoint);
  ASSERT_TENSOR_EQ(w2_grad_baseline, w2_grad_checkpoint);
}

#ifdef USE_CUDA
// (b) Memory: checkpointing a deep enough stack should measurably lower
// peak allocator usage vs. running the same stack without checkpointing,
// since intermediate per-layer activations for the checkpointed region
// aren't kept around until backward.
//
// NOTE: this test requires a real CUDA device and was not executed in the
// environment this port was developed in (no GPU available there) -- see
// the PR description. The API usage below (c10::cuda::CUDACachingAllocator::
// resetPeakStats/getDeviceStats, c10::CachingAllocator::StatType::AGGREGATE)
// was checked against c10/cuda/CUDACachingAllocator.h and c10/core/
// Allocator.h, not guessed from memory, but it is still unverified by an
// actual run.
TEST(CheckpointTest, ReducesPeakMemoryOnCuda) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  constexpr int64_t kNumLayers = 64;
  constexpr int64_t kDim = 2048;
  constexpr int64_t kBatch = 256;
  const auto device = torch::kCUDA;
  const c10::DeviceIndex device_index = 0;

  torch::manual_seed(0);
  LayerStack stack(kNumLayers, kDim);
  for (auto& layer : stack.layers) {
    layer->to(device);
  }

  auto run_and_measure_peak = [&](bool use_checkpoint) -> int64_t {
    stack.zero_grad();
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);

    auto x =
        torch::randn({kBatch, kDim}, torch::TensorOptions().device(device))
            .requires_grad_(true);

    Tensor h = x;
    if (use_checkpoint) {
      // One checkpointed segment covering the whole stack: this is the
      // regime where checkpointing's memory win is largest (none of the
      // per-layer activations across the whole stack are kept).
      h = checkpoint(
          [&](std::vector<Tensor> ins) -> std::vector<Tensor> {
            return {stack.apply_range(ins[0], 0, kNumLayers)};
          },
          {h})[0];
    } else {
      h = stack.apply_range(h, 0, kNumLayers);
    }
    h.pow(2).sum().backward();

    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
    return stats
        .allocated_bytes[static_cast<size_t>(
            c10::CachingAllocator::StatType::AGGREGATE)]
        .peak;
  };

  auto baseline_peak = run_and_measure_peak(/*use_checkpoint=*/false);
  auto checkpoint_peak = run_and_measure_peak(/*use_checkpoint=*/true);

  // Not a tight bound -- just confirms checkpointing is meaningfully
  // cheaper, not merely noise-level different.
  ASSERT_LT(checkpoint_peak, baseline_peak);
}
#endif // USE_CUDA
