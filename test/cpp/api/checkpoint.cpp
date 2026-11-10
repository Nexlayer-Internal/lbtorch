#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>
#include <torch/utils/checkpoint.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <algorithm>
#include <cstdio>
#include <string>
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
// (b) Memory: checkpointing a deep stack, split into several checkpointed
// segments, should measurably lower peak allocator usage vs. running the
// same stack without checkpointing.
//
// An earlier version of this test (dim=2048, batch=256, one checkpoint()
// call covering the entire 64-layer stack) WAS run on real GPU hardware (an
// RTX PRO 6000 Blackwell, by the project maintainer) and did NOT show a
// reduction -- checkpointed peak came out a hair *higher* than the
// baseline (2196.76 MB vs 2192.57 MB), both sitting almost exactly on the
// stack's fixed weight+gradient floor (64 layers * (16 MB weight + 16 MB
// gradient) ~= 2.1 GB). Investigating *why*, without GPU access to
// reproduce it directly, turned up two independent, compounding problems
// with that version of the test -- not a bug in torch::utils::checkpoint()
// itself (separately confirmed correct: the other 6 CheckpointTest cases,
// including both dropout/RNG tests, passed on that same GPU run):
//
// 1. Checkpointing only reduces *activation* memory; it has zero effect on
//    parameter + parameter-gradient memory, which depends only on model
//    size, not on checkpointing. At dim=2048, each layer's weight+gradient
//    (16 MB + 16 MB = 32 MB) outweighed its activation (batch=256 * dim=2048
//    * 4 bytes ~= 2 MB) by 16x. Worse: because regular (non-checkpointed)
//    backward already frees each layer's saved activation as it finishes
//    that layer (autograd's normal release_variables() behavior, nothing
//    checkpointing-specific), and gradient accumulation(+32MB/layer) swamps
//    activation release (-2MB/layer) at every step of backward, the
//    non-checkpointed baseline's peak *already* lands almost exactly on the
//    weight+gradient floor, with no "all 64 layers' activations held at
//    once" spike to avoid in the first place. There was barely anything
//    for checkpointing to save. This test now uses a small `dim` and a
//    large `batch` instead (activation size scales with batch*dim; weight/
//    gradient size scales with dim*dim and is batch-independent), so
//    activation memory dominates by roughly two orders of magnitude -- also
//    a more realistic proxy for why checkpointing matters in real
//    transformer training (activation memory, not parameter memory, is
//    what it exists to address).
// 2. Independent of (1): checkpointing the *entire* stack as a single
//    segment provides, provably, no benefit at all. Reasoning through it:
//    the no_grad forward pass inside checkpoint()'s forward() is cheap
//    (O(1) memory -- nothing but the current working tensor is retained,
//    same as any ordinary eager loop with no autograd graph), but
//    backward() has to recompute-then-differentiate the *entire*
//    checkpointed region as one block, which means reconstructing every
//    single layer's activation simultaneously before any of them can be
//    freed -- structurally identical to the plain, non-checkpointed
//    forward+backward. One giant segment buys nothing. The memory benefit
//    only shows up when the stack is split into *multiple* segments, so
//    that at any point during backward only one segment's worth of
//    activations needs to be alive, rather than the whole stack's. This
//    test now checkpoints the stack as several segments (kGroupSize layers
//    each) and -- purely as a diagnostic, not asserted, since it's a
//    property of how this test calls checkpoint() rather than of
//    checkpoint() itself -- also measures and prints the one-giant-segment
//    peak for direct comparison, to confirm (2) empirically rather than by
//    reasoning alone.
//
// NOTE: this corrected version has not itself been executed on real GPU
// hardware yet (still no GPU in the environment this fix was written in);
// the reasoning above is this port's best analysis of the prior run's
// result, not a confirmed fix. Re-run on GPU hardware before relying on it.
// Prints the allocator stats trajectory (reset / post-forward / post-
// backward, for all three variants) specifically so a real run's output
// can be inspected directly rather than trusting only the final assertion.
TEST(CheckpointTest, ReducesPeakMemoryOnCuda) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  constexpr int64_t kNumLayers = 64;
  constexpr int64_t kGroupSize = 8; // 8 checkpointed segments of 8 layers each
  constexpr int64_t kDim = 256;
  constexpr int64_t kBatch = 16384; // batch >> dim: activations now dominate
  const auto device = torch::kCUDA;
  const c10::DeviceIndex device_index = 0;

  torch::manual_seed(0);
  LayerStack stack(kNumLayers, kDim);
  for (auto& layer : stack.layers) {
    layer->to(device);
  }

  auto print_stats = [&](const char* label) {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
    const auto& agg = stats.allocated_bytes[static_cast<size_t>(
        c10::CachingAllocator::StatType::AGGREGATE)];
    std::fprintf(
        stderr,
        "[ReducesPeakMemoryOnCuda] %-32s current=%9.2f MB  peak=%9.2f MB\n",
        label,
        static_cast<double>(agg.current) / 1e6,
        static_cast<double>(agg.peak) / 1e6);
  };

  enum class Granularity { kNone, kOneGiantSegment, kGrouped };

  auto run_and_measure_peak = [&](Granularity granularity) -> int64_t {
    const char* name = granularity == Granularity::kNone
        ? "baseline"
        : (granularity == Granularity::kOneGiantSegment ? "one-giant-segment"
                                                         : "grouped");
    stack.zero_grad();
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);
    print_stats((std::string("reset (") + name + ")").c_str());

    auto x =
        torch::randn({kBatch, kDim}, torch::TensorOptions().device(device))
            .requires_grad_(true);

    Tensor h = x;
    switch (granularity) {
      case Granularity::kNone:
        h = stack.apply_range(h, 0, kNumLayers);
        break;
      case Granularity::kOneGiantSegment:
        h = checkpoint(
            [&](std::vector<Tensor> ins) -> std::vector<Tensor> {
              return {stack.apply_range(ins[0], 0, kNumLayers)};
            },
            {h})[0];
        break;
      case Granularity::kGrouped:
        for (int64_t start = 0; start < kNumLayers; start += kGroupSize) {
          const auto begin = static_cast<size_t>(start);
          const auto end = static_cast<size_t>(
              std::min(start + kGroupSize, kNumLayers));
          h = checkpoint(
              [&, begin, end](std::vector<Tensor> ins) -> std::vector<Tensor> {
                return {stack.apply_range(ins[0], begin, end)};
              },
              {h})[0];
        }
        break;
    }
    print_stats((std::string("after forward (") + name + ")").c_str());

    h.pow(2).sum().backward();
    print_stats((std::string("after backward (") + name + ")").c_str());

    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
    return stats
        .allocated_bytes[static_cast<size_t>(
            c10::CachingAllocator::StatType::AGGREGATE)]
        .peak;
  };

  auto baseline_peak = run_and_measure_peak(Granularity::kNone);
  auto giant_segment_peak = run_and_measure_peak(Granularity::kOneGiantSegment);
  auto grouped_peak = run_and_measure_peak(Granularity::kGrouped);

  std::fprintf(
      stderr,
      "[ReducesPeakMemoryOnCuda] SUMMARY baseline=%.2f MB  "
      "one-giant-segment=%.2f MB (expected ~= baseline, see comment)  "
      "grouped=%.2f MB  (baseline/grouped ratio=%.2fx)\n",
      static_cast<double>(baseline_peak) / 1e6,
      static_cast<double>(giant_segment_peak) / 1e6,
      static_cast<double>(grouped_peak) / 1e6,
      static_cast<double>(baseline_peak) / static_cast<double>(grouped_peak));

  // The actual claim this PR makes: checkpointing with reasonable
  // granularity reduces peak memory. (The one-giant-segment number is
  // logged above for comparison but deliberately not asserted on -- per
  // the comment above this test, it is expected to show ~no improvement,
  // which is a property of that calling pattern, not a correctness bug.)
  ASSERT_LT(grouped_peak, baseline_peak);
}
#endif // USE_CUDA
