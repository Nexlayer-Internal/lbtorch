// Real, end-to-end, 2-rank FSDP training regression test -- the task brief's
// item 6 ("the regression test that matters most"). Launched once per rank
// via run_multiproc.sh (same launcher collective_transport's own tests use;
// see that file's header comment), each process:
//   1. builds a small multi-layer model (a LayerStack-like stack of Linear
//      layers, same shape as test/cpp/api/checkpoint.cpp's own helper),
//   2. wraps each layer as its own FSDPUnit (this file's new code),
//   3. runs several real training steps: forward, backward, AdamW step,
//   4. confirms (a) the process completes and exits cleanly (the direct,
//      literal regression test for the original IP66 FSDP deadlock this
//      whole project traces back to), (b) loss decreases, (c) logs which
//      transport was actually selected.
//
// No Python anywhere in this build or test path, per the project's
// zero-Python goal -- same discipline run_multiproc.sh's own header comment
// states for collective_transport's tests.

#include <torch/csrc/distributed/c10d/collective_transport/test/test_common.h>
#include <torch/csrc/distributed/c10d/fsdp/fsdp_unit.h>

#include <torch/torch.h>
#include <torch/utils/checkpoint.h>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using torch::distributed::fsdp::FSDPUnit;
using torch::distributed::fsdp::UnitForwardFn;

namespace {

constexpr int64_t kDim = 512;
constexpr int64_t kNumLayers = 4;
constexpr int64_t kBatch = 64;
constexpr int64_t kSteps = 60;
// reduce_scatter SUMS across ranks (transport.h); this test deliberately
// feeds every rank the SAME data (seeded identically below) rather than a
// real data-parallel split, purely to make the loss trajectory trivial to
// reason about and assert on. That means each rank's local gradient is
// value-identical, so the summed gradient is exactly 2x (world_size x) the
// single-GPU gradient -- in principle this would call for halving the
// learning rate to match single-GPU AdamW step-for-step. Empirically it
// does not matter here: AdamW normalizes by a second-moment (RMS-like)
// estimate that scales with the same factor as the gradient itself, so its
// *update* is close to scale-invariant to a constant gradient-magnitude
// factor (beyond epsilon's small effect) -- confirmed by comparing this
// test's actual run against test_baseline_single_gpu.cpp at the SAME kLr:
// the two loss trajectories track closely without any LR adjustment. Using
// the identical kLr as the baseline, unmodified, for the most direct,
// least-caveated comparison.
constexpr double kLr = 0.05;

} // namespace

int main() {
  auto env = torch::collectives::test::read_proc_env();
  TORCH_CHECK(env.world_size == 2, "this test is written for world_size=2");
  c10::cuda::set_device(env.rank);
  torch::manual_seed(1234); // identical on every rank, see kLr's comment

  auto store = torch::collectives::test::make_tcp_store(env);

  torch::collectives::TransportOptions topts;
  // slot_capacity_bytes must cover the largest single all_gather/
  // reduce_scatter payload any unit issues: padded_numel * dtype_size for
  // the biggest layer. kDim*kDim fp32 weight + kDim bias, generously padded.
  const size_t slot_capacity_bytes =
      static_cast<size_t>(kDim * kDim + kDim) * sizeof(float) * 2;
  auto transport = torch::collectives::make_transport(
      store, env.rank, env.world_size, env.rank, slot_capacity_bytes, topts);

  std::cout << "[rank " << env.rank << "] transport selected: "
            << (transport->kind() == torch::collectives::TransportKind::kNcclFastPath
                    ? "nccl_fast_path"
                    : "host_staged_fallback")
            << std::endl;

  const auto cuda_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, env.rank);

  // --- Build the model: kNumLayers Linear layers, ReLU between, each its
  // own FSDPUnit (same per-layer-wrapping granularity PLAN.md/FSDP-DESIGN.md
  // discuss; a real GRPO actor would wrap whole transformer blocks the same
  // way). Every rank constructs IDENTICAL layers because torch::manual_seed
  // was set identically above and layer construction order/op sequence is
  // identical -- this test's substitute for a real broadcast-at-init step
  // (make_transport/CollectiveTransport deliberately has no broadcast op
  // yet, per transport.h's own scope note).
  std::vector<torch::nn::Linear> layers;
  std::vector<std::unique_ptr<FSDPUnit>> units;
  layers.reserve(kNumLayers);
  units.reserve(kNumLayers);
  for (const auto i : c10::irange(kNumLayers)) {
    torch::nn::Linear layer(kDim, kDim);
    layer->to(cuda_opts.device());
    layers.push_back(layer);

    // Layer index 1 proves task item 5 (gradient-checkpointing composition):
    // its forward_fn runs through torch::utils::checkpoint() internally
    // instead of calling the layer directly.
    const bool use_checkpoint = (i == 1);
    // No ReLU after the final layer: the regression target is an unbounded
    // Gaussian linear map, so forcing the last layer's output through ReLU
    // caps the achievable loss well above zero regardless of how well
    // training is going (verified on the single-GPU baseline: this exact
    // mistake, still present there until fixed, produced a loss that never
    // meaningfully moved -- a toy-model-design bug, not an FSDP bug; see
    // test_baseline_single_gpu.cpp's comments).
    const bool apply_relu = (i + 1 < kNumLayers);
    auto linear_module = layer.ptr(); // shared_ptr<Module>, kept alive by FSDPUnit
    UnitForwardFn fwd;
    if (use_checkpoint) {
      torch::nn::Linear captured = layer;
      fwd = [captured, apply_relu](std::vector<torch::Tensor> in) mutable {
        auto fn = [captured, apply_relu](std::vector<torch::Tensor> inner) mutable {
          torch::Tensor out = captured->forward(inner[0]);
          if (apply_relu) {
            out = torch::relu(out);
          }
          return std::vector<torch::Tensor>{out};
        };
        return torch::utils::checkpoint(fn, std::move(in));
      };
    } else {
      torch::nn::Linear captured = layer;
      fwd = [captured, apply_relu](std::vector<torch::Tensor> in) mutable {
        torch::Tensor out = captured->forward(in[0]);
        if (apply_relu) {
          out = torch::relu(out);
        }
        return std::vector<torch::Tensor>{out};
      };
    }
    units.push_back(std::make_unique<FSDPUnit>(linear_module, std::move(fwd), transport));
  }

  // --- AdamW over every unit's shard -- the ONLY thing the optimizer ever
  // sees is each unit's 1/world_size shard (task item 3).
  std::vector<torch::Tensor> optimizer_params;
  for (auto& u : units) {
    optimizer_params.push_back(u->shard_param());
  }
  torch::optim::AdamWOptions adamw_opts(kLr);
  torch::optim::AdamW optimizer(optimizer_params, adamw_opts);

  // --- Synthetic regression target: a fixed random linear map the model has
  // to learn to approximate (same input distribution every step; real loss
  // decrease over kSteps is the "gradients/sharding/optimizer wired
  // correctly" sanity check per task item 6b).
  torch::Tensor target_matrix = torch::randn({kDim, kDim}, cuda_opts);
  // Fixed batch, reused every step -- see test_baseline_single_gpu.cpp's
  // comment: this is a "does the wiring actually learn" check, not a
  // generalization benchmark, and avoids conflating slow/noisy convergence
  // from fresh-random-batches-every-step with an FSDP correctness bug
  // (verified: the identical issue reproduces on the non-FSDP baseline).
  torch::Tensor x = torch::randn({kBatch, kDim}, cuda_opts);
  torch::Tensor target = torch::matmul(x, target_matrix).detach();

  double first_loss = -1.0;
  double last_loss = -1.0;
  for (const auto step : c10::irange(kSteps)) {
    // Instrumentation for task item 5: unit[1] (the checkpointed one) must
    // unshard exactly twice per step (once pre-forward, once pre-backward),
    // never three times -- confirming checkpoint's recompute reused the
    // already-unsharded parameters instead of forcing a second all_gather.
    units[1]->reset_unshard_call_count();

    optimizer.zero_grad();
    torch::Tensor h = x;
    for (auto& u : units) {
      h = u->forward({h})[0];
    }
    torch::Tensor loss = torch::mse_loss(h, target);
    loss.backward();
    optimizer.step();

    TORCH_CHECK(
        units[1]->unshard_call_count() == 2,
        "checkpointed unit unsharded ",
        units[1]->unshard_call_count(),
        " times in one step (expected exactly 2: forward + single "
        "pre-backward re-unshard, independent of checkpoint's internal "
        "recompute call count)");

    const double loss_val = loss.item<double>();
    if (step == 0) {
      first_loss = loss_val;
    }
    last_loss = loss_val;
    std::cout << "[rank " << env.rank << "] step " << step
              << " loss=" << loss_val << std::endl;
  }

  std::cout << "[rank " << env.rank << "] first_loss=" << first_loss
            << " last_loss=" << last_loss << std::endl;
  TORCH_CHECK(
      last_loss < first_loss,
      "loss did not decrease: first=",
      first_loss,
      " last=",
      last_loss);

  std::cout << "[rank " << env.rank << "] PASS" << std::endl;
  return 0;
}
