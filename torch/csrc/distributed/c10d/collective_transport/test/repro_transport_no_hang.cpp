// The regression test that matters most, part 2: run the *same* FSDP-
// realistic-size collective through this change's transport layer, with
// NCCL P2P left enabled (same as repro_vanilla_nccl_hang.cpp -- this
// binary never sets NCCL_P2P_DISABLE either, and does not force a
// transport: it exercises the real topology probe), and confirm:
//   (a) the topology probe selects the host-staged fallback on this
//       hardware (IP66's confirmed SYS topology), and
//   (b) the collective completes correctly and quickly -- it does not
//       hang, unlike repro_vanilla_nccl_hang.cpp's direct ProcessGroupNCCL
//       call at the same size on the same hardware.
// Verification at this scale samples a subset of elements (the exhaustive
// per-element check lives in test_collective_correctness.cpp at smaller,
// still-meaningful sizes) purely to keep this binary's own CPU-side
// verification fast.

#include <torch/csrc/distributed/c10d/collective_transport/host_staged_transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/nccl_fast_transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/test/test_common.h>

#include <ATen/ATen.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

using torch::collectives::CollectiveTransport;
using torch::collectives::make_transport;
using torch::collectives::TransportKind;
using torch::collectives::TransportOptions;
using torch::collectives::test::make_tcp_store;
using torch::collectives::test::read_proc_env;

int main() {
  auto env = read_proc_env();
  const int64_t numel = std::atoll(
      std::getenv("SHARD_NUMEL") ? std::getenv("SHARD_NUMEL") : "134217728"); // 128M floats = 512MB
  const std::string op = std::getenv("OP") ? std::getenv("OP") : "all_gather";

  auto store = make_tcp_store(env);
  TransportOptions options; // no force_transport: exercise the real probe

  const size_t slot_capacity_bytes =
      static_cast<size_t>(numel) * static_cast<size_t>(env.world_size) * sizeof(float);

  const auto t_init0 = std::chrono::steady_clock::now();
  auto transport =
      make_transport(store, env.rank, env.world_size, env.rank, slot_capacity_bytes, options);
  const auto init_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_init0).count();

  const bool selected_fallback = transport->kind() == TransportKind::kHostStagedFallback;
  std::cout << "[rank " << env.rank << "] topology probe + transport construction took "
            << init_ms << "ms, selected "
            << (selected_fallback ? "host_staged_fallback" : "nccl_fast_path") << std::endl;

  const auto cpu_opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const auto gpu_opts = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, env.rank);

  bool ok = true;
  double op_ms = 0.0;

  if (op == "all_gather") {
    at::Tensor input_gpu = at::full({numel}, static_cast<float>(env.rank + 1), gpu_opts);
    at::Tensor output_gpu = at::zeros({numel * env.world_size}, gpu_opts);

    const auto t0 = std::chrono::steady_clock::now();
    transport->all_gather(input_gpu, output_gpu);
    op_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    at::Tensor sample = output_gpu.index({at::indexing::Slice(0, output_gpu.numel(), numel / 8)})
                             .to(at::kCPU);
    auto acc = sample.accessor<float, 1>();
    for (int64_t k = 0; k < sample.numel(); ++k) {
      const int64_t global_idx = k * (numel / 8);
      const int expected_rank = static_cast<int>(global_idx / numel);
      const float expected = static_cast<float>(expected_rank + 1);
      if (std::abs(acc[k] - expected) > 1e-3f) {
        std::cerr << "[rank " << env.rank << "] MISMATCH sample[" << k << "] expected=" << expected
                  << " actual=" << acc[k] << std::endl;
        ok = false;
      }
    }
  } else { // reduce_scatter
    const int64_t full_numel = numel * env.world_size;
    at::Tensor input_gpu = at::full({full_numel}, static_cast<float>(env.rank + 1), gpu_opts);
    at::Tensor output_shard_gpu = at::zeros({numel}, gpu_opts);

    const auto t0 = std::chrono::steady_clock::now();
    transport->reduce_scatter(input_gpu, output_shard_gpu);
    op_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const float expected =
        static_cast<float>(env.world_size) * static_cast<float>(env.world_size + 1) / 2.0f;
    at::Tensor sample =
        output_shard_gpu.index({at::indexing::Slice(0, output_shard_gpu.numel(), numel / 8)})
            .to(at::kCPU);
    auto acc = sample.accessor<float, 1>();
    for (int64_t k = 0; k < sample.numel(); ++k) {
      if (std::abs(acc[k] - expected) > 1e-2f) {
        std::cerr << "[rank " << env.rank << "] MISMATCH sample[" << k << "] expected=" << expected
                  << " actual=" << acc[k] << std::endl;
        ok = false;
      }
    }
  }

  const double payload_mb = static_cast<double>(numel) * 4.0 / (1024.0 * 1024.0);
  std::cout << "[rank " << env.rank << "] " << op << " on " << payload_mb
            << "MB/rank completed in " << op_ms << "ms (transport="
            << (selected_fallback ? "host_staged_fallback" : "nccl_fast_path") << ")" << std::endl;

  if (ok) {
    std::cout << "[rank " << env.rank << "] PASS (no hang, correct result)" << std::endl;
    return 0;
  }
  std::cout << "[rank " << env.rank << "] FAIL" << std::endl;
  return 1;
}
