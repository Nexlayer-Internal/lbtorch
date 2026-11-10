// Correctness + timing harness for CollectiveTransport, launched once per
// rank by run_multiproc.sh. Not a gtest binary (deliberately -- this is
// standalone C++ infrastructure with no dependency on the rest of
// libtorch's test harness); each process prints PASS/FAIL and exits
// 0/1 accordingly, and run_multiproc.sh fails the whole run if any
// rank's exit code is nonzero.
//
// Environment variables:
//   RANK, WORLD_SIZE, MASTER_PORT  (see test_common.h)
//   OP                 "all_gather" | "reduce_scatter"
//   SHARD_NUMEL         element count of this rank's shard (all_gather) or
//                       output shard (reduce_scatter)
//   DTYPE               "float32" | "bfloat16" (default float32)
//   FORCE_TRANSPORT     "fallback" | "fast" | "auto" (default "auto") --
//                       "fast" bypasses the topology probe entirely and
//                       forces ProcessGroupNCCL even on hardware the probe
//                       would otherwise reject; on IP66 specifically this
//                       is expected to hang (see repro_vanilla_nccl_hang.cpp
//                       and the PR description) -- that is a real property
//                       of this hardware with P2P enabled, not a bug in
//                       this flag. Use it under a bounded `timeout`
//                       (run_multiproc.sh always wraps this binary in one).
//   NUM_LOCAL_GPUS      number of physical GPUs to round-robin ranks over
//                       (default 1 -- all ranks share device 0, valid for
//                       the host-staged transport, which never touches
//                       another rank's device memory directly)
//   REPEAT              number of times to repeat the op (default 1) --
//                       used to report timing and to exercise the
//                       double-staging-slot alternation across calls.

#include <torch/csrc/distributed/c10d/collective_transport/host_staged_transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/nccl_fast_transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/test/test_common.h>

#include <ATen/ATen.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

using torch::collectives::CollectiveTransport;
using torch::collectives::make_transport;
using torch::collectives::TransportKind;
using torch::collectives::TransportOptions;
using torch::collectives::test::make_tcp_store;
using torch::collectives::test::read_proc_env;

namespace {

std::string getenv_or(const char* name, const std::string& def) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : def;
}

at::ScalarType parse_dtype(const std::string& s) {
  if (s == "bfloat16") {
    return at::kBFloat16;
  }
  return at::kFloat;
}

} // namespace

int main() {
  auto env = read_proc_env();
  const std::string op = getenv_or("OP", "all_gather");
  const int64_t shard_numel = std::atoll(getenv_or("SHARD_NUMEL", "4096").c_str());
  const at::ScalarType dtype = parse_dtype(getenv_or("DTYPE", "float32"));
  const std::string force = getenv_or("FORCE_TRANSPORT", "auto");
  const int num_local_gpus = std::atoi(getenv_or("NUM_LOCAL_GPUS", "1").c_str());
  const int repeat = std::atoi(getenv_or("REPEAT", "1").c_str());

  const int device_id = env.rank % num_local_gpus;
  // (Deliberately no explicit cudaSetDevice() call here: ATen's own tensor
  // creation with an explicit device index below sets the current device
  // itself, matching repro_vanilla_nccl_hang.cpp's flow exactly. See
  // HostStagedTransport, which does its own explicit cudaSetDevice()
  // internally for the fallback path.)

  auto store = make_tcp_store(env);

  TransportOptions options;
  if (force == "fallback") {
    options.force_transport = TransportKind::kHostStagedFallback;
  } else if (force == "fast") {
    options.force_transport = TransportKind::kNcclFastPath;
  }

  const size_t elem_size = (dtype == at::kBFloat16) ? 2 : 4;
  const size_t slot_capacity_bytes =
      static_cast<size_t>(shard_numel) * static_cast<size_t>(env.world_size) * elem_size;

  std::shared_ptr<CollectiveTransport> transport;
  try {
    transport = make_transport(
        store, env.rank, env.world_size, device_id, slot_capacity_bytes, options);
  } catch (const std::exception& e) {
    std::cerr << "[rank " << env.rank << "] FAIL: make_transport threw: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "[rank " << env.rank << "] selected transport kind="
            << (transport->kind() == TransportKind::kNcclFastPath ? "nccl_fast_path"
                                                                    : "host_staged_fallback")
            << std::endl;

  const auto cpu_opts = at::TensorOptions().dtype(dtype).device(at::kCPU);
  const auto gpu_opts = at::TensorOptions().dtype(dtype).device(at::kCUDA, device_id);

  bool ok = true;
  double total_ms = 0.0;

  if (op == "all_gather") {
    // input_shard[i] = rank * 1000 + i -- distinct per rank and per
    // position, so a bug that mixes up rank order or slot indexing is
    // caught, not masked by a uniform-value test.
    at::Tensor input_cpu = at::empty({shard_numel}, cpu_opts);
    {
      auto input_f = input_cpu.to(at::kFloat);
      auto acc = input_f.accessor<float, 1>();
      for (int64_t i = 0; i < shard_numel; ++i) {
        acc[i] = static_cast<float>(env.rank) * 1000.0f + static_cast<float>(i);
      }
      input_cpu = input_f.to(dtype);
    }
    at::Tensor input_gpu = input_cpu.to(gpu_opts);
    at::Tensor output_gpu = at::empty({shard_numel * env.world_size}, gpu_opts);

    for (int it = 0; it < repeat; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      transport->all_gather(input_gpu, output_gpu);
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    at::Tensor output_cpu = output_gpu.to(at::kCPU).to(at::kFloat);
    auto acc = output_cpu.accessor<float, 1>();
    for (int r = 0; r < env.world_size && ok; ++r) {
      for (int64_t i = 0; i < shard_numel; ++i) {
        const float expected = static_cast<float>(r) * 1000.0f + static_cast<float>(i);
        const float actual = acc[r * shard_numel + i];
        // bf16 has ~8 bits of mantissa, so e.g. 514.0f is not exactly
        // representable once input_cpu was converted to bf16 before upload
        // (it rounds to the nearest representable bf16 value); all_gather
        // itself does no arithmetic, so compare against what bf16 can
        // actually represent, not the pre-rounding float value.
        float expected_cmp = expected;
        float tol = 1e-3f;
        if (dtype == at::kBFloat16) {
          expected_cmp = at::tensor(expected, at::kFloat).to(at::kBFloat16).to(at::kFloat).item<float>();
          tol = std::max(1.0f, std::abs(expected_cmp) * 0.01f);
        }
        if (std::abs(actual - expected_cmp) > tol) {
          std::cerr << "[rank " << env.rank << "] MISMATCH at gathered[" << (r * shard_numel + i)
                    << "] expected=" << expected << " actual=" << actual << std::endl;
          ok = false;
          break;
        }
      }
    }
  } else if (op == "reduce_scatter") {
    // input[g] = rank * 1000 + g, same shape (world_size * shard_numel) on
    // every rank, distinct values per rank so summation order/placement
    // bugs are caught.
    const int64_t full_numel = shard_numel * env.world_size;
    at::Tensor input_cpu = at::empty({full_numel}, cpu_opts);
    {
      auto input_f = input_cpu.to(at::kFloat);
      auto acc = input_f.accessor<float, 1>();
      for (int64_t g = 0; g < full_numel; ++g) {
        acc[g] = static_cast<float>(env.rank) * 1000.0f + static_cast<float>(g);
      }
      input_cpu = input_f.to(dtype);
    }
    at::Tensor input_gpu = input_cpu.to(gpu_opts);
    at::Tensor output_shard_gpu = at::empty({shard_numel}, gpu_opts);

    for (int it = 0; it < repeat; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      transport->reduce_scatter(input_gpu, output_shard_gpu);
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    at::Tensor output_cpu = output_shard_gpu.to(at::kCPU).to(at::kFloat);
    auto acc = output_cpu.accessor<float, 1>();
    const int N = env.world_size;
    for (int64_t j = 0; j < shard_numel && ok; ++j) {
      const int64_t g = static_cast<int64_t>(env.rank) * shard_numel + j;
      // sum_{r=0}^{N-1} (r*1000 + g) = 1000*N*(N-1)/2 + N*g
      const float expected = 1000.0f * static_cast<float>(N) * static_cast<float>(N - 1) / 2.0f +
          static_cast<float>(N) * static_cast<float>(g);
      const float actual = acc[j];
      // bf16 accumulation of several large-magnitude terms loses precision;
      // use a relative tolerance.
      const float tol = std::max(1.0f, std::abs(expected) * 0.02f);
      if (std::abs(actual - expected) > tol) {
        std::cerr << "[rank " << env.rank << "] MISMATCH at output_shard[" << j
                  << "] expected=" << expected << " actual=" << actual << std::endl;
        ok = false;
        break;
      }
    }
  } else {
    std::cerr << "[rank " << env.rank << "] FAIL: unknown OP=" << op << std::endl;
    return 1;
  }

  std::cout << "[rank " << env.rank << "] " << op << " x" << repeat
            << " total_wall_ms=" << total_ms << " avg_ms=" << (total_ms / repeat) << std::endl;

  if (ok) {
    std::cout << "[rank " << env.rank << "] PASS" << std::endl;
    return 0;
  }
  std::cout << "[rank " << env.rank << "] FAIL" << std::endl;
  return 1;
}
