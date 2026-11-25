// The regression test that matters most, part 1: reproduce the *failure*
// mode first. Constructs a stock ProcessGroupNCCL directly (bypassing this
// change's transport layer entirely) and calls all_gather_single at an
// FSDP-flat-parameter-shard-realistic size, on IP66's confirmed-SYS-
// topology hardware, with NCCL P2P left *enabled* (this binary does not
// set, and must not be run with, NCCL_P2P_DISABLE=1 -- that env-var
// workaround is exactly what FSDP-DESIGN.md Sec 0 calls "a real fix,
// not": the whole point of this change is to supersede it).
//
// This process is expected to hang or be pathologically slow on this
// hardware. It is launched by run_multiproc.sh wrapped in a bounded
// `timeout`, from a separate script-level process, specifically so a real
// hang here cannot hang the test run itself -- never call this binary
// without that wrapper.
//
// IMPORTANT, found during this project's own validation (see the PR
// description): do not trust Work::wait() alone to mean the collective
// genuinely finished. On this hardware, ProcessGroupNCCL::Work::wait() was
// observed to return successfully (and quickly) while the GPU was still
// wedged underneath it -- only a real, separate cudaDeviceSynchronize()
// afterward exposed the hang. This binary therefore follows wait() with
// torch::collectives::bounded_cuda_device_synchronize() (the one utility
// borrowed from this change's topology.h, specifically because it is a
// bounded, abortable primitive and not itself part of the transport
// selection/fallback logic this binary exists to bypass) before declaring
// the collective actually complete.

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/collective_transport/test/test_common.h>
#include <torch/csrc/distributed/c10d/collective_transport/topology.h>

#include <ATen/ATen.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

using torch::collectives::test::make_tcp_store;
using torch::collectives::test::read_proc_env;

int main() {
  auto env = read_proc_env();
  const int64_t numel = std::atoll(
      std::getenv("SHARD_NUMEL") ? std::getenv("SHARD_NUMEL") : "134217728"); // 128M floats = 512MB

  if (std::getenv("NCCL_P2P_DISABLE") != nullptr) {
    std::cerr << "[rank " << env.rank
              << "] refusing to run: NCCL_P2P_DISABLE is set, which defeats the "
                 "point of this repro (it must observe P2P left enabled)"
              << std::endl;
    return 2;
  }

  auto store = make_tcp_store(env);
  auto pg_store = c10::make_intrusive<::c10d::PrefixStore>("repro_vanilla_nccl", store);
  auto pg_options = ::c10d::ProcessGroupNCCL::Options::create();
  auto pg = c10::make_intrusive<::c10d::ProcessGroupNCCL>(
      pg_store, env.rank, env.world_size, pg_options);

  at::Tensor input = at::ones(
      {numel}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, env.rank));
  at::Tensor output = at::empty(
      {numel * env.world_size}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, env.rank));

  std::cout << "[rank " << env.rank << "] constructed vanilla ProcessGroupNCCL, "
            << "NCCL P2P enabled (not disabled), about to call all_gather_single on "
            << (numel * 4) / (1024 * 1024) << "MB per rank" << std::endl;
  std::cout.flush();

  const auto t0 = std::chrono::steady_clock::now();
  auto work = pg->all_gather_single(output, input);
  const bool wait_ok = work->wait(std::chrono::milliseconds(30000));
  const auto wait_elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  if (!wait_ok) {
    std::cout << "[rank " << env.rank << "] all_gather_single Work::wait() did NOT confirm "
              << "completion after " << wait_elapsed_ms << "ms" << std::endl;
    return 1;
  }
  std::cout << "[rank " << env.rank << "] all_gather_single Work::wait() returned after "
            << wait_elapsed_ms << "ms -- NOT yet trusted, verifying with a genuine "
            << "cudaDeviceSynchronize()" << std::endl;
  std::cout.flush();

  const bool sync_ok =
      torch::collectives::bounded_cuda_device_synchronize(env.rank, std::chrono::milliseconds(30000));
  const auto total_elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  if (!sync_ok) {
    std::cout << "[rank " << env.rank
              << "] all_gather_single's Work::wait() reported success, but a genuine "
                 "cudaDeviceSynchronize() afterward did NOT complete within the bounded "
                 "deadline (total elapsed "
              << total_elapsed_ms << "ms) -- Work::wait()'s own completion signal was NOT "
                                     "trustworthy on this hardware"
              << std::endl;
    return 1;
  }

  std::cout << "[rank " << env.rank << "] all_gather_single COMPLETED and VERIFIED in "
            << total_elapsed_ms << "ms" << std::endl;
  return 0;
}
