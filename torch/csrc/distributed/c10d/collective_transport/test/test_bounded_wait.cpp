// Proves the host-staged fallback's core safety property: a stalled peer
// turns into a *bounded, diagnosable error*, never an indefinite freeze --
// FSDP-DESIGN.md Sec 0's whole point, and the property that would have
// prevented IP66's original flat_param_to deadlock.
//
// Two ranks; rank 1 deliberately never calls the collective op rank 0
// calls (simulating a stalled/crashed peer). Two modes:
//   MODE=timeout : rank 0's wait_arrival() should throw
//                  TransportTimeoutError once BARRIER_TIMEOUT_MS elapses.
//   MODE=abort   : rank 0 aborts its own transport from a second thread
//                  shortly after issuing the call; wait_arrival() should
//                  throw TransportAbortedError almost immediately, well
//                  before BARRIER_TIMEOUT_MS would otherwise elapse.
// In both modes, the whole process (both ranks) must exit within a few
// seconds of BARRIER_TIMEOUT_MS -- run_multiproc.sh additionally wraps the
// run in `timeout` as a backstop, but the point of this test is that the
// backstop should never actually be needed.

#include <torch/csrc/distributed/c10d/collective_transport/host_staged_transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/test/test_common.h>

#include <ATen/ATen.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

using torch::collectives::CollectiveTransport;
using torch::collectives::make_transport;
using torch::collectives::TransportAbortedError;
using torch::collectives::TransportKind;
using torch::collectives::TransportOptions;
using torch::collectives::TransportTimeoutError;
using torch::collectives::test::make_tcp_store;
using torch::collectives::test::read_proc_env;

int main() {
  auto env = read_proc_env();
  const char* mode_s = std::getenv("MODE");
  const std::string mode = mode_s ? mode_s : "timeout";
  const int64_t barrier_timeout_ms = std::atoll(
      std::getenv("BARRIER_TIMEOUT_MS") ? std::getenv("BARRIER_TIMEOUT_MS") : "3000");

  auto store = make_tcp_store(env);

  TransportOptions options;
  options.force_transport = TransportKind::kHostStagedFallback;
  options.barrier_timeout = std::chrono::milliseconds(barrier_timeout_ms);

  const int64_t shard_numel = 1024;
  const size_t slot_capacity_bytes = static_cast<size_t>(shard_numel) * env.world_size * sizeof(float);

  auto transport = make_transport(store, env.rank, env.world_size, 0, slot_capacity_bytes, options);

  if (env.rank == 1) {
    // Simulate a stalled peer: never call the collective. Sleep long
    // enough for rank 0's wait to resolve (one way or the other), then
    // exit on its own -- this test is specifically about rank 0 failing
    // *cleanly and quickly*, not about actually hanging anything.
    std::this_thread::sleep_for(std::chrono::milliseconds(barrier_timeout_ms + 2000));
    std::cout << "[rank 1] exiting without ever calling the collective (simulated stall)"
              << std::endl;
    return 0;
  }

  // rank 0
  at::Tensor input = at::zeros({shard_numel}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor output =
      at::zeros({shard_numel * env.world_size}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  if (mode == "abort") {
    std::thread aborter([&]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      std::cout << "[rank 0] aborting transport after 500ms" << std::endl;
      transport->abort();
    });

    const auto t0 = std::chrono::steady_clock::now();
    bool got_expected_error = false;
    try {
      transport->all_gather(input, output);
    } catch (const TransportAbortedError& e) {
      got_expected_error = true;
      std::cout << "[rank 0] caught expected TransportAbortedError: " << e.what() << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "[rank 0] FAIL: wrong exception type: " << e.what() << std::endl;
    }
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    aborter.join();
    std::cout << "[rank 0] all_gather returned via exception after " << elapsed_ms << "ms"
              << " (barrier_timeout=" << barrier_timeout_ms << "ms)" << std::endl;

    // The abort must have short-circuited the wait well before the full
    // timeout -- otherwise this is indistinguishable from an ordinary
    // timeout and the abort mechanism isn't doing anything.
    if (got_expected_error && elapsed_ms < static_cast<double>(barrier_timeout_ms) * 0.8) {
      std::cout << "[rank 0] PASS" << std::endl;
      return 0;
    }
    std::cout << "[rank 0] FAIL" << std::endl;
    return 1;
  } else { // mode == "timeout"
    const auto t0 = std::chrono::steady_clock::now();
    bool got_expected_error = false;
    try {
      transport->all_gather(input, output);
    } catch (const TransportTimeoutError& e) {
      got_expected_error = true;
      std::cout << "[rank 0] caught expected TransportTimeoutError: " << e.what() << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "[rank 0] FAIL: wrong exception type: " << e.what() << std::endl;
    }
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "[rank 0] all_gather returned via exception after " << elapsed_ms << "ms"
              << " (barrier_timeout=" << barrier_timeout_ms << "ms)" << std::endl;

    // Bounded: must fail at roughly barrier_timeout_ms, not hang
    // indefinitely (generous upper bound to absorb scheduling jitter) and
    // not suspiciously instantly either (would indicate the wait isn't
    // actually happening).
    if (got_expected_error && elapsed_ms >= static_cast<double>(barrier_timeout_ms) * 0.8 &&
        elapsed_ms < static_cast<double>(barrier_timeout_ms) * 3.0) {
      std::cout << "[rank 0] PASS" << std::endl;
      return 0;
    }
    std::cout << "[rank 0] FAIL" << std::endl;
    return 1;
  }
}
