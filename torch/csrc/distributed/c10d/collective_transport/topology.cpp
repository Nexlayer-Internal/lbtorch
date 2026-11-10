#include <torch/csrc/distributed/c10d/collective_transport/topology.h>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

namespace torch::collectives {

PeerProbeResult probe_peer_pair(
    int device_a,
    int device_b,
    const ProbeOptions& opts) {
  PeerProbeResult result;
  result.device_a = device_a;
  result.device_b = device_b;
  const auto t_start = std::chrono::steady_clock::now();

  auto elapsed_ms = [&]() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t_start)
        .count();
  };

  int can_access = 0;
  cudaError_t err = cudaDeviceCanAccessPeer(&can_access, device_a, device_b);
  if (err != cudaSuccess) {
    result.detail =
        std::string("cudaDeviceCanAccessPeer failed: ") + cudaGetErrorString(err);
    result.live_probe_error = true;
    result.probe_wall_time_ms = elapsed_ms();
    return result;
  }
  result.static_can_access_peer = (can_access != 0);

  if (!result.static_can_access_peer) {
    result.detail =
        "cudaDeviceCanAccessPeer reports no peer access; not attempting live probe";
    result.probe_wall_time_ms = elapsed_ms();
    return result;
  }

  // Enable peer access in both directions. Already-enabled is not an error
  // (e.g. a previous probe of the same pair already did this).
  cudaSetDevice(device_a);
  cudaError_t e1 = cudaDeviceEnablePeerAccess(device_b, 0);
  const bool ok1 = (e1 == cudaSuccess || e1 == cudaErrorPeerAccessAlreadyEnabled);
  cudaSetDevice(device_b);
  cudaError_t e2 = cudaDeviceEnablePeerAccess(device_a, 0);
  const bool ok2 = (e2 == cudaSuccess || e2 == cudaErrorPeerAccessAlreadyEnabled);
  result.peer_access_enabled = ok1 && ok2;

  if (!result.peer_access_enabled) {
    result.detail = "cudaDeviceEnablePeerAccess failed in at least one direction";
    result.live_probe_error = true;
    result.probe_wall_time_ms = elapsed_ms();
    return result;
  }

  // The live, bounded correctness probe -- the part that actually matters
  // per FSDP-DESIGN.md Sec 3.4. Attempt a real P2P-enabled transfer and
  // bound the wait by wall clock, polling cudaEventQuery (never a blocking
  // sync call), so a real hang here becomes a clean, observed timeout
  // instead of a frozen process.
  void* buf_a = nullptr;
  void* buf_b = nullptr;

  cudaSetDevice(device_a);
  if (cudaMalloc(&buf_a, opts.probe_bytes) != cudaSuccess) {
    result.detail = "cudaMalloc on device_a failed during probe";
    result.live_probe_error = true;
    result.probe_wall_time_ms = elapsed_ms();
    return result;
  }
  cudaSetDevice(device_b);
  if (cudaMalloc(&buf_b, opts.probe_bytes) != cudaSuccess) {
    cudaSetDevice(device_a);
    cudaFree(buf_a);
    result.detail = "cudaMalloc on device_b failed during probe";
    result.live_probe_error = true;
    result.probe_wall_time_ms = elapsed_ms();
    return result;
  }

  cudaSetDevice(device_a);
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

  cudaError_t cpy_err =
      cudaMemcpyPeerAsync(buf_b, device_b, buf_a, device_a, opts.probe_bytes, stream);
  if (cpy_err != cudaSuccess) {
    result.detail =
        std::string("cudaMemcpyPeerAsync enqueue failed: ") + cudaGetErrorString(cpy_err);
    result.live_probe_error = true;
  } else {
    cudaEventRecord(event, stream);
    const auto deadline = t_start + opts.timeout;
    while (true) {
      cudaError_t q = cudaEventQuery(event);
      if (q == cudaSuccess) {
        result.live_probe_completed = true;
        break;
      }
      if (q != cudaErrorNotReady) {
        result.detail =
            std::string("cudaEventQuery failed during probe: ") + cudaGetErrorString(q);
        result.live_probe_error = true;
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        result.live_probe_timed_out = true;
        result.detail =
            "P2P probe transfer did not complete within the bounded deadline ("
            + std::to_string(opts.timeout.count())
            + "ms). This is exactly the IP66 hazard: a static topology query "
              "can report P2P as usable while the real transfer hangs.";
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }

  // Deliberately do not synchronize or free resources on timeout: see the
  // header comment. The GPU/driver state behind a timed-out transfer is
  // suspect, and a "clean up properly" blocking call could itself hang.
  if (!result.live_probe_timed_out) {
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    cudaFree(buf_b);
    cudaSetDevice(device_a);
    cudaFree(buf_a);
  }

  result.probe_wall_time_ms = elapsed_ms();
  return result;
}

bool bounded_cuda_device_synchronize(int device_id, std::chrono::milliseconds timeout) {
  auto done = std::make_shared<std::atomic<bool>>(false);
  auto sync_ok = std::make_shared<std::atomic<bool>>(false);

  std::thread worker([device_id, done, sync_ok]() {
    if (cudaSetDevice(device_id) != cudaSuccess) {
      done->store(true, std::memory_order_release);
      return;
    }
    const bool ok = (cudaDeviceSynchronize() == cudaSuccess);
    sync_ok->store(ok, std::memory_order_release);
    done->store(true, std::memory_order_release);
  });
  worker.detach();

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!done->load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false; // timed out; the detached thread is deliberately leaked
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  return sync_ok->load(std::memory_order_acquire);
}

} // namespace torch::collectives
