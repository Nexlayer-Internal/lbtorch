#pragma once

// Live, bounded topology probing for peer-to-peer GPU access.
//
// Why this exists, and why it is not just a static cudaDeviceCanAccessPeer()
// query: see FSDP-DESIGN.md Sec 0 / Sec 3.4. On IP66 (2x RTX PRO 6000
// Blackwell, SYS topology, no NVLink, cross-NUMA/cross-CPU-socket), both
// `nvidia-smi topo -p2p r` and NCCL's own internal P2P detection
// (`isAllDirectP2p 1 directMode 1`) reported P2P as usable, and
// `ncclAllReduce` hung indefinitely with P2P enabled anyway. A driver-level
// capability query can lie about whether P2P actually works at runtime on a
// given piece of hardware. The only trustworthy signal is an actual,
// bounded, abortable attempt to move data over the path in question.

#include <chrono>
#include <cstddef>
#include <string>

namespace torch::collectives {

// Result of probing one ordered device pair.
struct PeerProbeResult {
  int device_a = -1;
  int device_b = -1;

  // cudaDeviceCanAccessPeer() answer, and whether
  // cudaDeviceEnablePeerAccess() succeeded in both directions. This is the
  // part of the story that has been observed, on this exact hardware, to
  // report "fine" while the real transfer hangs -- see above.
  bool static_can_access_peer = false;
  bool peer_access_enabled = false;

  // Did an actual P2P-enabled data transfer complete within the deadline?
  bool live_probe_completed = false;
  bool live_probe_timed_out = false;
  bool live_probe_error = false;
  std::string detail;

  double probe_wall_time_ms = 0.0;

  // The only field callers should branch on: true only if every check,
  // static and live, passed within the bounded deadline.
  bool p2p_usable() const {
    return static_can_access_peer && live_probe_completed &&
        !live_probe_timed_out && !live_probe_error;
  }
};

struct ProbeOptions {
  // Wall-clock deadline for the live probe transfer. This bound is
  // enforced by polling (cudaEventQuery), never by a blocking CUDA sync
  // call, so the calling thread is itself the "host thread with a hard
  // wall-clock timeout, independent of CUDA's own synchronization"
  // FSDP-DESIGN.md Sec 3.4 calls for.
  std::chrono::milliseconds timeout{3000};

  // Size of the bounded probe transfer. Deliberately large enough to
  // exercise the real DMA/transport path (a toy few-byte handshake can
  // "complete" over a degenerate code path even when the real collective
  // hangs), small enough to probe quickly at process-group init time.
  size_t probe_bytes = 64 * 1024 * 1024;
};

// Probes whether a P2P-enabled transfer between `device_a` and `device_b`
// is actually usable, not just reported as available:
//   1. cudaDeviceCanAccessPeer() + cudaDeviceEnablePeerAccess() (the static,
//      "trust the driver" check).
//   2. If (1) passes, actually enqueue a cudaMemcpyPeerAsync of
//      `opts.probe_bytes` and poll for completion on a bounded wall-clock
//      deadline. Concludes P2P is usable only if the transfer genuinely
//      completes in time.
//
// Both devices must be visible to the calling process (this is a
// single-node, local-pair probe; it says nothing about cross-node
// reachability, which is NCCL's network transport's concern, not this
// layer's).
//
// On timeout, this function deliberately does not attempt to synchronize
// or free the in-flight stream/event/buffers: a blocking "clean up
// properly" call could itself hang, which is exactly the failure mode this
// probe exists to avoid. The cost is a few MB of leaked device memory and
// two handles, paid only in the (already-anomalous) failure case.
PeerProbeResult probe_peer_pair(
    int device_a,
    int device_b,
    const ProbeOptions& opts = ProbeOptions());

// General bounded synchronization primitive, factored out for reuse beyond
// the raw-memcpy probe above (see transport_factory.cpp's live-collective
// verification stage, and its doc comment for why this exists): calls
// cudaDeviceSynchronize() on `device_id`, on a background thread, with a
// wall-clock deadline enforced by the *calling* thread.
// cudaDeviceSynchronize() itself takes no timeout and can genuinely never
// return on this hardware class -- empirically confirmed during this
// project's own validation (see the PR description): ProcessGroupNCCL's
// Work::wait() returned successfully (reporting the collective complete)
// while the GPU itself was still wedged, and only a real
// cudaDeviceSynchronize() afterward exposed it, hanging indefinitely. On
// timeout, the background thread is deliberately left detached (never
// joined) rather than risking the caller blocking forever trying to clean
// it up -- the same leak-on-timeout tradeoff documented above for
// probe_peer_pair.
bool bounded_cuda_device_synchronize(int device_id, std::chrono::milliseconds timeout);

} // namespace torch::collectives
