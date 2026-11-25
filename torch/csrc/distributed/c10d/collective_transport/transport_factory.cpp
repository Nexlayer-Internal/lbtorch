#include <torch/csrc/distributed/c10d/collective_transport/transport.h>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/collective_transport/host_staged_transport.h>
#include <torch/csrc/distributed/c10d/collective_transport/nccl_fast_transport.h>

#include <c10/util/Logging.h>

#include <sstream>
#include <vector>

namespace torch::collectives {

namespace {

// A throwaway ProcessGroupNCCL whose live-collective probe (see
// verify_live_nccl_collective below) times out is deliberately never torn
// down: FSDP-DESIGN.md Sec 0's incident #2 is a confirmed, real case of an
// explicit ncclCommDestroy() call *hanging in turn* when the underlying
// communicator is already wedged (the defect lived in NCCL's own
// communicator-teardown machinery, not in a missing destroy call). This
// project's own validation found a second, related failure mode in the
// same family: ProcessGroupNCCL's own watchdog thread can independently
// detect the same timeout and call abort()/crash the process on its own
// initiative (observed as a SIGABRT/SIGSEGV during static destruction, see
// the PR description) -- so "leaking" a timed-out probe PG is only a real
// fix if the object's destructor truly never runs, including at process
// exit, not just "not explicitly called". A function-local or namespace-
// scope container (e.g. a static std::vector) is NOT enough: its own
// destructor still runs during normal static-destruction at process exit,
// which would run ~ProcessGroupNCCL() anyway and hit the exact same
// hazard. Instead, heap-allocate the holder itself with `new` and never
// delete it, so nothing ever decrements the intrusive_ptr's refcount to
// zero and ~ProcessGroupNCCL() genuinely never runs, not even at exit.
void leak_forever(c10::intrusive_ptr<::c10d::ProcessGroupNCCL> pg) {
  auto* leaked = new c10::intrusive_ptr<::c10d::ProcessGroupNCCL>(std::move(pg));
  (void)leaked; // never deleted, on purpose -- see comment above
}

// The real question this whole layer exists to answer is not "can these
// two devices do a raw peer-to-peer memcpy" (probe_peer_pair's question)
// but "will a real ProcessGroupNCCL collective on these devices actually
// finish" -- and this project's own validation found those two questions
// have *different answers* on IP66: probe_peer_pair's raw
// cudaMemcpyPeerAsync completes fine, but a real all_gather_single's
// ProcessGroupNCCL::Work::wait() can report success while the GPU is still
// wedged underneath it -- confirmed by following wait() with a genuine
// cudaDeviceSynchronize(), which then hung indefinitely (see the PR
// description for the full repro). Work::wait()'s own signal is therefore
// not trustworthy evidence on this hardware class, which is exactly the
// "static/simple check can lie" lesson FSDP-DESIGN.md Sec 3.4 already
// warned about -- it turns out to apply to NCCL's *own* completion
// signaling, not only to nvidia-smi/cudaDeviceCanAccessPeer.
//
// So: construct a throwaway, group-wide ProcessGroupNCCL (this is
// necessarily a whole-group operation, not a per-pair one -- NCCL
// communicators are not pairwise), run one tiny real all_gather_single,
// call work->wait(), and then do NOT trust that alone -- require a
// bounded, backgrounded cudaDeviceSynchronize() (topology.h) to also
// complete before concluding the fast path is safe.
bool verify_live_nccl_collective(
    int rank,
    int world_size,
    int device_id,
    const c10::intrusive_ptr<::c10d::Store>& store,
    std::chrono::milliseconds timeout) {
  auto pg_store =
      c10::make_intrusive<::c10d::PrefixStore>("lbtorch/collective_transport/probe_nccl", store);
  auto pg_options = ::c10d::ProcessGroupNCCL::Options::create();
  auto pg = c10::make_intrusive<::c10d::ProcessGroupNCCL>(pg_store, rank, world_size, pg_options);

  constexpr int64_t kProbeNumel = 256;
  at::Tensor in = at::ones({kProbeNumel}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_id));
  at::Tensor out = at::empty(
      {kProbeNumel * world_size}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_id));

  bool wait_ok = false;
  try {
    auto work = pg->all_gather_single(out, in);
    wait_ok = work->wait(timeout);
  } catch (const std::exception& e) {
    LOG(INFO) << "[collective_transport] rank=" << rank
              << " live NCCL collective probe threw during enqueue/wait: " << e.what();
    leak_forever(std::move(pg));
    return false;
  }
  if (!wait_ok) {
    LOG(INFO) << "[collective_transport] rank=" << rank
              << " live NCCL collective probe: Work::wait() itself did not confirm completion";
    leak_forever(std::move(pg));
    return false;
  }

  // The check that actually matters: do not trust Work::wait()'s own
  // signal (see the comment above this function) -- require a real,
  // independently-bounded device sync too.
  const bool sync_ok = bounded_cuda_device_synchronize(device_id, timeout);
  if (!sync_ok) {
    LOG(INFO) << "[collective_transport] rank=" << rank
              << " live NCCL collective probe: Work::wait() returned but a genuine "
                 "cudaDeviceSynchronize() afterward did not complete within "
              << timeout.count()
              << "ms -- Work::wait()'s completion signal is not trustworthy on this hardware";
    leak_forever(std::move(pg));
    return false;
  }

  LOG(INFO) << "[collective_transport] rank=" << rank
            << " live NCCL collective probe passed (enqueue + wait + genuine device sync all "
               "completed within the bounded deadline)";
  // Success case only: ordinary teardown is safe here because we just
  // proved, via a real device sync, that this communicator is not wedged.
  return true;
}

// Every rank probes every pair that includes its own device against every
// other rank's device (single-node scope: all devices assumed visible to
// every process on this node -- see transport.h's doc comment and
// FSDP-DESIGN.md Sec 2 point 1) as a cheap first-pass filter, then --
// only if every pair passes that filter -- every rank jointly runs
// verify_live_nccl_collective above, the check that actually matters.
// Every rank's combined local verdict is published and read back via the
// Store so the whole group reaches one consensus decision. One bad pair
// or a failed live-collective probe falls the *whole* group back to the
// host-staged transport -- transport selection is per process-group, per
// FSDP-DESIGN.md Sec 3, not per-pair.
bool probe_group_p2p_usable(
    int rank,
    int world_size,
    int device_id,
    const c10::intrusive_ptr<::c10d::Store>& store,
    const ProbeOptions& probe_opts) {
  bool local_ok = true;
  for (int peer = 0; peer < world_size; ++peer) {
    if (peer == rank) {
      continue;
    }
    PeerProbeResult result = probe_peer_pair(device_id, peer, probe_opts);
    LOG(INFO) << "[collective_transport] topology probe rank=" << rank
              << " device=" << device_id << " peer_device=" << peer
              << " static_can_access_peer=" << result.static_can_access_peer
              << " live_probe_completed=" << result.live_probe_completed
              << " live_probe_timed_out=" << result.live_probe_timed_out
              << " live_probe_error=" << result.live_probe_error
              << " wall_time_ms=" << result.probe_wall_time_ms
              << " detail=\"" << result.detail << "\"";
    if (!result.p2p_usable()) {
      local_ok = false;
    }
  }

  // The raw pairwise probe is necessary but, as this project's own
  // validation found, not sufficient (see verify_live_nccl_collective's
  // doc comment): only run the live-collective check if the cheap filter
  // above passed for every pair, but require it too before trusting the
  // fast path.
  //
  // NOTE: this is itself a collective operation -- every rank must call it
  // together, in the same order every time, or the Store-based rendezvous
  // inside it (and the subsequent verdict publication below) will
  // deadlock. A rank whose own pairwise probes failed still participates
  // (so ranks that passed are not left waiting on a peer that silently
  // skipped the group operation), but ignores the result.
  const bool live_ok =
      verify_live_nccl_collective(rank, world_size, device_id, store, probe_opts.timeout) &&
      local_ok;

  const std::string key_prefix = "lbtorch/collective_transport/topology_verdict/";
  std::vector<uint8_t> verdict_bytes{static_cast<uint8_t>(live_ok ? 1 : 0)};
  store->set(key_prefix + std::to_string(rank), verdict_bytes);

  std::vector<std::string> keys;
  keys.reserve(world_size);
  for (int r = 0; r < world_size; ++r) {
    keys.push_back(key_prefix + std::to_string(r));
  }
  // Bounded by the probe deadline plus slack for every other rank to reach
  // and publish its own verdict -- not an unbounded wait.
  store->wait(keys, probe_opts.timeout * 2 + std::chrono::milliseconds(10000));

  bool group_ok = true;
  for (int r = 0; r < world_size; ++r) {
    auto v = store->get(key_prefix + std::to_string(r));
    if (v.empty() || v[0] == 0) {
      group_ok = false;
    }
  }
  return group_ok;
}

} // namespace

std::shared_ptr<CollectiveTransport> make_transport(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int world_size,
    int device_id,
    size_t slot_capacity_bytes,
    const TransportOptions& options) {
  TransportKind chosen;
  if (options.force_transport.has_value()) {
    chosen = *options.force_transport;
    LOG(INFO) << "[collective_transport] rank=" << rank
              << " force_transport set, skipping topology probe, kind="
              << (chosen == TransportKind::kNcclFastPath ? "nccl_fast_path"
                                                          : "host_staged_fallback");
  } else {
    const bool fast_ok =
        probe_group_p2p_usable(rank, world_size, device_id, store, options.probe_options);
    chosen = fast_ok ? TransportKind::kNcclFastPath : TransportKind::kHostStagedFallback;
    LOG(INFO) << "[collective_transport] rank=" << rank
              << " topology probe verdict: group_p2p_usable=" << fast_ok
              << " -> selecting "
              << (fast_ok ? "nccl_fast_path" : "host_staged_fallback");
  }

  if (chosen == TransportKind::kNcclFastPath) {
    auto pg_store =
        c10::make_intrusive<::c10d::PrefixStore>("lbtorch/collective_transport/nccl", store);
    auto pg_options = ::c10d::ProcessGroupNCCL::Options::create();
    auto pg = c10::make_intrusive<::c10d::ProcessGroupNCCL>(pg_store, rank, world_size, pg_options);
    return std::make_shared<NcclFastTransport>(std::move(pg), rank, world_size);
  }

  return std::make_shared<HostStagedTransport>(
      std::move(store), rank, world_size, device_id, slot_capacity_bytes, options.barrier_timeout);
}

} // namespace torch::collectives
