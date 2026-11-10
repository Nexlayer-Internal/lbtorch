#pragma once

// Transport-agnostic N-rank collective interface for the (separate, later)
// C++ FSDP replacement to call, per FSDP-DESIGN.md Sec 3.2. Nothing in this
// header or anything it depends on is wired into FSDP in this change --
// this is standalone transport infrastructure, built and validated on its
// own merits (FSDP-DESIGN.md Sec 4, "Phase 0").
//
// Two concrete transports implement this interface:
//   - NcclFastTransport (nccl_fast_transport.h): a thin wrapper around
//     stock ProcessGroupNCCL, used when the topology probe says real P2P
//     is usable.
//   - HostStagedTransport (host_staged_transport.h): a from-scratch,
//     N-rank, multi-dtype, pinned-host-memory-staged fallback, used
//     otherwise. See its header for the full design rationale.
//
// `make_transport()` below picks between them automatically per
// topology.h's live, bounded P2P probe.

#include <ATen/core/Tensor.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/collective_transport/topology.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>

namespace torch::collectives {

enum class TransportKind { kNcclFastPath, kHostStagedFallback };

// Base for every error this layer raises. The whole point of the bounded-
// wait discipline throughout this library (FSDP-DESIGN.md Sec 3.3 point 2)
// is that a hang becomes one of these clean, catchable errors instead of an
// indefinite freeze.
struct TransportError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct TransportTimeoutError : public TransportError {
  using TransportError::TransportError;
};
struct TransportAbortedError : public TransportError {
  using TransportError::TransportError;
};

struct TransportOptions {
  ProbeOptions probe_options{};
  // Bound on every blocking wait inside the host-staged fallback (peer
  // arrival barriers, device-copy completion polls, and the Store-based
  // init rendezvous). Irrelevant to the NCCL fast path, which uses
  // ProcessGroupNCCL's own existing watchdog/timeout mechanism.
  std::chrono::milliseconds barrier_timeout{30000};
  // Test/diagnostic override: bypass the live topology probe entirely and
  // force a specific transport. Never used for production selection.
  std::optional<TransportKind> force_transport{};
};

class CollectiveTransport {
 public:
  virtual ~CollectiveTransport() = default;

  // Every rank contributes `input_shard` (identical shape/dtype on every
  // rank); `output` receives all ranks' shards concatenated in rank order,
  // identically on every rank. `output.numel() == input_shard.numel() *
  // world_size()` is required.
  virtual void all_gather(const at::Tensor& input_shard, at::Tensor& output) = 0;

  // Every rank contributes `input` (identical shape/dtype on every rank);
  // `output_shard` receives this rank's 1/world_size() slice of the
  // element-wise sum of every rank's `input`. `input.numel() ==
  // output_shard.numel() * world_size()` is required.
  virtual void reduce_scatter(const at::Tensor& input, at::Tensor& output_shard) = 0;

  // Unblocks any in-flight (or future) call on this transport instance,
  // causing it to fail with TransportAbortedError instead of continuing to
  // wait. No-op on transports (the NCCL fast path) that have no bounded
  // internal wait of their own to abort.
  virtual void abort() {}

  virtual TransportKind kind() const = 0;
  virtual int rank() const = 0;
  virtual int world_size() const = 0;
};

// Probes topology for every local rank pair and selects + constructs the
// transport for this process: the NCCL fast path if every pair's live,
// bounded P2P probe succeeds, the host-staged fallback otherwise (or
// whichever transport `options.force_transport` names, bypassing the probe
// entirely -- intended for tests).
//
// `store` is used both for the topology-verdict consensus rendezvous (every
// rank's local probe result is published and read back, so one bad pair
// falls the *whole* group back rather than only that pair -- transport
// selection is per process-group, per FSDP-DESIGN.md Sec 3) and, depending
// on which transport is chosen, for that transport's own rendezvous
// (ProcessGroupNCCL's usual store-based bootstrap, or the host-staged
// transport's shared-memory-segment-name exchange).
//
// `device_id` is the CUDA device this rank owns. All ranks' devices are
// assumed visible to every process on the same node (this phase's scope is
// single-node, matching IP66 and matching FSDP's per-node sharding groups;
// see FSDP-DESIGN.md Sec 2 point 1 on why a genuine multi-node topology
// model is future work).
//
// `slot_capacity_bytes` upper-bounds any single all_gather/reduce_scatter
// call's per-rank payload for the lifetime of the returned transport, when
// the host-staged fallback is selected (mirrors direct_collectives' fixed-
// at-bootstrap `slot_capacity`; irrelevant to the NCCL fast path).
std::shared_ptr<CollectiveTransport> make_transport(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int world_size,
    int device_id,
    size_t slot_capacity_bytes,
    const TransportOptions& options = TransportOptions());

} // namespace torch::collectives
