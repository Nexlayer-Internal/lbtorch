#pragma once

// Fallback N-rank collective transport, used when topology.h's live probe
// says real P2P is not usable between this process group's GPUs.
//
// This adopts the *mechanism and safety philosophy* of Nexlayer's
// liquid-brain-native (lbx) `direct_collectives` -- pinned-host-memory
// staging instead of cudaMemcpyPeerAsync/direct P2P DMA, bounded+abortable
// synchronization instead of an unbounded spin-wait, no persistent
// communicator object with its own teardown protocol to hang in -- without
// reusing any of its code, and generalizes well past its actual shape:
//   * arbitrary rank count N, not hardcoded to 2.
//   * a genuine multi-process-per-GPU deployment (one process per rank, as
//     real FSDP training runs), via POSIX shared memory (shm_open/mmap) +
//     Store-based rendezvous -- not lbx's single process driving every
//     device's CUDA context directly.
//   * real all-gather / reduce-scatter semantics, not an all-reduce
//     adapted after the fact.
//   * at least fp32 and bf16 (in fact any ATen dtype at all, via ATen's own
//     tensor ops for the host-side reduction -- see the .cpp), not a
//     bf16-only kernel.
// See FSDP-DESIGN.md Sec 2-3 for the full design rationale, and Sec 2's
// verdict for exactly what was and was not judged reusable from
// direct_collectives' actual code (nothing; the mechanism is reimplemented
// from scratch against this layer's different process/rendezvous model).
//
// Wire protocol per collective call, at a high level:
//   1. Each rank stages its own payload into its own slot of a shared,
//      pinned (cudaHostRegister'd) staging region via a plain
//      cudaMemcpyAsync (device -> host, or host -> device on the way out;
//      never device -> device). Completion of that copy is confirmed by
//      polling cudaEventQuery on a bounded wall-clock deadline -- never a
//      blocking cudaStreamSynchronize/cudaEventSynchronize.
//   2. Each rank publishes an epoch-tagged arrival flag (a plain atomic
//      store into the same shared memory) once its own copy-in is
//      confirmed complete, then waits (bounded, abortable, polling) for
//      every other rank's arrival flag for this call's epoch.
//   3. Once every rank has arrived, each rank reads out of the shared
//      staging region (all-gather: copies every slot into its output, in
//      rank order; reduce-scatter: sums, in host memory via ATen tensor
//      ops, only the elements belonging to its own output shard) and
//      copies the result back to its own device.
// There are two physical staging slots (kNumStagingSlots), alternated per
// call index, so a future caller could begin staging call N+1 while call
// N's barrier is still draining -- this phase's all_gather/reduce_scatter
// are themselves synchronous (block until their own call completes), so
// that overlap is not yet exposed at this API level; exposing it is
// explicitly FSDP orchestration's job (FSDP-DESIGN.md Sec 3.1, a separate,
// later task), not this transport layer's.
//
// Known, documented scope limits for this phase (see FSDP-DESIGN.md Sec 2
// for why each is called out explicitly rather than silently assumed):
//   * Every rank's shard/slice must be equal-sized (uneven last-shard
//     sharding is not handled).
//   * Single-node only: `device_id` values across ranks must all be
//     visible in every participating process (this is also topology.h's
//     scope).

#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/collective_transport/transport.h>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace torch::collectives {

constexpr int kNumStagingSlots = 2;

class HostStagedTransport : public CollectiveTransport {
 public:
  HostStagedTransport(
      c10::intrusive_ptr<::c10d::Store> store,
      int rank,
      int world_size,
      int device_id,
      size_t slot_capacity_bytes,
      std::chrono::milliseconds barrier_timeout = std::chrono::milliseconds(30000));

  ~HostStagedTransport() override;

  HostStagedTransport(const HostStagedTransport&) = delete;
  HostStagedTransport& operator=(const HostStagedTransport&) = delete;

  void all_gather(const at::Tensor& input_shard, at::Tensor& output) override;
  void reduce_scatter(const at::Tensor& input, at::Tensor& output_shard) override;
  void abort() override;

  TransportKind kind() const override {
    return TransportKind::kHostStagedFallback;
  }
  int rank() const override {
    return rank_;
  }
  int world_size() const override {
    return world_size_;
  }

 public:
  struct FixedHeader; // layout detail, defined in the .cpp (public only so
                       // the .cpp's free functions can name the type; the
                       // type itself is still useless outside the .cpp
                       // since its definition is private to that TU)

 private:
  FixedHeader* header() const;
  char* slot_ptr(int slot_index, int for_rank) const;
  std::atomic<uint64_t>* epoch_ptr(int slot_index, int for_rank) const;

  void wait_arrival(int slot_index, uint64_t expected_epoch);
  void publish_arrival(int slot_index, uint64_t epoch);
  // Bounded poll of a single cudaEvent; throws TransportTimeoutError /
  // TransportAbortedError on deadline / abort rather than blocking.
  void bounded_event_wait(cudaEvent_t event);

  c10::intrusive_ptr<::c10d::Store> store_;
  int rank_;
  int world_size_;
  int device_id_;
  size_t slot_capacity_bytes_;
  std::chrono::milliseconds barrier_timeout_;

  std::string shm_name_;
  int shm_fd_ = -1;
  void* shm_base_ = nullptr;
  size_t shm_size_ = 0;
  uint64_t call_counter_ = 0;

  cudaStream_t stream_ = nullptr;
};

} // namespace torch::collectives
