#pragma once

// Fast path: a thin wrapper around stock ProcessGroupNCCL. Already pure
// C++, already libtorch (PLAN.md row 13 -- a free win, no reason to touch
// it). Used only when topology.h's live probe says real P2P is usable for
// every pair in the process group; see FSDP-DESIGN.md Sec 3.3.
//
// IMPORTANT, found during this project's own validation on IP66 (see the
// PR description): ProcessGroupNCCL::Work::wait() is not sufficient
// evidence, on its own, that a collective genuinely completed on hardware
// in this hazard class -- it was observed to return successfully while the
// GPU was still wedged underneath it; only a following, genuine
// cudaDeviceSynchronize() exposed the hang. That is precisely why
// transport_factory.cpp's topology probe does not trust a throwaway
// collective's Work::wait() alone either (see
// verify_live_nccl_collective's doc comment there) before selecting this
// transport. This class itself still only calls Work::wait() (matching
// stock ProcessGroupNCCL usage elsewhere in this codebase) because by the
// time this class is actually selected and used, the probe has already
// independently verified a real device sync succeeds on this process
// group's hardware -- that verification, not anything added at the call
// site here, is what makes trusting Work::wait() safe in practice.

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/collective_transport/transport.h>

namespace torch::collectives {

class NcclFastTransport : public CollectiveTransport {
 public:
  NcclFastTransport(
      c10::intrusive_ptr<::c10d::ProcessGroupNCCL> pg,
      int rank,
      int world_size)
      : pg_(std::move(pg)), rank_(rank), world_size_(world_size) {}

  void all_gather(const at::Tensor& input_shard, at::Tensor& output) override;
  void reduce_scatter(const at::Tensor& input, at::Tensor& output_shard) override;

  TransportKind kind() const override {
    return TransportKind::kNcclFastPath;
  }
  int rank() const override {
    return rank_;
  }
  int world_size() const override {
    return world_size_;
  }

  // Exposed mainly so tests/diagnostics can confirm this really is just a
  // pass-through to the underlying process group.
  const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& process_group() const {
    return pg_;
  }

 private:
  c10::intrusive_ptr<::c10d::ProcessGroupNCCL> pg_;
  int rank_;
  int world_size_;
};

} // namespace torch::collectives
