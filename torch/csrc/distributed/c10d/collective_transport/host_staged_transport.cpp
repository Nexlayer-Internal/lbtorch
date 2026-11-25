#include <torch/csrc/distributed/c10d/collective_transport/host_staged_transport.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

namespace torch::collectives {

namespace {

constexpr uint64_t kShmMagic = 0x4C42544F52434854ULL; // "LBTORCHT"

size_t align_up(size_t v, size_t a) {
  return (v + a - 1) / a * a;
}

} // namespace

// Shared-memory header, placed at the start of the mapping. Followed by an
// array of world_size * kNumStagingSlots epoch counters, then the data
// region (world_size * kNumStagingSlots slots of slot_capacity_bytes each).
// This whole struct plays the role lbx's small `arrived` flags array +
// `aborted` flag word play in direct_collectives, generalized from a
// fixed 2-rank array to an N-rank, 2-slot one.
struct HostStagedTransport::FixedHeader {
  std::atomic<uint64_t> magic;
  std::atomic<uint32_t> abort_flag;
  std::atomic<uint32_t> ready_world_size; // sanity check only
};

namespace {
size_t epoch_array_offset() {
  return align_up(sizeof(HostStagedTransport::FixedHeader), alignof(std::atomic<uint64_t>));
}
size_t epoch_array_bytes(int world_size) {
  return sizeof(std::atomic<uint64_t>) * static_cast<size_t>(world_size) * kNumStagingSlots;
}
size_t data_region_offset(int world_size) {
  return align_up(epoch_array_offset() + epoch_array_bytes(world_size), 4096);
}
size_t total_shm_bytes(int world_size, size_t slot_capacity_bytes) {
  return data_region_offset(world_size) +
      static_cast<size_t>(world_size) * kNumStagingSlots * slot_capacity_bytes;
}
} // namespace

HostStagedTransport::FixedHeader* HostStagedTransport::header() const {
  return reinterpret_cast<FixedHeader*>(shm_base_);
}

std::atomic<uint64_t>* HostStagedTransport::epoch_ptr(int slot_index, int for_rank) const {
  auto* arr = reinterpret_cast<std::atomic<uint64_t>*>(
      static_cast<char*>(shm_base_) + epoch_array_offset());
  return &arr[static_cast<size_t>(slot_index) * world_size_ + for_rank];
}

char* HostStagedTransport::slot_ptr(int slot_index, int for_rank) const {
  const size_t off = data_region_offset(world_size_) +
      (static_cast<size_t>(slot_index) * world_size_ + for_rank) * slot_capacity_bytes_;
  return static_cast<char*>(shm_base_) + off;
}

HostStagedTransport::HostStagedTransport(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int world_size,
    int device_id,
    size_t slot_capacity_bytes,
    std::chrono::milliseconds barrier_timeout)
    : store_(std::move(store)),
      rank_(rank),
      world_size_(world_size),
      device_id_(device_id),
      slot_capacity_bytes_(slot_capacity_bytes),
      barrier_timeout_(barrier_timeout) {
  TORCH_CHECK(world_size_ >= 1, "world_size must be >= 1");
  TORCH_CHECK(slot_capacity_bytes_ > 0, "slot_capacity_bytes must be > 0");

  shm_size_ = total_shm_bytes(world_size_, slot_capacity_bytes_);

  // Rendezvous: rank 0 creates the POSIX shared-memory segment and
  // publishes its name via the Store; every other rank reads that name and
  // attaches to the existing segment. This is this layer's multi-process
  // equivalent of direct_collectives' single-process cudaHostAlloc -- see
  // FSDP-DESIGN.md Sec 3.3's explicit call for "a POSIX shared-memory
  // segment per node (shm_open/mmap) ... plus the normal c10d-style
  // rendezvous (store-based rank/address exchange)".
  const std::string name_key = "lbtorch/collective_transport/host_staged/shm_name";
  if (rank_ == 0) {
    std::ostringstream oss;
    oss << "/lbtorch_ct_" << ::getpid() << "_" << reinterpret_cast<uintptr_t>(this);
    shm_name_ = oss.str();
    shm_fd_ = ::shm_open(shm_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    TORCH_CHECK(
        shm_fd_ >= 0,
        "shm_open (create) failed for ",
        shm_name_,
        ": ",
        std::strerror(errno));
    TORCH_CHECK(
        ::ftruncate(shm_fd_, static_cast<off_t>(shm_size_)) == 0,
        "ftruncate failed: ",
        std::strerror(errno));
    std::vector<uint8_t> name_bytes(shm_name_.begin(), shm_name_.end());
    store_->set(name_key, name_bytes);
  } else {
    auto name_bytes = store_->get(name_key);
    shm_name_.assign(name_bytes.begin(), name_bytes.end());
    shm_fd_ = ::shm_open(shm_name_.c_str(), O_RDWR, 0600);
    TORCH_CHECK(
        shm_fd_ >= 0,
        "shm_open (attach) failed for ",
        shm_name_,
        ": ",
        std::strerror(errno));
  }

  shm_base_ = ::mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  TORCH_CHECK(shm_base_ != MAP_FAILED, "mmap failed: ", std::strerror(errno));

  // Pin the mapping for CUDA so cudaMemcpyAsync against it is a real,
  // asynchronous DMA transfer rather than an implicitly-synchronous copy
  // through a bounce buffer. Registering a POSIX shared-memory mapping is
  // this layer's multi-process equivalent of direct_collectives'
  // cudaHostAlloc(Mapped|Portable) -- see FSDP-DESIGN.md Sec 3.3 point 1.
  cudaError_t set_dev_err = cudaSetDevice(device_id_);
  TORCH_CHECK(set_dev_err == cudaSuccess, "cudaSetDevice failed: ", cudaGetErrorString(set_dev_err));
  cudaError_t reg_err = cudaHostRegister(shm_base_, shm_size_, cudaHostRegisterPortable);
  TORCH_CHECK(
      reg_err == cudaSuccess || reg_err == cudaErrorHostMemoryAlreadyRegistered,
      "cudaHostRegister failed: ",
      cudaGetErrorString(reg_err));

  if (rank_ == 0) {
    header()->magic.store(kShmMagic, std::memory_order_relaxed);
    header()->abort_flag.store(0, std::memory_order_relaxed);
    header()->ready_world_size.store(static_cast<uint32_t>(world_size_), std::memory_order_relaxed);
    for (int s = 0; s < kNumStagingSlots; ++s) {
      for (int r = 0; r < world_size_; ++r) {
        epoch_ptr(s, r)->store(0, std::memory_order_relaxed);
      }
    }
  }

  // Bounded init barrier: every rank publishes a ready key, every rank
  // waits (bounded by barrier_timeout_, via the Store's own timeout
  // mechanism) for every rank's key to exist. Ensures nobody touches the
  // header/data region before rank 0 has finished initializing it.
  const std::string ready_key_prefix = "lbtorch/collective_transport/host_staged/ready/";
  std::vector<uint8_t> yes{1};
  store_->set(ready_key_prefix + std::to_string(rank_), yes);
  std::vector<std::string> ready_keys;
  ready_keys.reserve(world_size_);
  for (int r = 0; r < world_size_; ++r) {
    ready_keys.push_back(ready_key_prefix + std::to_string(r));
  }
  store_->wait(ready_keys, barrier_timeout_);

  TORCH_CHECK(
      header()->magic.load(std::memory_order_acquire) == kShmMagic,
      "host-staged transport: shared memory header magic mismatch after init barrier");

  cudaError_t stream_err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  TORCH_CHECK(stream_err == cudaSuccess, "cudaStreamCreateWithFlags failed: ", cudaGetErrorString(stream_err));
}

HostStagedTransport::~HostStagedTransport() {
  // Plain, bounded, synchronous teardown calls only -- no persistent
  // "communicator" object with its own teardown protocol to hang in. This
  // is a direct, deliberate adoption of the lifecycle lesson behind
  // direct_collectives' incident #2 (FSDP-DESIGN.md Sec 0 / Sec 3.3 point
  // 3): an explicit ncclCommDestroy() did not fix lbx's orphaned-teardown
  // hang because the defect lived in NCCL's own communicator-teardown
  // machinery, not in a missing destroy call -- so this transport has no
  // analogous persistent communicator object at all.
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
  if (shm_base_ != nullptr && shm_base_ != MAP_FAILED) {
    cudaHostUnregister(shm_base_);
    ::munmap(shm_base_, shm_size_);
  }
  if (shm_fd_ >= 0) {
    ::close(shm_fd_);
  }
  if (rank_ == 0 && !shm_name_.empty()) {
    ::shm_unlink(shm_name_.c_str());
  }
}

void HostStagedTransport::abort() {
  if (shm_base_ != nullptr) {
    header()->abort_flag.store(1, std::memory_order_release);
  }
}

void HostStagedTransport::bounded_event_wait(cudaEvent_t event) {
  const auto deadline = std::chrono::steady_clock::now() + barrier_timeout_;
  while (true) {
    cudaError_t st = cudaEventQuery(event);
    if (st == cudaSuccess) {
      return;
    }
    if (st != cudaErrorNotReady) {
      TORCH_CHECK(false, "cudaEventQuery failed: ", cudaGetErrorString(st));
    }
    if (header()->abort_flag.load(std::memory_order_acquire) != 0) {
      throw TransportAbortedError(
          "host-staged transport aborted while waiting for a device copy to complete");
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw TransportTimeoutError(
          "device copy did not complete within the bounded deadline ("
          + std::to_string(barrier_timeout_.count()) + "ms) -- failing cleanly "
          "instead of hanging indefinitely");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

void HostStagedTransport::publish_arrival(int slot_index, uint64_t epoch) {
  epoch_ptr(slot_index, rank_)->store(epoch, std::memory_order_release);
}

void HostStagedTransport::wait_arrival(int slot_index, uint64_t expected_epoch) {
  const auto deadline = std::chrono::steady_clock::now() + barrier_timeout_;
  for (int r = 0; r < world_size_; ++r) {
    auto* ep = epoch_ptr(slot_index, r);
    while (ep->load(std::memory_order_acquire) < expected_epoch) {
      if (header()->abort_flag.load(std::memory_order_acquire) != 0) {
        throw TransportAbortedError(
            "host-staged transport aborted while waiting for peer rank " +
            std::to_string(r) + " to arrive");
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        // This bounded failure, instead of an unbounded spin, is the exact
        // property that would have turned IP66's original flat_param_to
        // freeze into a clean, diagnosable error -- FSDP-DESIGN.md Sec 0.
        throw TransportTimeoutError(
            "timed out after " + std::to_string(barrier_timeout_.count()) +
            "ms waiting for rank " + std::to_string(r) +
            " to arrive at this collective call (bounded wait, no indefinite spin)");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }
}

void HostStagedTransport::all_gather(const at::Tensor& input_shard, at::Tensor& output) {
  TORCH_CHECK(
      input_shard.is_cuda() && input_shard.get_device() == device_id_,
      "input_shard must live on this transport's CUDA device (", device_id_, ")");
  TORCH_CHECK(
      output.is_cuda() && output.get_device() == device_id_,
      "output must live on this transport's CUDA device (", device_id_, ")");
  TORCH_CHECK(
      input_shard.is_contiguous() && output.is_contiguous(),
      "host-staged transport requires contiguous tensors");
  TORCH_CHECK(
      input_shard.scalar_type() == output.scalar_type(),
      "input_shard and output must share a dtype");
  TORCH_CHECK(
      output.numel() == input_shard.numel() * static_cast<int64_t>(world_size_),
      "output must hold world_size shards concatenated");

  const size_t shard_bytes =
      static_cast<size_t>(input_shard.numel()) * input_shard.element_size();
  TORCH_CHECK(
      shard_bytes <= slot_capacity_bytes_,
      "input shard (",
      shard_bytes,
      " bytes) exceeds this transport's configured slot capacity (",
      slot_capacity_bytes_,
      " bytes)");

  const uint64_t call_index = call_counter_++;
  const int slot = static_cast<int>(call_index % kNumStagingSlots);
  const uint64_t epoch = call_index / kNumStagingSlots + 1;

  // Stage this rank's shard: device -> pinned host shared memory. Never a
  // GPU-to-GPU cudaMemcpyPeerAsync / direct P2P DMA -- FSDP-DESIGN.md
  // Sec 3.3 point 1.
  cudaEvent_t ev;
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  cudaError_t cpy_err = cudaMemcpyAsync(
      slot_ptr(slot, rank_), input_shard.data_ptr(), shard_bytes, cudaMemcpyDeviceToHost, stream_);
  TORCH_CHECK(cpy_err == cudaSuccess, "cudaMemcpyAsync (D2H) failed: ", cudaGetErrorString(cpy_err));
  cudaEventRecord(ev, stream_);
  bounded_event_wait(ev); // bounded, never cudaStreamSynchronize
  cudaEventDestroy(ev);

  publish_arrival(slot, epoch);
  // Bounded, abortable barrier -- the direct fix for the original
  // unbounded-spin hazard class (FSDP-DESIGN.md Sec 0).
  wait_arrival(slot, epoch);

  // Finishing step: copy every rank's slot back out into this rank's
  // output, in rank order. Plain H2D copies, never P2P.
  for (int r = 0; r < world_size_; ++r) {
    void* dst = static_cast<char*>(output.data_ptr()) + static_cast<size_t>(r) * shard_bytes;
    cudaEvent_t ev2;
    cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming);
    cudaError_t cpy2 = cudaMemcpyAsync(
        dst, slot_ptr(slot, r), shard_bytes, cudaMemcpyHostToDevice, stream_);
    TORCH_CHECK(cpy2 == cudaSuccess, "cudaMemcpyAsync (H2D) failed: ", cudaGetErrorString(cpy2));
    cudaEventRecord(ev2, stream_);
    bounded_event_wait(ev2);
    cudaEventDestroy(ev2);
  }
}

void HostStagedTransport::reduce_scatter(const at::Tensor& input, at::Tensor& output_shard) {
  TORCH_CHECK(
      input.is_cuda() && input.get_device() == device_id_,
      "input must live on this transport's CUDA device (", device_id_, ")");
  TORCH_CHECK(
      output_shard.is_cuda() && output_shard.get_device() == device_id_,
      "output_shard must live on this transport's CUDA device (", device_id_, ")");
  TORCH_CHECK(
      input.is_contiguous() && output_shard.is_contiguous(),
      "host-staged transport requires contiguous tensors");
  TORCH_CHECK(
      input.scalar_type() == output_shard.scalar_type(),
      "input and output_shard must share a dtype");
  TORCH_CHECK(
      input.numel() % world_size_ == 0,
      "reduce_scatter requires input evenly divisible by world_size "
      "(uneven-shard FSDP support is a documented later-phase limitation)");

  const int64_t shard_numel = input.numel() / world_size_;
  TORCH_CHECK(output_shard.numel() == shard_numel, "output_shard size mismatch");

  const size_t full_bytes = static_cast<size_t>(input.numel()) * input.element_size();
  TORCH_CHECK(
      full_bytes <= slot_capacity_bytes_,
      "input tensor (",
      full_bytes,
      " bytes) exceeds this transport's configured slot capacity (",
      slot_capacity_bytes_,
      " bytes)");

  const uint64_t call_index = call_counter_++;
  const int slot = static_cast<int>(call_index % kNumStagingSlots);
  const uint64_t epoch = call_index / kNumStagingSlots + 1;

  cudaEvent_t ev;
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  cudaError_t cpy_err = cudaMemcpyAsync(
      slot_ptr(slot, rank_), input.data_ptr(), full_bytes, cudaMemcpyDeviceToHost, stream_);
  TORCH_CHECK(cpy_err == cudaSuccess, "cudaMemcpyAsync (D2H) failed: ", cudaGetErrorString(cpy_err));
  cudaEventRecord(ev, stream_);
  bounded_event_wait(ev);
  cudaEventDestroy(ev);

  publish_arrival(slot, epoch);
  wait_arrival(slot, epoch);

  // Sum this rank's output-shard range across every rank's staged full
  // tensor, entirely in host memory, reusing ATen's own already-correct,
  // already-vectorized elementwise add instead of hand-writing a
  // dtype-specific reduction loop -- PLAN.md Sec 5's "reuse, don't
  // rewrite" philosophy, and the generalization FSDP-DESIGN.md Sec 2 calls
  // for past direct_collectives' bf16-only reduce kernel.
  const auto options = at::TensorOptions().dtype(input.scalar_type()).device(at::kCPU);
  at::Tensor acc = at::zeros({shard_numel}, options);
  const size_t shard_bytes = static_cast<size_t>(shard_numel) * input.element_size();
  const size_t shard_offset_bytes = static_cast<size_t>(rank_) * shard_bytes;
  for (int r = 0; r < world_size_; ++r) {
    void* src = slot_ptr(slot, r) + shard_offset_bytes;
    at::Tensor view = at::from_blob(src, {shard_numel}, options);
    acc.add_(view);
  }

  cudaError_t cpy2 = cudaMemcpyAsync(
      output_shard.data_ptr(), acc.data_ptr(), shard_bytes, cudaMemcpyHostToDevice, stream_);
  TORCH_CHECK(cpy2 == cudaSuccess, "cudaMemcpyAsync (H2D) failed: ", cudaGetErrorString(cpy2));
  cudaEvent_t ev2;
  cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming);
  cudaEventRecord(ev2, stream_);
  bounded_event_wait(ev2);
  cudaEventDestroy(ev2);
}

} // namespace torch::collectives
