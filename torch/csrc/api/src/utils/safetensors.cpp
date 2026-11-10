#include <torch/utils/safetensors.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace torch::utils::safetensors {

namespace {

torch::ScalarType DtypeFromSafetensors(const std::string& s) {
  // https://github.com/huggingface/safetensors#format
  if (s == "F32")
    return torch::kFloat32;
  if (s == "F16")
    return torch::kFloat16;
  if (s == "BF16")
    return torch::kBFloat16;
  if (s == "F64")
    return torch::kFloat64;
  if (s == "I64")
    return torch::kInt64;
  if (s == "I32")
    return torch::kInt32;
  if (s == "I16")
    return torch::kInt16;
  if (s == "I8")
    return torch::kInt8;
  if (s == "U8")
    return torch::kUInt8;
  if (s == "BOOL")
    return torch::kBool;
  TORCH_CHECK(false, "safetensors: unsupported dtype string '", s, "'");
}

} // namespace

MappedShard::MappedShard(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  TORCH_CHECK(fd >= 0, "safetensors: failed to open '", path, "': ", std::strerror(errno));
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    TORCH_CHECK(false, "safetensors: fstat failed for '", path, "'");
  }
  size_ = static_cast<size_t>(st.st_size);
  if (size_ < 8) {
    ::close(fd);
    TORCH_CHECK(false, "safetensors: file too small to contain a header: ", path);
  }
  void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd); // fd not needed once mapped
  if (mapped == MAP_FAILED) {
    TORCH_CHECK(false, "safetensors: mmap failed for '", path, "': ", std::strerror(errno));
  }
  base_ = static_cast<uint8_t*>(mapped);

  uint64_t header_len = 0;
  // File is little-endian per the safetensors spec; host is assumed
  // little-endian (x86_64/aarch64), matching every other binary-format
  // reader elsewhere in this codebase.
  std::memcpy(&header_len, base_, sizeof(header_len));
  data_offset_ = 8 + static_cast<size_t>(header_len);
  if (data_offset_ > size_) {
    ::munmap(base_, size_);
    base_ = nullptr;
    TORCH_CHECK(false, "safetensors: header_len overruns file size in '", path, "'");
  }
}

MappedShard::~MappedShard() {
  if (base_ != nullptr) {
    ::munmap(base_, size_);
  }
}

void SafetensorsCollection::AddShard(
    const std::string& path,
    const std::vector<std::string>* only_names) {
  auto shard = std::make_shared<MappedShard>(path);
  size_t shard_index = shards_.size();

  // MappedShard only exposes the tensor *data* block (post-header); re-read
  // the (small) header JSON directly from the file with a couple of
  // pread()s rather than having MappedShard expose its raw header span too.
  int fd = ::open(path.c_str(), O_RDONLY);
  TORCH_CHECK(fd >= 0, "safetensors: failed to reopen '", path, "' for header parse");
  uint64_t raw_len = 0;
  if (::pread(fd, &raw_len, sizeof(raw_len), 0) != static_cast<ssize_t>(sizeof(raw_len))) {
    ::close(fd);
    TORCH_CHECK(false, "safetensors: failed to read header length of '", path, "'");
  }
  std::string header_json(raw_len, '\0');
  if (::pread(fd, header_json.data(), raw_len, 8) != static_cast<ssize_t>(raw_len)) {
    ::close(fd);
    TORCH_CHECK(false, "safetensors: failed to read header JSON of '", path, "'");
  }
  ::close(fd);

  nlohmann::json header = nlohmann::json::parse(header_json);

  for (auto it = header.begin(); it != header.end(); ++it) {
    const std::string& name = it.key();
    if (name == "__metadata__")
      continue;
    if (only_names != nullptr &&
        std::find(only_names->begin(), only_names->end(), name) == only_names->end()) {
      continue;
    }
    const nlohmann::json& meta = it.value();
    TensorMeta tm;
    tm.dtype = DtypeFromSafetensors(meta.at("dtype").get<std::string>());
    tm.shape = meta.at("shape").get<std::vector<int64_t>>();
    const auto offsets = meta.at("data_offsets").get<std::vector<size_t>>();
    TORCH_CHECK(offsets.size() == 2, "safetensors: malformed data_offsets for '", name, "'");
    tm.begin = offsets[0];
    tm.end = offsets[1];
    tm.shard_index = shard_index;
    tensors_[name] = tm;
  }

  shards_.push_back(std::move(shard));
}

SafetensorsCollection SafetensorsCollection::FromDirectory(const std::string& dir) {
  SafetensorsCollection coll;

  const std::string index_path = dir + "/model.safetensors.index.json";
  struct stat st {};
  if (::stat(index_path.c_str(), &st) == 0) {
    std::ifstream f(index_path);
    TORCH_CHECK(f, "safetensors: could not open index '", index_path, "'");
    std::stringstream ss;
    ss << f.rdbuf();
    nlohmann::json index = nlohmann::json::parse(ss.str());
    const auto& weight_map = index.at("weight_map");

    // Group tensor names by which shard file holds them, then load each
    // shard file exactly once.
    std::unordered_map<std::string, std::vector<std::string>> by_shard;
    for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
      by_shard[it.value().get<std::string>()].push_back(it.key());
    }
    for (auto& [shard_file, names] : by_shard) {
      coll.AddShard(dir + "/" + shard_file, &names);
    }
  } else {
    // Single-file checkpoint.
    coll.AddShard(dir + "/model.safetensors", nullptr);
  }

  return coll;
}

std::vector<std::string> SafetensorsCollection::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (auto& [name, meta] : tensors_)
    out.push_back(name);
  return out;
}

torch::Tensor SafetensorsCollection::get(const std::string& name, torch::ScalarType target_dtype)
    const {
  auto it = tensors_.find(name);
  TORCH_CHECK(it != tensors_.end(), "safetensors: no such tensor '", name, "'");
  const TensorMeta& tm = it->second;
  std::shared_ptr<MappedShard> shard = shards_.at(tm.shard_index);

  const uint8_t* ptr = shard->data() + tm.begin;

  auto options = torch::TensorOptions().dtype(tm.dtype);
  // from_blob's deleter captures `shard` (a shared_ptr) purely to keep the
  // mmap alive for as long as this *view* tensor exists; `view` itself is
  // ephemeral here (immediately cloned+cast into an owned tensor below),
  // but this is the general technique for any caller that wants a
  // zero-copy view directly instead.
  torch::Tensor view = torch::from_blob(
      const_cast<uint8_t*>(ptr),
      tm.shape,
      [shard](void*) mutable { shard.reset(); },
      options);

  return view.to(target_dtype).clone();
}

} // namespace torch::utils::safetensors
