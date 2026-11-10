#pragma once

/// Minimal, dependency-light reader for the `safetensors` checkpoint format
/// (https://github.com/huggingface/safetensors#format), plus HuggingFace's
/// sharded-checkpoint convention (`<name>.safetensors.index.json` mapping
/// each tensor name to the shard file that holds it).
///
/// Format recap: each `.safetensors` file is
///   [8 bytes: u64 LE header_len]
///   [header_len bytes: UTF-8 JSON header]
///   [raw tensor bytes, back-to-back]
/// The JSON header maps tensor name -> {"dtype": ..., "shape": [...],
/// "data_offsets": [begin, end]} (byte offsets relative to the start of the
/// data block, i.e. relative to byte `8 + header_len` in the file), plus an
/// optional "__metadata__" key that is not a tensor.
///
/// No third-party safetensors library is used: the header is parsed with
/// nlohmann::json (already vendored in this tree at
/// third_party/nlohmann/include and already linked into libtorch's own
/// build -- see caffe2/CMakeLists.txt), and the raw tensor bytes are mapped
/// directly into torch::Tensor via torch::from_blob.
///
/// This is the C++ equivalent of Python's `safetensors.torch.load_file` /
/// HuggingFace's `from_pretrained` checkpoint-reading step -- part of
/// PLAN.md's Python-elimination effort (loading a real model checkpoint
/// without ever invoking the `transformers` or `safetensors` Python
/// packages). See torch/models/qwen3.h for a concrete consumer.

#include <torch/csrc/Export.h>
#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch::utils::safetensors {

// RAII wrapper around a read-only mmap of one safetensors file. Shared (via
// shared_ptr) by every torch::Tensor view created from this shard, so the
// mapping outlives any tensor that still points into it: SafetensorsCollection::get()
// below creates its (ephemeral) from_blob view with a deleter that captures a
// shared_ptr<MappedShard>, which is the mechanism that keeps the mapping
// alive for exactly as long as something still references it, rather than
// requiring the caller to separately track the mmap's lifetime.
class TORCH_API MappedShard {
 public:
  explicit MappedShard(const std::string& path);
  ~MappedShard();

  MappedShard(const MappedShard&) = delete;
  MappedShard& operator=(const MappedShard&) = delete;

  // Pointer to the start of the tensor *data* block (i.e. just past the
  // 8-byte length prefix and the JSON header).
  const uint8_t* data() const {
    return base_ + data_offset_;
  }

 private:
  uint8_t* base_ = nullptr;  // mmap base (includes header bytes)
  size_t size_ = 0;          // total mapped file size
  size_t data_offset_ = 0;   // 8 + header_len
};

struct TensorMeta {
  torch::ScalarType dtype;
  std::vector<int64_t> shape;
  size_t begin = 0;  // byte offset into the shard's data block
  size_t end = 0;
  size_t shard_index = 0;
};

// Loads one or more safetensors shards (as named by a HuggingFace-style
// `*.safetensors.index.json`, or a single `model.safetensors` file) and
// vends individual tensors by name (e.g.
// "model.layers.3.self_attn.q_proj.weight").
class TORCH_API SafetensorsCollection {
 public:
  // `dir` must contain either:
  //   - `model.safetensors.index.json` + the shard files it references, or
  //   - a single `model.safetensors` file.
  static SafetensorsCollection FromDirectory(const std::string& dir);

  bool has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
  }

  std::vector<std::string> names() const;
  size_t size() const {
    return tensors_.size();
  }

  // Returns an owned (cloned) tensor with the given target dtype -- the
  // from_blob view used to read the raw bytes is temporary; the returned
  // tensor does not depend on the mmap staying alive afterward.
  torch::Tensor get(
      const std::string& name,
      torch::ScalarType target_dtype = torch::kFloat32) const;

 private:
  std::vector<std::shared_ptr<MappedShard>> shards_;
  std::unordered_map<std::string, TensorMeta> tensors_;

  void AddShard(const std::string& path, const std::vector<std::string>* only_names);
};

} // namespace torch::utils::safetensors
