#include <torch/optim/adamw_checkpoint_bridge.h>

#include <c10/util/irange.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace torch::optim::checkpoint_bridge {

namespace {

constexpr char kMagic[4] = {'L', 'B', 'A', '1'};

enum class DtypeCode : int32_t {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kBFloat16 = 3,
};

DtypeCode dtype_to_code(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return DtypeCode::kFloat32;
    case at::kDouble:
      return DtypeCode::kFloat64;
    case at::kHalf:
      return DtypeCode::kFloat16;
    case at::kBFloat16:
      return DtypeCode::kBFloat16;
    default:
      TORCH_CHECK(
          false,
          "adamw_checkpoint_bridge: unsupported param dtype ",
          dtype,
          " (supported: float32, float64, float16, bfloat16)");
  }
}

at::ScalarType code_to_dtype(int32_t code) {
  switch (static_cast<DtypeCode>(code)) {
    case DtypeCode::kFloat32:
      return at::kFloat;
    case DtypeCode::kFloat64:
      return at::kDouble;
    case DtypeCode::kFloat16:
      return at::kHalf;
    case DtypeCode::kBFloat16:
      return at::kBFloat16;
    default:
      TORCH_CHECK(false, "adamw_checkpoint_bridge: unknown dtype code ", code);
  }
}

template <typename T>
void write_pod(std::ofstream& f, const T& value) {
  f.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void read_pod(std::ifstream& f, T& value) {
  f.read(reinterpret_cast<char*>(&value), sizeof(T));
  TORCH_CHECK(f.good(), "adamw_checkpoint_bridge: unexpected EOF while reading checkpoint");
}

void write_tensor_bytes(std::ofstream& f, const Tensor& t) {
  // Always written contiguous, on CPU, in the tensor's own dtype -- the
  // reader casts back to whatever dtype the destination tensor expects.
  auto cpu_t = t.detach().to(at::kCPU).contiguous();
  f.write(
      reinterpret_cast<const char*>(cpu_t.const_data_ptr()),
      static_cast<std::streamsize>(cpu_t.numel() * cpu_t.element_size()));
}

Tensor read_tensor_bytes(
    std::ifstream& f,
    const std::vector<int64_t>& shape,
    at::ScalarType dtype) {
  auto t = torch::empty(shape, torch::TensorOptions().dtype(dtype));
  f.read(
      reinterpret_cast<char*>(t.mutable_data_ptr()),
      static_cast<std::streamsize>(t.numel() * t.element_size()));
  TORCH_CHECK(f.good(), "adamw_checkpoint_bridge: unexpected EOF while reading tensor payload");
  return t;
}

std::vector<Tensor> flattened_params(const AdamW& optimizer) {
  std::vector<Tensor> params;
  for (auto& group : optimizer.param_groups()) {
    for (auto& p : group.params()) {
      params.push_back(p);
    }
  }
  return params;
}

} // namespace

void export_checkpoint(const AdamW& optimizer, const std::string& path) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  TORCH_CHECK(f.is_open(), "adamw_checkpoint_bridge: could not open '", path, "' for writing");

  f.write(kMagic, sizeof(kMagic));
  write_pod<int32_t>(f, 1); // format version

  auto params = flattened_params(optimizer);
  write_pod<int64_t>(f, static_cast<int64_t>(params.size()));

  // `optimizer.state()` is keyed by TensorImpl*; the only external-facing
  // thing AdamW exposes is `const ska::flat_hash_map<...>& state() const`,
  // so we look each param up by its impl pointer, same as AdamW::step()
  // does internally.
  for (auto& p : params) {
    auto it = optimizer.state().find(p.unsafeGetTensorImpl());
    TORCH_CHECK(
        it != optimizer.state().end(),
        "adamw_checkpoint_bridge: parameter has no AdamW state yet -- "
        "run at least one optimizer.step() (or otherwise lazily-initialize "
        "state) before exporting a checkpoint");
    const auto& state = static_cast<const AdamWParamState&>(*it->second);

    auto shape = p.sizes().vec();
    write_pod<int32_t>(f, static_cast<int32_t>(shape.size()));
    for (auto s : shape) {
      write_pod<int64_t>(f, s);
    }
    write_pod<int32_t>(f, static_cast<int32_t>(dtype_to_code(p.scalar_type())));

    bool has_max = state.max_exp_avg_sq().defined();
    write_pod<uint8_t>(f, has_max ? 1 : 0);

    // Canonical step value: prefer the device-resident tensor (authoritative
    // under capturable/fused) if present, else the host int64_t.
    double step_value = state.step_tensor().defined()
        ? state.step_tensor().item<double>()
        : static_cast<double>(state.step());
    write_pod<double>(f, step_value);

    write_tensor_bytes(f, p);
    write_tensor_bytes(f, state.exp_avg());
    write_tensor_bytes(f, state.exp_avg_sq());
    if (has_max) {
      write_tensor_bytes(f, state.max_exp_avg_sq());
    }
  }
  TORCH_CHECK(f.good(), "adamw_checkpoint_bridge: write failed for '", path, "'");
}

void import_checkpoint(AdamW& optimizer, const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  TORCH_CHECK(f.is_open(), "adamw_checkpoint_bridge: could not open '", path, "' for reading");

  char magic[4];
  f.read(magic, sizeof(magic));
  TORCH_CHECK(
      f.good() && std::memcmp(magic, kMagic, sizeof(kMagic)) == 0,
      "adamw_checkpoint_bridge: '",
      path,
      "' is not an lbtorch AdamW checkpoint (bad magic)");

  int32_t version;
  read_pod(f, version);
  TORCH_CHECK(version == 1, "adamw_checkpoint_bridge: unsupported format version ", version);

  int64_t num_params;
  read_pod(f, num_params);

  auto params = flattened_params(optimizer);
  TORCH_CHECK(
      static_cast<int64_t>(params.size()) == num_params,
      "adamw_checkpoint_bridge: checkpoint has ",
      num_params,
      " params but optimizer has ",
      params.size(),
      " -- construct the AdamW optimizer over the matching model params "
      "before importing");

  for (const auto idx : c10::irange(num_params)) {
    int32_t ndim;
    read_pod(f, ndim);
    std::vector<int64_t> shape(ndim);
    for (auto& s : shape) {
      read_pod(f, s);
    }
    int32_t dtype_code;
    read_pod(f, dtype_code);
    at::ScalarType dtype = code_to_dtype(dtype_code);

    uint8_t has_max_u8;
    read_pod(f, has_max_u8);
    bool has_max = has_max_u8 != 0;

    double step_value;
    read_pod(f, step_value);

    auto& p = params[idx];
    TORCH_CHECK(
        p.sizes().vec() == shape,
        "adamw_checkpoint_bridge: param ",
        idx,
        " shape mismatch: checkpoint has ",
        shape,
        ", optimizer param has ",
        p.sizes());

    Tensor param_data = read_tensor_bytes(f, shape, dtype);
    Tensor exp_avg = read_tensor_bytes(f, shape, dtype);
    Tensor exp_avg_sq = read_tensor_bytes(f, shape, dtype);
    Tensor max_exp_avg_sq;
    if (has_max) {
      max_exp_avg_sq = read_tensor_bytes(f, shape, dtype);
    }

    {
      torch::NoGradGuard no_grad;
      p.copy_(param_data.to(p.scalar_type()).to(p.device()));
    }

    auto state = std::make_unique<AdamWParamState>();
    state->step(static_cast<int64_t>(step_value));
    state->exp_avg(exp_avg.to(p.scalar_type()).to(p.device()));
    state->exp_avg_sq(exp_avg_sq.to(p.scalar_type()).to(p.device()));
    if (has_max) {
      state->max_exp_avg_sq(max_exp_avg_sq.to(p.scalar_type()).to(p.device()));
    }
    // Note: we intentionally do NOT populate `step_tensor` here. If the
    // destination optimizer's options have capturable/fused enabled,
    // AdamW::step_single_tensor/step_foreach/step_fused all lazily
    // materialize `step_tensor` from `state.step()` on first use (see
    // adamw.cpp), so the imported step value is still honored correctly
    // either way -- this keeps the import path agnostic to which mode the
    // checkpoint was produced under vs. which mode it's resumed under.
    optimizer.state()[p.unsafeGetTensorImpl()] = std::move(state);
  }
  TORCH_CHECK(f.good() || f.eof(), "adamw_checkpoint_bridge: read failed for '", path, "'");
}

} // namespace torch::optim::checkpoint_bridge
