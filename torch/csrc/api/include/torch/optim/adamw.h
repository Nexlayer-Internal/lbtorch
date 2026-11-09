#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

struct TORCH_API AdamWOptions : public OptimizerCloneableOptions<AdamWOptions> {
  AdamWOptions(double lr = 1e-3);
  TORCH_ARG(double, lr) = 1e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 1e-2;
  TORCH_ARG(bool, amsgrad) = false;
  // --- lbtorch extension (see PLAN.md row 9 / adamw-foreach-fused-capturable) ---
  // Mirrors `torch/optim/adamw.py`'s `foreach`/`fused`/`capturable` kwargs.
  // Unlike Python, these are plain bools here (no `None` "auto-select"
  // tri-state) -- the C++ frontend favors explicit control over device-
  // dependent heuristics. `foreach` and `fused` are mutually exclusive.
  TORCH_ARG(bool, foreach) = false;
  TORCH_ARG(bool, fused) = false;
  TORCH_ARG(bool, capturable) = false;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdamWOptions& lhs,
      const AdamWOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API AdamWParamState
    : public OptimizerCloneableParamState<AdamWParamState> {
  // Host-resident step counter. This remains the source of truth for the
  // original (non-foreach, non-capturable, non-fused) code path and is kept
  // backwards compatible with every existing serialized archive / call site.
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);
  TORCH_ARG(torch::Tensor, max_exp_avg_sq);
  // --- lbtorch extension ---
  // Device-resident scalar step counter, used only when `capturable` and/or
  // `fused` is enabled (mirrors Python's `state["step"]` always being a
  // tensor; see `torch/optim/adam.py::_init_group`, "special device hosting
  // for step"). Keeping it device-resident lets the increment and the bias-
  // correction powers be expressed as ordinary tensor ops that can be
  // captured into a CUDA graph -- no `.item()` host sync is on the hot path.
  // Left undefined for the default (non-capturable, non-fused) path.
  TORCH_ARG(torch::Tensor, step_tensor);

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdamWParamState& lhs,
      const AdamWParamState& rhs);
};

class TORCH_API AdamW : public Optimizer {
 public:
  explicit AdamW(
      const std::vector<OptimizerParamGroup>& param_groups,
      AdamWOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<AdamWOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        !(defaults.foreach() && defaults.fused()),
        "`fused` and `foreach` cannot both be true");
  }
  explicit AdamW(std::vector<Tensor> params, AdamWOptions defaults = {})
      : AdamW({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(AdamW);
  }

  // One AdamW::step() call dispatches a param group to exactly one of these,
  // based on `options.foreach()` / `options.fused()`. Each mirrors the
  // correspondingly named function in `torch/optim/adam.py` as closely as
  // possible (bias-correction terms, epsilon placement, and weight-decay
  // order are copied verbatim from the Python reference). `capturable` is
  // NOT a fourth, separate path -- exactly as in Python, it's an orthogonal
  // flag that `step_single_tensor` and `step_foreach` both branch on
  // internally, trading a host-scalar bias-correction computation for an
  // all-device-tensor one that never needs to read `step` back to the host.
  void step_single_tensor(OptimizerParamGroup& group, AdamWOptions& options);
  void step_foreach(OptimizerParamGroup& group, AdamWOptions& options);
  void step_fused(OptimizerParamGroup& group, AdamWOptions& options);
};
} // namespace torch::optim
