#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/optim/adamw_checkpoint_bridge.h>
#include <torch/torch.h>

#include <test/cpp/api/optim_baseline.h>
#include <test/cpp/api/support.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace torch::nn;
using namespace torch::optim;

template <typename OptimizerClass, typename Options>
static bool test_optimizer_xor(Options options) {
  torch::manual_seed(0);

  Sequential model(
      Linear(2, 8),
      Functional(torch::sigmoid),
      Linear(8, 1),
      Functional(torch::sigmoid));

  const int64_t kBatchSize = 200;
  const int64_t kMaximumNumberOfEpochs = 3000;

  OptimizerClass optimizer(model->parameters(), std::move(options));

  double running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    auto inputs = torch::empty({kBatchSize, 2});
    auto labels = torch::empty({kBatchSize});
    for (const auto i : c10::irange(kBatchSize)) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
    }

    inputs.set_requires_grad(true);

    auto step = [&](OptimizerClass& optimizer,
                    Sequential model,
                    const torch::Tensor& inputs,
                    const torch::Tensor& labels) {
      auto closure = [&]() {
        optimizer.zero_grad();
        auto x = model->forward(inputs);
        auto loss = torch::binary_cross_entropy(x, labels);
        loss.backward();
        return loss;
      };
      return optimizer.step(closure);
    };

    torch::Tensor loss = step(optimizer, model, inputs, labels);

    running_loss = running_loss * 0.99 + loss.item<double>() * 0.01;
    if (epoch > kMaximumNumberOfEpochs) {
      std::cout << "Loss is too high after epoch " << epoch << ": "
                << running_loss << '\n';
      return false;
    }
    epoch++;
  }
  return true;
}

template <typename Parameters>
static void assign_parameter(
    const Parameters& parameters,
    const char* name,
    const torch::Tensor& new_tensor) {
  auto parameter = parameters[name];
  parameter.set_requires_grad(false);
  parameter.flatten().copy_(new_tensor);
  parameter.set_requires_grad(true);
}

template <typename OptimizerClass, typename Options>
static void check_exact_values(
    Options options,
    std::vector<std::vector<torch::Tensor>> expected_parameters) {
  const size_t kIterations = 1001;
  const size_t kSampleEvery = 100;

  torch::manual_seed(0);

  Sequential model(
      Linear(2, 3),
      Functional(torch::sigmoid),
      Linear(3, 1),
      Functional(torch::sigmoid));

  model->to(torch::kFloat64);

  // Use exact input values because matching random values is hard.
  auto parameters = model->named_parameters();
  assign_parameter(
      parameters,
      "0.weight",
      torch::tensor(
          {-0.2109, -0.4976, -0.1413, -0.3420, -0.2524, 0.6976},
          torch::kFloat64));
  assign_parameter(
      parameters,
      "0.bias",
      torch::tensor({-0.1085, -0.2979, 0.6892}, torch::kFloat64));
  assign_parameter(
      parameters,
      "2.weight",
      torch::tensor({-0.0508, -0.3941, -0.2843}, torch::kFloat64));
  assign_parameter(
      parameters, "2.bias", torch::tensor({-0.0711}, torch::kFloat64));

  auto optimizer = OptimizerClass(parameters.values(), std::move(options));
  torch::Tensor input =
      torch::tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, torch::kFloat64)
          .reshape({3, 2});

  for (const auto i : c10::irange(kIterations)) {
    optimizer.zero_grad();
    auto output = model->forward(input);
    auto loss = output.sum();
    loss.backward();

    auto closure = []() { return torch::tensor({10}); };
    optimizer.step(closure);

    if (i % kSampleEvery == 0) {
      ASSERT_TRUE(
          expected_parameters.at(i / kSampleEvery).size() == parameters.size());
      for (const auto p : c10::irange(parameters.size())) {
        ASSERT_TRUE(parameters[p]->defined());
        // Always compare using double dtype, regardless of the original dtype
        // of the tensors
        auto computed = parameters[p]->flatten().to(torch::kFloat64);
        auto expected =
            expected_parameters.at(i / kSampleEvery).at(p).to(torch::kFloat64);
        if (!computed.allclose(expected, /*rtol=*/1e-3, /*atol=*/5e-4)) {
          std::cout << "Iteration " << i << ": " << computed
                    << " != " << expected << " (parameter " << p << ')' << '\n';
          ASSERT_TRUE(false);
        }
      }
    }
  }
}

TEST(OptimTest, OptimizerAccessors) {
  auto options = AdagradOptions(1.0);
  std::vector<torch::Tensor> params;
  for ([[maybe_unused]] const auto i : c10::irange(3)) {
    params.push_back(torch::randn(10));
  }
  auto optimizer = Adagrad(params, options);
  // test for defaults() method with non-const reference
  auto& options_ = static_cast<AdagradOptions&>(optimizer.defaults());
  ASSERT_TRUE(options == options_);
  // test for param_groups() with non-const reference return
  auto& params_groups = optimizer.param_groups();
  params_groups.emplace_back(params);
  auto& params_1 = params_groups[1].params();
  for (const auto i : c10::irange(params_1.size())) {
    torch::equal(params[i], params_1[i]);
  }

  // test for add_param_group() when one or more params existing in another
  // param_group are passed in the new param group to be added
  ASSERT_THROWS_WITH(
      optimizer.add_param_group(OptimizerParamGroup(params)),
      "some parameters appear in more than one parameter group");

  // test for state() with non-const reference return
  auto& state_ = static_cast<AdagradParamState&>(
      *(optimizer.state()[params_1[0].unsafeGetTensorImpl()]));
  state_.step(state_.step() + 1);

  const auto& optimizer_ = Adagrad(params, options);
  optimizer_.defaults();
  // test for param_groups() with const reference return
  (void)optimizer_.param_groups();
  // test for state() with const reference return
  optimizer_.state();
}

#define OLD_INTERFACE_WARNING_CHECK(func)       \
  {                                             \
    torch::test::WarningCapture warnings;       \
    func;                                       \
    ASSERT_EQ(                                  \
        torch::test::count_substr_occurrences(  \
            warnings.str(), "will be removed"), \
        1);                                     \
  }

struct MyOptimizerOptions
    : public OptimizerCloneableOptions<MyOptimizerOptions> {
  MyOptimizerOptions(double lr = 1.0) : lr_(lr) {}
  TORCH_ARG(double, lr) = 1.0;
};

TEST(OptimTest, OldInterface) {
  struct MyOptimizer : Optimizer {
    using Optimizer::Optimizer;
    torch::Tensor step(LossClosure closure = nullptr) override {
      return {};
    }
    explicit MyOptimizer(
        std::vector<at::Tensor> params,
        const MyOptimizerOptions& defaults = {})
        : Optimizer(
              std::move(params),
              std::make_unique<MyOptimizerOptions>(defaults)) {}
  };
  std::vector<torch::Tensor> parameters = {
      torch::ones({2, 3}), torch::zeros({2, 3}), torch::rand({2, 3})};
  {
    MyOptimizer optimizer(parameters);
    size_t size = 0;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, parameters.size());
  }
  {
    std::vector<at::Tensor> params;
    MyOptimizer optimizer(params);

    size_t size = 0;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, 0);

    OLD_INTERFACE_WARNING_CHECK(optimizer.add_parameters(parameters));

    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, parameters.size());

    std::vector<torch::Tensor> params_;
    OLD_INTERFACE_WARNING_CHECK(params_ = optimizer.parameters());
    for (const auto p : c10::irange(size)) {
      ASSERT_TRUE(params_[p].allclose(parameters[p]));
    }
  }
  {
    Linear linear(3, 4);
    MyOptimizer optimizer(linear->parameters());

    size_t size = 0;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, linear->parameters().size());
  }
}

TEST(OptimTest, XORConvergence_SGD) {
  ASSERT_TRUE(test_optimizer_xor<SGD>(
      SGDOptions(0.1).momentum(0.9).nesterov(true).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_LBFGS) {
  ASSERT_TRUE(test_optimizer_xor<LBFGS>(LBFGSOptions(1.0)));
  ASSERT_TRUE(test_optimizer_xor<LBFGS>(
      LBFGSOptions(1.0).line_search_fn("strong_wolfe")));
}

TEST(OptimTest, XORConvergence_Adagrad) {
  ASSERT_TRUE(test_optimizer_xor<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3)));
}

TEST(OptimTest, XORConvergence_RMSprop) {
  ASSERT_TRUE(test_optimizer_xor<RMSprop>(RMSpropOptions(0.1).centered(true)));
}

TEST(OptimTest, XORConvergence_RMSpropWithMomentum) {
  ASSERT_TRUE(test_optimizer_xor<RMSprop>(
      RMSpropOptions(0.1).momentum(0.9).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_Adam) {
  ASSERT_TRUE(test_optimizer_xor<Adam>(AdamOptions(0.1).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_AdamWithAmsgrad) {
  ASSERT_TRUE(test_optimizer_xor<Adam>(
      AdamOptions(0.1).weight_decay(1e-6).amsgrad(true)));
}

TEST(OptimTest, ProducesPyTorchValues_Adam) {
  check_exact_values<Adam>(AdamOptions(1.0), expected_parameters::Adam());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWithWeightDecay) {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adam_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWithWeightDecayAndAMSGrad) {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-6).amsgrad(true),
      expected_parameters::Adam_with_weight_decay_and_amsgrad());
}

TEST(OptimTest, XORConvergence_AdamW) {
  ASSERT_TRUE(test_optimizer_xor<AdamW>(AdamWOptions(0.1)));
}

TEST(OptimTest, XORConvergence_AdamWWithAmsgrad) {
  ASSERT_TRUE(test_optimizer_xor<AdamW>(AdamWOptions(0.1).amsgrad(true)));
}

TEST(OptimTest, ProducesPyTorchValues_AdamW) {
  check_exact_values<AdamW>(AdamWOptions(1.0), expected_parameters::AdamW());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWWithoutWeightDecay) {
  check_exact_values<AdamW>(
      AdamWOptions(1.0).weight_decay(0),
      expected_parameters::AdamW_without_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWWithAMSGrad) {
  check_exact_values<AdamW>(
      AdamWOptions(1.0).amsgrad(true),
      expected_parameters::AdamW_with_amsgrad());
}

// --- lbtorch extension: foreach / fused / capturable --------------------
// These mirror the standalone-harness comparisons documented in
// docs/adamw_checkpoint_interop.md (run there against Python's
// torch.optim.AdamW on a pip wheel, since a full lbtorch source build
// wasn't feasible in that validation environment). They're added here so
// they run for real under lbtorch's own build/CI, against lbtorch's own
// compiled AdamW -- that run is NOT something this change's author was able
// to execute (no cmake/ninja + no time budget for an hours-long from-source
// build in this environment); said explicitly rather than claimed as
// verified.
namespace {

// Builds `n` fixed-shape float32 tensors with deterministic (not
// RNG-sourced) values so this test's expectations don't depend on an RNG
// engine matching anything -- same scheme as the standalone harness for
// exactly the same reason, just inlined here instead of shared, to keep
// this test self-contained.
std::vector<torch::Tensor> adamw_ext_test_tensors(int64_t seed_offset) {
  std::vector<std::vector<int64_t>> shapes = {{4, 4}, {8}, {16, 4}, {3, 5, 2}};
  std::vector<torch::Tensor> out;
  int64_t off = seed_offset;
  for (auto& shape : shapes) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    std::vector<float> data(numel);
    for (int64_t i = 0; i < numel; ++i) {
      double x = 0.1 * static_cast<double>(i) + static_cast<double>(off);
      data[i] = static_cast<float>(0.05 * std::sin(x) + 0.001 * std::cos(3.0 * x));
    }
    out.push_back(torch::from_blob(data.data(), shape, torch::kFloat32).clone());
    off += 1000;
  }
  return out;
}

// Runs `num_steps` AdamW steps under `options` starting from the same
// deterministic params/grads every time, returning the final flattened
// parameter values concatenated into one tensor for easy allclose().
torch::Tensor run_adamw_ext(AdamWOptions options, int64_t num_steps) {
  auto params = adamw_ext_test_tensors(/*seed_offset=*/0);
  for (auto& p : params) p.set_requires_grad(true);
  AdamW optimizer(params, options);
  for (int64_t step = 0; step < num_steps; ++step) {
    auto grads = adamw_ext_test_tensors(/*seed_offset=*/500 + step * 37);
    optimizer.zero_grad();
    for (const auto i : c10::irange(params.size())) {
      params[i].mutable_grad() = grads[i];
    }
    optimizer.step();
  }
  std::vector<torch::Tensor> flat;
  flat.reserve(params.size());
  for (auto& p : params) {
    flat.push_back(p.detach().reshape({-1}).to(torch::kFloat64));
  }
  return torch::cat(flat);
}

AdamWOptions adamw_ext_test_options() {
  return AdamWOptions(0.05)
      .betas(std::make_tuple(0.9, 0.999))
      .eps(1e-8)
      .weight_decay(0.01);
}

} // namespace

TEST(OptimTest, AdamWForeachMatchesSingleTensor) {
  auto single = run_adamw_ext(adamw_ext_test_options(), 20);
  auto foreach = run_adamw_ext(adamw_ext_test_options().foreach(true), 20);
  // Same registered _foreach_* ops Python's foreach=True path uses against
  // the same inputs; expect this tight (see docs/adamw_checkpoint_interop.md,
  // observed 0.0 max abs diff against Python in the standalone harness).
  ASSERT_TRUE(torch::allclose(single, foreach, /*rtol=*/1e-6, /*atol=*/1e-6));
}

TEST(OptimTest, AdamWFusedMatchesSingleTensor) {
  auto single = run_adamw_ext(adamw_ext_test_options(), 20);
  auto fused = run_adamw_ext(adamw_ext_test_options().fused(true), 20);
  // Calls at::_fused_adamw_ directly (the real registered fused op, same one
  // Python's fused=True dispatches to) -- not a reimplementation.
  ASSERT_TRUE(torch::allclose(single, fused, /*rtol=*/1e-6, /*atol=*/1e-6));
}

TEST(OptimTest, AdamWCapturableMatchesNonCapturable) {
  // torch.optim.AdamW(capturable=True) raises on CPU tensors upstream, so
  // there is no Python CPU ground truth to compare against here (or in the
  // standalone harness -- see docs/adamw_checkpoint_interop.md). capturable
  // is cross-checked against this same, already Python-validated,
  // non-capturable path instead: it's the same math, computed via device
  // tensor ops (float32) instead of host scalars (double), so a small
  // float32-precision-level gap vs the host path is expected, not a bug.
  auto single = run_adamw_ext(adamw_ext_test_options(), 20);
  auto capturable = run_adamw_ext(adamw_ext_test_options().capturable(true), 20);
  ASSERT_TRUE(torch::allclose(single, capturable, /*rtol=*/1e-5, /*atol=*/1e-5));

  auto foreach = run_adamw_ext(adamw_ext_test_options().foreach(true), 20);
  auto foreach_capturable = run_adamw_ext(
      adamw_ext_test_options().foreach(true).capturable(true), 20);
  ASSERT_TRUE(
      torch::allclose(foreach, foreach_capturable, /*rtol=*/1e-5, /*atol=*/1e-5));
}

TEST(OptimTest, AdamWForeachFusedAreMutuallyExclusive) {
  ASSERT_THROW(
      AdamW(
          {torch::zeros({2, 2}, torch::requires_grad())},
          adamw_ext_test_options().foreach(true).fused(true)),
      c10::Error);
}

TEST(OptimTest, AdamWCheckpointBridgeRoundTrip) {
  // Export after K steps, import into a second optimizer over fresh
  // same-shape params, run one more step on each, and confirm they match --
  // i.e. "save, load, continue" produces the same result as "just keep
  // training," which is the whole point of the checkpoint bridge (see
  // docs/adamw_checkpoint_interop.md). This exercises the C++ <-> C++ round
  // trip in-process; the cross-language (Python <-> C++) directions were
  // validated via the standalone harness documented there, not executed as
  // part of this GTest binary.
  auto options = adamw_ext_test_options();
  const int64_t kStepsBeforeCheckpoint = 10;

  auto params_a = adamw_ext_test_tensors(/*seed_offset=*/0);
  for (auto& p : params_a) p.set_requires_grad(true);
  AdamW optimizer_a(params_a, options);
  for (int64_t step = 0; step < kStepsBeforeCheckpoint; ++step) {
    auto grads = adamw_ext_test_tensors(/*seed_offset=*/500 + step * 37);
    optimizer_a.zero_grad();
    for (const auto i : c10::irange(params_a.size())) {
      params_a[i].mutable_grad() = grads[i];
    }
    optimizer_a.step();
  }

  auto tmpfile = c10::make_tempfile();
  torch::optim::checkpoint_bridge::export_checkpoint(optimizer_a, tmpfile.name);

  auto params_b = adamw_ext_test_tensors(/*seed_offset=*/0);
  for (auto& p : params_b) p.set_requires_grad(true);
  AdamW optimizer_b(params_b, options);
  torch::optim::checkpoint_bridge::import_checkpoint(optimizer_b, tmpfile.name);

  auto next_grads = adamw_ext_test_tensors(
      /*seed_offset=*/500 + kStepsBeforeCheckpoint * 37);

  optimizer_a.zero_grad();
  for (const auto i : c10::irange(params_a.size())) {
    params_a[i].mutable_grad() = next_grads[i];
  }
  optimizer_a.step();

  optimizer_b.zero_grad();
  for (const auto i : c10::irange(params_b.size())) {
    params_b[i].mutable_grad() = next_grads[i];
  }
  optimizer_b.step();

  for (const auto i : c10::irange(params_a.size())) {
    ASSERT_TRUE(torch::allclose(params_a[i], params_b[i], 1e-6, 1e-6));
  }
}

TEST(OptimTest, ProducesPyTorchValues_Adagrad) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0), expected_parameters::Adagrad());
}

TEST(OptimTest, ProducesPyTorchValues_AdagradWithWeightDecay) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adagrad_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdagradWithWeightDecayAndLRDecay) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3),
      expected_parameters::Adagrad_with_weight_decay_and_lr_decay());
}

TEST(OptimTest, ProducesPyTorchValues_RMSprop) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1), expected_parameters::RMSprop());
}

TEST(OptimTest, ProducesPyTorchValues_RMSpropWithWeightDecay) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-2),
      expected_parameters::RMSprop_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_RMSpropWithWeightDecayAndCentered) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true),
      expected_parameters::RMSprop_with_weight_decay_and_centered());
}

TEST(
    OptimTest,
    ProducesPyTorchValues_RMSpropWithWeightDecayAndCenteredAndMomentum) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true).momentum(0.9),
      expected_parameters::
          RMSprop_with_weight_decay_and_centered_and_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_SGD) {
  check_exact_values<SGD>(SGDOptions(0.1), expected_parameters::SGD());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecay) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2),
      expected_parameters::SGD_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecayAndMomentum) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2).momentum(0.9),
      expected_parameters::SGD_with_weight_decay_and_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecayAndNesterovMomentum) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-6).momentum(0.9).nesterov(true),
      expected_parameters::SGD_with_weight_decay_and_nesterov_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_LBFGS) {
  check_exact_values<LBFGS>(LBFGSOptions(1.0), expected_parameters::LBFGS());
}

TEST(OptimTest, ProducesPyTorchValues_LBFGS_with_line_search) {
  check_exact_values<LBFGS>(
      LBFGSOptions(1.0).line_search_fn("strong_wolfe"),
      expected_parameters::LBFGS_with_line_search());
}

TEST(OptimTest, ZeroGrad) {
  torch::manual_seed(0);

  Linear model(2, 8);
  SGD optimizer(model->parameters(), 0.1);

  for (const auto& parameter : model->parameters()) {
    ASSERT_FALSE(parameter.grad().defined());
  }

  auto output = model->forward(torch::ones({5, 2}));
  auto loss = output.sum();
  loss.backward();

  for (const auto& parameter : model->parameters()) {
    ASSERT_TRUE(parameter.grad().defined());
    ASSERT_GT(parameter.grad().sum().item<float>(), 0);
  }

  optimizer.zero_grad();

  for (const auto& parameter : model->parameters()) {
    ASSERT_FALSE(parameter.grad().defined());
  }
}

TEST(OptimTest, ExternalVectorOfParameters) {
  torch::manual_seed(0);

  std::vector<torch::Tensor> parameters = {
      torch::randn({2, 2}), torch::randn({3, 3}), torch::randn({4, 4})};
  std::vector<torch::Tensor> original_parameters = {
      parameters[0].clone(), parameters[1].clone(), parameters[2].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.mutable_grad() = torch::ones_like(parameter);
  }

  SGD optimizer(parameters, 1.0);

  optimizer.step();

  ASSERT_TRUE(parameters[0].allclose(original_parameters[0] - 1.0));
  ASSERT_TRUE(parameters[1].allclose(original_parameters[1] - 1.0));
  ASSERT_TRUE(parameters[2].allclose(original_parameters[2] - 1.0));
}

TEST(OptimTest, AddParameter_LBFGS) {
  torch::manual_seed(0);

  std::vector<torch::Tensor> parameters = {torch::randn({5, 5})};
  std::vector<torch::Tensor> original_parameters = {parameters[0].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.mutable_grad() = torch::ones_like(parameter);
  }

  LBFGS optimizer(std::vector<torch::Tensor>{}, 1.0);
  OLD_INTERFACE_WARNING_CHECK(optimizer.add_parameters(parameters));

  optimizer.step([]() { return torch::tensor(1); });

  // REQUIRE this doesn't throw
}

// Check whether the learning rate of the parameter groups in the optimizer are
// the same as the expected learning rates given in the epoch:learning rate map
static void check_lr_change(
    Optimizer& optimizer,
    LRScheduler& lr_scheduler,
    std::map<unsigned, double> expected_epoch_lrs) {
  // Find maximum epoch in map
  unsigned kIterations = std::max_element(
                             expected_epoch_lrs.begin(),
                             expected_epoch_lrs.end(),
                             [](const std::pair<unsigned, double>& a,
                                const std::pair<unsigned, double>& b) -> bool {
                               return a.second > b.second;
                             })
                             ->first;

  for (unsigned i = 0; i <= kIterations; i++) {
    const auto epoch_iter = expected_epoch_lrs.find(i);
    if (epoch_iter != expected_epoch_lrs.end()) {
      // Compare the similarity of the two floating point learning rates
      ASSERT_TRUE(
          fabs(
              epoch_iter->second -
              optimizer.param_groups()[0].options().get_lr()) <
          std::numeric_limits<double>::epsilon());
    }
    optimizer.step();
    lr_scheduler.step();
  }
}

// Very similar to check_lr_change, but for ReduceLROnPlateauScheduler
// which does not inherit from LRScheduler and requires a metrics
// input to step().
static void check_lr_change_for_reduce_on_plateau(
    Optimizer& optimizer,
    ReduceLROnPlateauScheduler& lr_scheduler,
    std::map<unsigned, double> expected_epoch_lrs) {
  // Find maximum epoch in map
  unsigned kIterations = std::max_element(
                             expected_epoch_lrs.begin(),
                             expected_epoch_lrs.end(),
                             [](const std::pair<unsigned, double>& a,
                                const std::pair<unsigned, double>& b) -> bool {
                               return a.second > b.second;
                             })
                             ->first;

  for (unsigned i = 0; i <= kIterations; i++) {
    const auto epoch_iter = expected_epoch_lrs.find(i);
    if (epoch_iter != expected_epoch_lrs.end()) {
      // Compare the similarity of the two floating point learning rates
      ASSERT_TRUE(
          fabs(
              epoch_iter->second -
              optimizer.param_groups()[0].options().get_lr()) <
          std::numeric_limits<double>::epsilon());
    }
    optimizer.step();
    lr_scheduler.step(5.0);
  }
}

TEST(OptimTest, CheckLRChange_StepLR_Adam) {
  torch::Tensor parameters = torch::zeros({1});
  auto optimizer = Adam({parameters}, AdamOptions().lr(1e-3));

  const unsigned step_size = 20;
  const double gamma = 0.5;
  StepLR step_lr_scheduler(optimizer, step_size, gamma);

  // The learning rate should have halved at epoch 20
  const std::map<unsigned, double> expected_epoch_lrs = {{1, 1e-3}, {25, 5e-4}};

  check_lr_change(optimizer, step_lr_scheduler, expected_epoch_lrs);
}

TEST(OptimTest, CheckLRChange_ReduceLROnPlateau_Adam) {
  torch::Tensor parameters = torch::zeros({1});
  auto optimizer = Adam({parameters}, AdamOptions().lr(1e-3));
  const float factor = 0.5;
  const int patience = 20;
  ReduceLROnPlateauScheduler reduce_lr_on_plateau_scheduler(
      optimizer,
      ReduceLROnPlateauScheduler::SchedulerMode::min,
      factor,
      patience);

  // The learning rate should have halved at epoch 20
  const std::map<unsigned, double> expected_epoch_lrs = {{1, 1e-3}, {25, 5e-4}};

  check_lr_change_for_reduce_on_plateau(
      optimizer, reduce_lr_on_plateau_scheduler, expected_epoch_lrs);
}
// Tests for Issue 141884: Parameter group inheritance functionality
// Validates that partial options in parameter groups correctly inherit
// defaults from the optimizer while preserving explicitly set values
TEST(OptimTest, MergeWithDefaultOptions_Adam) {
  // Create tensors for parameter groups
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);

  // Create param groups with partial options
  std::vector<OptimizerParamGroup> param_groups;

  // Group 1: Only weight_decay specified, should inherit lr, betas, eps,
  // amsgrad
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor1},
      std::make_unique<AdamOptions>(AdamOptions().weight_decay(0.11)));

  // Group 2: Only eps specified, should inherit others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor2},
      std::make_unique<AdamOptions>(AdamOptions().eps(1e-6)));

  // Create optimizer with specific defaults
  AdamOptions defaults;
  defaults.lr(0.002)
      .betas(std::make_tuple(0.8, 0.88))
      .eps(1e-12)
      .weight_decay(0.05)
      .amsgrad(true);

  Adam optimizer(param_groups, defaults);

  // Check Group 1: weight_decay preserved, others inherited
  auto& group1_opts =
      static_cast<AdamOptions&>(optimizer.param_groups()[0].options());
  ASSERT_EQ(group1_opts.lr(), 0.002); // Inherited
  ASSERT_EQ(group1_opts.betas(), std::make_tuple(0.8, 0.88)); // Inherited
  ASSERT_EQ(group1_opts.eps(), 1e-12); // Inherited
  ASSERT_EQ(group1_opts.weight_decay(), 0.11); // Preserved
  ASSERT_TRUE(group1_opts.amsgrad()); // Inherited

  // Check Group 2: eps preserved, others inherited
  auto& group2_opts =
      static_cast<AdamOptions&>(optimizer.param_groups()[1].options());
  ASSERT_EQ(group2_opts.lr(), 0.002); // Inherited
  ASSERT_EQ(group2_opts.betas(), std::make_tuple(0.8, 0.88)); // Inherited
  ASSERT_EQ(group2_opts.eps(), 1e-6); // Preserved
  ASSERT_EQ(group2_opts.weight_decay(), 0.05); // Inherited
  ASSERT_TRUE(group2_opts.amsgrad()); // Inherited
}

TEST(OptimTest, MergeWithDefaultOptions_SGD) {
  // Create tensors for parameter groups
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);

  // Create param groups with partial options
  std::vector<OptimizerParamGroup> param_groups;

  // Group 1: Only lr and weight_decay specified, should inherit momentum,
  // dampening, nesterov
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor1},
      std::make_unique<SGDOptions>(SGDOptions(0.01).weight_decay(0.22)));

  // Group 2: Only lr specified, should inherit all others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor2},
      std::make_unique<SGDOptions>(SGDOptions(0.02)));

  // Create optimizer with specific defaults
  SGDOptions defaults(0.001); // lr should be overridden by param groups
  defaults.momentum(0.9)
      .dampening(0.0) // Must be 0 for Nesterov
      .weight_decay(0.05)
      .nesterov(true);

  SGD optimizer(param_groups, defaults);

  // Check Group 1: lr and weight_decay preserved, others inherited
  auto& group1_opts =
      static_cast<SGDOptions&>(optimizer.param_groups()[0].options());
  ASSERT_EQ(group1_opts.lr(), 0.01); // Preserved
  ASSERT_EQ(group1_opts.momentum(), 0.9); // Inherited
  ASSERT_EQ(group1_opts.dampening(), 0.0); // Inherited
  ASSERT_EQ(group1_opts.weight_decay(), 0.22); // Preserved
  ASSERT_TRUE(group1_opts.nesterov()); // Inherited

  // Check Group 2: lr preserved, others inherited
  auto& group2_opts =
      static_cast<SGDOptions&>(optimizer.param_groups()[1].options());
  ASSERT_EQ(group2_opts.lr(), 0.02); // Preserved
  ASSERT_EQ(group2_opts.momentum(), 0.9); // Inherited
  ASSERT_EQ(group2_opts.dampening(), 0.0); // Inherited
  ASSERT_EQ(group2_opts.weight_decay(), 0.05); // Inherited
  ASSERT_TRUE(group2_opts.nesterov()); // Inherited
}

TEST(OptimTest, MergeWithDefaultOptions_AdamW) {
  // Create tensors for parameter groups
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);

  // Create param groups with partial options
  std::vector<OptimizerParamGroup> param_groups;

  // Group 1: Only eps specified, should inherit others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor1},
      std::make_unique<AdamWOptions>(AdamWOptions().eps(1e-6)));

  // Group 2: Only betas specified, should inherit others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor2},
      std::make_unique<AdamWOptions>(
          AdamWOptions().betas(std::make_tuple(0.95, 0.999))));

  // Create optimizer with specific defaults
  AdamWOptions defaults;
  defaults.lr(0.003)
      .betas(std::make_tuple(0.9, 0.98))
      .eps(1e-8)
      .weight_decay(0.02)
      .amsgrad(false);

  AdamW optimizer(param_groups, defaults);

  // Check Group 1: eps preserved, others inherited
  auto& group1_opts =
      static_cast<AdamWOptions&>(optimizer.param_groups()[0].options());
  ASSERT_EQ(group1_opts.lr(), 0.003); // Inherited
  ASSERT_EQ(group1_opts.betas(), std::make_tuple(0.9, 0.98)); // Inherited
  ASSERT_EQ(group1_opts.eps(), 1e-6); // Preserved
  ASSERT_EQ(group1_opts.weight_decay(), 0.02); // Inherited
  ASSERT_FALSE(group1_opts.amsgrad()); // Inherited

  // Check Group 2: betas preserved, others inherited
  auto& group2_opts =
      static_cast<AdamWOptions&>(optimizer.param_groups()[1].options());
  ASSERT_EQ(group2_opts.lr(), 0.003); // Inherited
  ASSERT_EQ(group2_opts.betas(), std::make_tuple(0.95, 0.999)); // Preserved
  ASSERT_EQ(group2_opts.eps(), 1e-8); // Inherited
  ASSERT_EQ(group2_opts.weight_decay(), 0.02); // Inherited
  ASSERT_FALSE(group2_opts.amsgrad()); // Inherited
}

TEST(OptimTest, MergeWithDefaultOptions_Adagrad) {
  // Create tensors for parameter groups
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);

  // Create param groups with partial options
  std::vector<OptimizerParamGroup> param_groups;

  // Group 1: Only lr_decay specified, should inherit others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor1},
      std::make_unique<AdagradOptions>(AdagradOptions().lr_decay(0.001)));

  // Group 2: Only initial_accumulator_value specified, should inherit others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor2},
      std::make_unique<AdagradOptions>(
          AdagradOptions().initial_accumulator_value(0.5)));

  // Create optimizer with specific defaults
  AdagradOptions defaults;
  defaults.lr(0.04)
      .lr_decay(0.002)
      .weight_decay(0.03)
      .initial_accumulator_value(0.1)
      .eps(1e-11);

  Adagrad optimizer(param_groups, defaults);

  // Check Group 1: lr_decay preserved, others inherited
  auto& group1_opts =
      static_cast<AdagradOptions&>(optimizer.param_groups()[0].options());
  ASSERT_EQ(group1_opts.lr(), 0.04); // Inherited
  ASSERT_EQ(group1_opts.lr_decay(), 0.001); // Preserved
  ASSERT_EQ(group1_opts.weight_decay(), 0.03); // Inherited
  ASSERT_EQ(group1_opts.initial_accumulator_value(), 0.1); // Inherited
  ASSERT_EQ(group1_opts.eps(), 1e-11); // Inherited

  // Check Group 2: initial_accumulator_value preserved, others inherited
  auto& group2_opts =
      static_cast<AdagradOptions&>(optimizer.param_groups()[1].options());
  ASSERT_EQ(group2_opts.lr(), 0.04); // Inherited
  ASSERT_EQ(group2_opts.lr_decay(), 0.002); // Inherited
  ASSERT_EQ(group2_opts.weight_decay(), 0.03); // Inherited
  ASSERT_EQ(group2_opts.initial_accumulator_value(), 0.5); // Preserved
  ASSERT_EQ(group2_opts.eps(), 1e-11); // Inherited
}

TEST(OptimTest, MergeWithDefaultOptions_RMSprop) {
  // Create tensors for parameter groups
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);

  // Create param groups with partial options
  std::vector<OptimizerParamGroup> param_groups;

  // Group 1: Only alpha specified, should inherit others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor1},
      std::make_unique<RMSpropOptions>(RMSpropOptions().alpha(0.95)));

  // Group 2: Only momentum and centered specified, should inherit others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor2},
      std::make_unique<RMSpropOptions>(
          RMSpropOptions().momentum(0.8).centered(true)));

  // Create optimizer with specific defaults
  RMSpropOptions defaults;
  defaults.lr(0.015)
      .alpha(0.98)
      .eps(1e-9)
      .weight_decay(0.01)
      .momentum(0.7)
      .centered(false);

  RMSprop optimizer(param_groups, defaults);

  // Check Group 1: alpha preserved, others inherited
  auto& group1_opts =
      static_cast<RMSpropOptions&>(optimizer.param_groups()[0].options());
  ASSERT_EQ(group1_opts.lr(), 0.015); // Inherited
  ASSERT_EQ(group1_opts.alpha(), 0.95); // Preserved
  ASSERT_EQ(group1_opts.eps(), 1e-9); // Inherited
  ASSERT_EQ(group1_opts.weight_decay(), 0.01); // Inherited
  ASSERT_EQ(group1_opts.momentum(), 0.7); // Inherited
  ASSERT_FALSE(group1_opts.centered()); // Inherited

  // Check Group 2: momentum and centered preserved, others inherited
  auto& group2_opts =
      static_cast<RMSpropOptions&>(optimizer.param_groups()[1].options());
  ASSERT_EQ(group2_opts.lr(), 0.015); // Inherited
  ASSERT_EQ(group2_opts.alpha(), 0.98); // Inherited
  ASSERT_EQ(group2_opts.eps(), 1e-9); // Inherited
  ASSERT_EQ(group2_opts.weight_decay(), 0.01); // Inherited
  ASSERT_EQ(group2_opts.momentum(), 0.8); // Preserved
  ASSERT_TRUE(group2_opts.centered()); // Preserved
}

TEST(OptimTest, MergeWithDefaultOptions_LBFGS) {
  // Create tensors for single parameter group (LBFGS limitation)
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);

  // Create param group with partial options
  std::vector<OptimizerParamGroup> param_groups;

  // Single group: Only max_iter specified, should inherit others
  param_groups.emplace_back(
      std::vector<torch::Tensor>{
          tensor1, tensor2}, // Combine tensors in single group
      std::make_unique<LBFGSOptions>(LBFGSOptions().max_iter(15)));

  // Create optimizer with specific defaults
  LBFGSOptions defaults;
  defaults.lr(0.8)
      .max_iter(25)
      .max_eval(31) // Use same value that appears to be auto-calculated
      .tolerance_grad(1e-5)
      .tolerance_change(1e-8)
      .history_size(80)
      .line_search_fn("strong_wolfe");

  LBFGS optimizer(param_groups, defaults);

  // Check Group: max_iter preserved, others inherited
  auto& group_opts =
      static_cast<LBFGSOptions&>(optimizer.param_groups()[0].options());
  ASSERT_EQ(group_opts.lr(), 0.8); // Inherited
  ASSERT_EQ(group_opts.max_iter(), 15); // Preserved
  ASSERT_EQ(group_opts.max_eval(), 31); // Inherited
  ASSERT_EQ(group_opts.tolerance_grad(), 1e-5); // Inherited
  ASSERT_EQ(group_opts.tolerance_change(), 1e-8); // Inherited
  ASSERT_EQ(group_opts.history_size(), 80); // Inherited
  ASSERT_EQ(group_opts.line_search_fn(), "strong_wolfe"); // Inherited
}

TEST(OptimTest, MergeWithDefaultOptions_NoOptionsInheritance) {
  // Test that param groups without options get full defaults
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);

  std::vector<OptimizerParamGroup> param_groups;

  // Groups with no options - should inherit everything
  param_groups.emplace_back(std::vector<torch::Tensor>{tensor1});
  param_groups.emplace_back(std::vector<torch::Tensor>{tensor2});

  // Create optimizer with specific defaults
  AdamOptions defaults;
  defaults.lr(0.005)
      .betas(std::make_tuple(0.85, 0.95))
      .eps(1e-7)
      .weight_decay(0.08)
      .amsgrad(true);

  Adam optimizer(param_groups, defaults);

  // Both groups should have exactly the default options
  for (int i = 0; i < 2; i++) {
    auto& group_opts =
        static_cast<AdamOptions&>(optimizer.param_groups()[i].options());
    ASSERT_EQ(group_opts.lr(), 0.005);
    ASSERT_EQ(group_opts.betas(), std::make_tuple(0.85, 0.95));
    ASSERT_EQ(group_opts.eps(), 1e-7);
    ASSERT_EQ(group_opts.weight_decay(), 0.08);
    ASSERT_TRUE(group_opts.amsgrad());
  }
}

// Test that field tracking survives serialization/deserialization cycles
TEST(OptimTest, SerializationPreservesFieldTracking_Adam) {
  // Create tensors for parameter groups
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);

  // Create param groups with partial options using fluent API (marks fields as
  // explicit)
  std::vector<OptimizerParamGroup> param_groups;

  // Group 1: Only weight_decay and amsgrad explicitly set via fluent API
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor1},
      std::make_unique<AdamOptions>(
          AdamOptions().weight_decay(0.11).amsgrad(true)));

  // Group 2: Only eps explicitly set via fluent API
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor2},
      std::make_unique<AdamOptions>(AdamOptions().eps(1e-6)));

  // Create optimizer with specific defaults
  AdamOptions defaults;
  defaults.lr(0.002)
      .betas(std::make_tuple(0.8, 0.88))
      .eps(1e-12)
      .weight_decay(0.05)
      .amsgrad(false);

  Adam original_optimizer(param_groups, defaults);

  // Capture original state for comparison
  auto& orig_group1_opts =
      static_cast<AdamOptions&>(original_optimizer.param_groups()[0].options());
  auto& orig_group2_opts =
      static_cast<AdamOptions&>(original_optimizer.param_groups()[1].options());

  // Verify original state (sanity check)
  ASSERT_NEAR(orig_group1_opts.weight_decay(), 0.11, 1e-6); // Explicitly set
  ASSERT_TRUE(orig_group1_opts.amsgrad()); // Explicitly set
  ASSERT_NEAR(orig_group1_opts.lr(), 0.002, 1e-6); // Inherited
  ASSERT_NEAR(orig_group2_opts.eps(), 1e-6, 1e-9); // Explicitly set
  ASSERT_NEAR(orig_group2_opts.lr(), 0.002, 1e-6); // Inherited

  // Test serialization of the options objects (where field tracking lives)
  std::stringstream ss1, ss2;

  // Serialize the parameter group options
  {
    torch::serialize::OutputArchive archive;
    orig_group1_opts.serialize(archive);
    archive.save_to(ss1);
  }
  {
    torch::serialize::OutputArchive archive;
    orig_group2_opts.serialize(archive);
    archive.save_to(ss2);
  }

  // Create new options objects and deserialize
  AdamOptions loaded_group1_opts;
  AdamOptions loaded_group2_opts;

  {
    torch::serialize::InputArchive archive;
    archive.load_from(ss1);
    loaded_group1_opts.serialize(archive);
  }
  {
    torch::serialize::InputArchive archive;
    archive.load_from(ss2);
    loaded_group2_opts.serialize(archive);
  }

  // Verify that all parameter values are preserved after deserialization

  // Group 1: weight_decay and amsgrad should be preserved as explicitly set,
  // others inherited
  ASSERT_NEAR(loaded_group1_opts.lr(), 0.002, 1e-6); // Inherited
  ASSERT_EQ(
      loaded_group1_opts.betas(), std::make_tuple(0.8, 0.88)); // Inherited
  ASSERT_NEAR(loaded_group1_opts.eps(), 1e-12, 1e-15); // Inherited
  ASSERT_NEAR(loaded_group1_opts.weight_decay(), 0.11, 1e-6); // Explicitly set
  ASSERT_TRUE(loaded_group1_opts.amsgrad()); // Explicitly set

  // Group 2: eps should be preserved as explicitly set, others inherited
  ASSERT_NEAR(loaded_group2_opts.lr(), 0.002, 1e-6); // Inherited
  ASSERT_EQ(
      loaded_group2_opts.betas(), std::make_tuple(0.8, 0.88)); // Inherited
  ASSERT_NEAR(loaded_group2_opts.eps(), 1e-6, 1e-9); // Explicitly set
  ASSERT_NEAR(loaded_group2_opts.weight_decay(), 0.05, 1e-6); // Inherited
  ASSERT_FALSE(loaded_group2_opts.amsgrad()); // Inherited

  // CRITICAL: Test that field tracking is preserved after serialization
  // Create a new optimizer using the deserialized options to test inheritance
  auto tensor3 = torch::randn({2, 2}).requires_grad_(true);
  auto tensor4 = torch::randn({3, 3}).requires_grad_(true);

  std::vector<OptimizerParamGroup> test_param_groups;
  test_param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor3},
      std::make_unique<AdamOptions>(loaded_group1_opts));
  test_param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor4},
      std::make_unique<AdamOptions>(loaded_group2_opts));

  Adam test_optimizer(test_param_groups, defaults);

  // The field tracking should work correctly for inheritance
  auto& final_group1_opts =
      static_cast<AdamOptions&>(test_optimizer.param_groups()[0].options());
  auto& final_group2_opts =
      static_cast<AdamOptions&>(test_optimizer.param_groups()[1].options());

  // Group 1: weight_decay and amsgrad should still be preserved as explicitly
  // set
  ASSERT_NEAR(
      final_group1_opts.weight_decay(),
      0.11,
      1e-6); // Explicitly set (preserved)
  ASSERT_TRUE(final_group1_opts.amsgrad()); // Explicitly set (preserved)
  ASSERT_NEAR(final_group1_opts.lr(), 0.002, 1e-6); // Inherited from defaults

  // Group 2: eps should still be preserved as explicitly set
  ASSERT_NEAR(
      final_group2_opts.eps(), 1e-6, 1e-9); // Explicitly set (preserved)
  ASSERT_NEAR(final_group2_opts.lr(), 0.002, 1e-6); // Inherited from defaults
}

// Test serialization with SGD (different parameter types)
TEST(OptimTest, SerializationPreservesFieldTracking_SGD) {
  // Create tensors
  auto tensor1 = torch::randn({2, 2}).requires_grad_(true);

  // Create param group with partial options using fluent API
  std::vector<OptimizerParamGroup> param_groups;
  param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor1},
      std::make_unique<SGDOptions>(
          SGDOptions(0.01).weight_decay(0.22).nesterov(true)));

  // Create optimizer with defaults
  SGDOptions defaults(0.001);
  defaults.momentum(0.9).dampening(0.0).weight_decay(0.05).nesterov(false);

  SGD original_optimizer(param_groups, defaults);

  // Test serialization of the SGD options (where field tracking lives)
  auto& original_opts =
      static_cast<SGDOptions&>(original_optimizer.param_groups()[0].options());

  std::stringstream ss;
  {
    torch::serialize::OutputArchive archive;
    original_opts.serialize(archive);
    archive.save_to(ss);
  }

  SGDOptions loaded_opts(0.0); // Dummy initial value
  {
    torch::serialize::InputArchive archive;
    archive.load_from(ss);
    loaded_opts.serialize(archive);
  }
  ASSERT_NEAR(loaded_opts.lr(), 0.01, 1e-6); // Explicitly set
  ASSERT_NEAR(loaded_opts.momentum(), 0.9, 1e-6); // Inherited
  ASSERT_NEAR(loaded_opts.dampening(), 0.0, 1e-6); // Inherited
  ASSERT_NEAR(loaded_opts.weight_decay(), 0.22, 1e-6); // Explicitly set
  ASSERT_TRUE(loaded_opts.nesterov()); // Explicitly set

  // Test that field tracking still works after deserialization by creating new
  // optimizer
  auto tensor2 = torch::randn({3, 3}).requires_grad_(true);
  std::vector<OptimizerParamGroup> test_param_groups;
  test_param_groups.emplace_back(
      std::vector<torch::Tensor>{tensor2},
      std::make_unique<SGDOptions>(loaded_opts));

  SGD test_optimizer(test_param_groups, defaults);

  auto& final_opts =
      static_cast<SGDOptions&>(test_optimizer.param_groups()[0].options());
  ASSERT_NEAR(final_opts.lr(), 0.01, 1e-6); // Explicitly set (preserved)
  ASSERT_NEAR(
      final_opts.weight_decay(), 0.22, 1e-6); // Explicitly set (preserved)
  ASSERT_TRUE(final_opts.nesterov()); // Explicitly set (preserved)
  ASSERT_NEAR(final_opts.momentum(), 0.9, 1e-6); // Inherited from defaults
  ASSERT_NEAR(final_opts.dampening(), 0.0, 1e-6); // Inherited from defaults
}
