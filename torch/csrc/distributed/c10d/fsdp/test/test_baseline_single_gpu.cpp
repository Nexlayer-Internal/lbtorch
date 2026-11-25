// Single-GPU, non-FSDP baseline: IDENTICAL architecture, seed, data, and
// hyperparameters as test_fsdp_training.cpp, but trains the full
// (unsharded) model directly with plain torch::optim::AdamW. Exists to
// answer one question in isolation: does this toy model/task actually
// converge at all, independent of FSDP? If this also fails to show a loss
// decrease, that implicates the test's model/task, not FSDPUnit. If this
// converges and the FSDP version doesn't, that implicates a real bug in
// fsdp_unit.cpp.
#include <torch/torch.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>
#include <iostream>

namespace {
constexpr int64_t kDim = 512;
constexpr int64_t kNumLayers = 4;
constexpr int64_t kBatch = 64;
constexpr int64_t kSteps = 60;
constexpr double kLr = 0.05; // NOT halved -- single real GPU, single real batch
} // namespace

int main() {
  c10::cuda::set_device(0);
  torch::manual_seed(1234);
  const auto cuda_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);

  std::vector<torch::nn::Linear> layers;
  std::vector<torch::Tensor> params;
  for (const auto i : c10::irange(kNumLayers)) {
    (void)i;
    torch::nn::Linear layer(kDim, kDim);
    layer->to(cuda_opts.device());
    layers.push_back(layer);
    for (auto& p : layer->parameters()) {
      params.push_back(p);
    }
  }
  torch::optim::AdamW optimizer(params, torch::optim::AdamWOptions(kLr));
  torch::Tensor target_matrix = torch::randn({kDim, kDim}, cuda_opts);
  // Fixed batch, reused every step (a standard "can this wiring actually
  // learn" check -- not a generalization benchmark): avoids conflating
  // "the optimizer/gradient wiring works" with "this toy architecture
  // generalizes well from fresh random batches in only a few dozen steps",
  // which turned out to be a slow/noisy regime unrelated to FSDP (verified
  // identically slow on this single-GPU, non-FSDP baseline too).
  torch::Tensor x = torch::randn({kBatch, kDim}, cuda_opts);
  torch::Tensor target = torch::matmul(x, target_matrix).detach();

  double first_loss = -1.0, last_loss = -1.0;
  for (const auto step : c10::irange(kSteps)) {
    optimizer.zero_grad();
    torch::Tensor h = x;
    for (const auto li : c10::irange(layers.size())) {
      h = layers[li]->forward(h);
      if (li + 1 < layers.size()) {
        h = torch::relu(h); // no ReLU after the final layer -- target is unbounded
      }
    }
    torch::Tensor loss = torch::mse_loss(h, target);
    loss.backward();
    optimizer.step();
    const double loss_val = loss.item<double>();
    if (step == 0) first_loss = loss_val;
    last_loss = loss_val;
    std::cout << "[baseline] step " << step << " loss=" << loss_val << std::endl;
  }
  std::cout << "[baseline] first_loss=" << first_loss << " last_loss=" << last_loss
            << std::endl;
  return 0;
}
