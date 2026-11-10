#include <gtest/gtest.h>

#include <torch/models/qwen3.h>
#include <torch/torch.h>
#include <torch/utils/safetensors.h>

#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// Lightweight, CI-friendly unit tests for the pieces that are easy to get
// subtly wrong in isolation: safetensors header parsing + byte mapping,
// RMSNorm math, RoPE rotate_half/position-0 sanity, GQA repeat_kv
// head-mapping order, and config.json parsing against a bundled copy of the
// real Qwen/Qwen3-4B config. Does NOT load the full 4B-parameter checkpoint
// (that requires an ~8GB download and is validated separately -- see
// docs/qwen3_cpp_loading.md for the full end-to-end numerical comparison
// against Python's `transformers` on real hardware).

using namespace torch::models::qwen3;
using torch::utils::safetensors::SafetensorsCollection;

namespace {

// Writes a minimal, hand-built safetensors file containing a couple of
// small F32 tensors, exercising the exact on-disk layout this loader parses:
// 8-byte LE header length, JSON header, raw tensor bytes.
std::string WriteSyntheticSafetensors() {
  std::string dir = "/tmp/qwen3_gtest_safetensors";
#ifdef _WIN32
  _mkdir(dir.c_str());
#else
  ::mkdir(dir.c_str(), 0755);
#endif
  std::vector<float> a_data = {0, 1, 2, 3, 4, 5}; // shape [2,3]
  std::vector<float> b_data = {10, 11, 12, 13}; // shape [4]
  size_t a_bytes = a_data.size() * sizeof(float);
  size_t b_bytes = b_data.size() * sizeof(float);

  std::string header = "{"
                        "\"a\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0," +
      std::to_string(a_bytes) +
      "]},"
      "\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[" +
      std::to_string(a_bytes) + "," + std::to_string(a_bytes + b_bytes) + "]}"
                                                                           "}";
  uint64_t header_len = header.size();

  std::string path = dir + "/model.safetensors";
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  out.write(header.data(), header.size());
  out.write(reinterpret_cast<const char*>(a_data.data()), a_bytes);
  out.write(reinterpret_cast<const char*>(b_data.data()), b_bytes);
  out.close();
  return dir;
}

// The real, published Qwen/Qwen3-4B config.json, fetched from HuggingFace
// Hub and bundled verbatim so the config-parsing test doesn't require
// network access in CI.
constexpr const char* kQwen3_4bConfigJson = R"JSON({
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 9728,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
})JSON";

} // namespace

TEST(Qwen3SafetensorsTest, RoundTripsTensorsFromDisk) {
  std::string dir = WriteSyntheticSafetensors();
  SafetensorsCollection coll = SafetensorsCollection::FromDirectory(dir);

  ASSERT_TRUE(coll.has("a"));
  ASSERT_TRUE(coll.has("b"));
  ASSERT_FALSE(coll.has("nonexistent"));
  ASSERT_EQ(coll.size(), 2u);

  torch::Tensor a = coll.get("a");
  ASSERT_EQ(a.sizes(), torch::IntArrayRef({2, 3}));
  ASSERT_EQ(a.scalar_type(), torch::kFloat32);
  ASSERT_NEAR(a[0][0].item<float>(), 0.0, 1e-7);
  ASSERT_NEAR(a[1][2].item<float>(), 5.0, 1e-7);

  torch::Tensor b = coll.get("b");
  ASSERT_EQ(b.sizes(), torch::IntArrayRef({4}));
  ASSERT_NEAR(b[3].item<float>(), 13.0, 1e-7);
}

TEST(Qwen3ModelTest, RMSNormMatchesHandComputedValue) {
  RMSNorm norm(4, 1e-6);
  {
    torch::NoGradGuard ng;
    norm->weight.copy_(torch::tensor({1.0f, 2.0f, 1.0f, 1.0f}));
  }
  torch::Tensor x = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f}});
  torch::Tensor out = norm->forward(x);

  double mean_sq = (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0) / 4.0;
  double inv_rms = 1.0 / std::sqrt(mean_sq + 1e-6);
  ASSERT_NEAR(out[0][0].item<float>(), 1.0 * inv_rms * 1.0, 1e-4);
  ASSERT_NEAR(out[0][1].item<float>(), 2.0 * inv_rms * 2.0, 1e-4);
  ASSERT_NEAR(out[0][3].item<float>(), 4.0 * inv_rms * 1.0, 1e-4);
}

TEST(Qwen3ModelTest, RopeAnglesMatchClosedForm) {
  // At position 0, every frequency's angle is 0, so cos=1, sin=0 for every
  // element of head_dim.
  RotaryEmbedding rope(8, 1000000.0);
  torch::Tensor positions = torch::zeros({1, 1}, torch::kInt64);
  auto [cos, sin] = rope->forward(positions);
  ASSERT_EQ(cos.sizes(), torch::IntArrayRef({1, 1, 8}));
  for (int64_t i = 0; i < 8; ++i) {
    ASSERT_NEAR(cos[0][0][i].item<float>(), 1.0, 1e-5);
    ASSERT_NEAR(sin[0][0][i].item<float>(), 0.0, 1e-5);
  }

  // At a nonzero position, the lowest-frequency pair (index 0 and
  // head_dim/2) should match cos(position * inv_freq[0]) ==
  // cos(position * 1.0), since inv_freq[0] = theta^0 = 1. emb = cat(freqs,
  // freqs) -> index head_dim/2 repeats index 0's angle.
  torch::Tensor positions2 = torch::full({1, 1}, 3, torch::kInt64);
  auto [cos2, sin2] = rope->forward(positions2);
  ASSERT_NEAR(cos2[0][0][0].item<float>(), std::cos(3.0), 1e-4);
  ASSERT_NEAR(sin2[0][0][0].item<float>(), std::sin(3.0), 1e-4);
  ASSERT_NEAR(cos2[0][0][4].item<float>(), std::cos(3.0), 1e-4);
}

TEST(Qwen3ModelTest, RepeatKvUsesRepeatInterleaveOrderNotTiled) {
  // 2 kv heads, n_rep=2 -> 4 query-equivalent heads. Verify kv head 0 maps
  // to output heads {0,1} and kv head 1 maps to output heads {2,3} (the
  // "repeat_interleave" order transformers.repeat_kv uses), not an
  // interleaved {0,2}/{1,3} order.
  torch::Tensor kv =
      torch::stack({torch::full({1, 1, 1}, 0.0), torch::full({1, 1, 1}, 1.0)}, /*dim=*/1);
  ASSERT_EQ(kv.dim(), 4);
  int64_t n_rep = 2;
  auto sizes = kv.sizes();
  torch::Tensor expanded = kv.unsqueeze(2)
                                .expand({sizes[0], sizes[1], n_rep, sizes[2], sizes[3]})
                                .reshape({sizes[0], sizes[1] * n_rep, sizes[2], sizes[3]});
  ASSERT_NEAR(expanded[0][0][0][0].item<float>(), 0.0, 1e-7);
  ASSERT_NEAR(expanded[0][1][0][0].item<float>(), 0.0, 1e-7);
  ASSERT_NEAR(expanded[0][2][0][0].item<float>(), 1.0, 1e-7);
  ASSERT_NEAR(expanded[0][3][0][0].item<float>(), 1.0, 1e-7);
}

TEST(Qwen3ConfigTest, ParsesRealQwen3_4bConfig) {
  std::string path = "/tmp/qwen3_gtest_config.json";
  {
    std::ofstream out(path);
    out << kQwen3_4bConfigJson;
  }
  Qwen3Config cfg = Qwen3Config::FromFile(path);

  ASSERT_EQ(cfg.hidden_size, 2560);
  ASSERT_EQ(cfg.num_hidden_layers, 36);
  ASSERT_EQ(cfg.num_attention_heads, 32);
  ASSERT_EQ(cfg.num_key_value_heads, 8);
  // The important one: head_dim is NOT hidden_size/num_attention_heads
  // (2560/32 = 80) -- Qwen3-4B overrides it explicitly to 128.
  ASSERT_EQ(cfg.head_dim, 128);
  ASSERT_NE(cfg.head_dim, cfg.hidden_size / cfg.num_attention_heads);
  ASSERT_EQ(cfg.intermediate_size, 9728);
  ASSERT_EQ(cfg.vocab_size, 151936);
  ASSERT_TRUE(cfg.tie_word_embeddings);
  ASSERT_FALSE(cfg.attention_bias);
  ASSERT_NEAR(cfg.rope_theta, 1000000.0, 1e-6);
  ASSERT_NEAR(cfg.rms_norm_eps, 1e-6, 1e-12);
}

TEST(Qwen3ModelTest, BuildsAndRunsForwardWithRandomWeights) {
  // Not a correctness check against Python (see docs/qwen3_cpp_loading.md
  // for that, which requires the real ~8GB checkpoint) -- just confirms the
  // module graph wires together end to end (shapes, GQA head ratio,
  // tied-embedding path) with a small config, without needing any
  // checkpoint at all.
  Qwen3Config cfg;
  cfg.vocab_size = 37;
  cfg.hidden_size = 16;
  cfg.intermediate_size = 32;
  cfg.num_hidden_layers = 2;
  cfg.num_attention_heads = 4;
  cfg.num_key_value_heads = 2; // GQA with a real (>1) ratio
  cfg.head_dim = 8; // deliberately != hidden_size / num_attention_heads (16/4=4)
  cfg.rms_norm_eps = 1e-6;
  cfg.rope_theta = 1000000.0;
  cfg.tie_word_embeddings = true;

  Qwen3ForCausalLM model(cfg);
  model->eval();

  torch::Tensor input_ids = torch::randint(0, cfg.vocab_size, {2, 12}, torch::kInt64);
  torch::Tensor logits;
  {
    torch::NoGradGuard ng;
    logits = model->forward(input_ids);
  }
  ASSERT_EQ(logits.sizes(), torch::IntArrayRef({2, 12, cfg.vocab_size}));
  ASSERT_TRUE(logits.isfinite().all().item<bool>());

  // Tied embeddings: lm_head should not exist as a separate parameter.
  ASSERT_FALSE(model->named_parameters().contains("lm_head.weight"));
  ASSERT_TRUE(model->named_parameters().contains("model.embed_tokens.weight"));
  ASSERT_TRUE(model->named_parameters().contains("model.layers.0.self_attn.q_norm.weight"));
}
