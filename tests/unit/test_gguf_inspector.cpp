/**
 * @file test_gguf_inspector.cpp
 * @brief Unit tests for core::GgufInspector + auto_configure heuristics.
 */

#include "zoo/core/gguf_inspector.hpp"
#include "zoo/core/json.hpp"
#include "zoo/core/model_info.hpp"
#include "zoo/core/system_probe.hpp"
#include "zoo/core/types.hpp"

#include <gtest/gtest.h>
#include <llama.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <string>

namespace {

constexpr uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;

std::filesystem::path project_source_dir() {
#ifdef ZOO_PROJECT_SOURCE_DIR
    return ZOO_PROJECT_SOURCE_DIR;
#else
    return std::filesystem::current_path().parent_path();
#endif
}

std::filesystem::path fixture_vocab_model_path() {
    return project_source_dir() / "tests/fixtures/ggml-vocab-gpt-2.gguf";
}

void sentinel_log_callback(ggml_log_level, const char*, void*) {}

class ScopedLlamaLogger {
  public:
    ScopedLlamaLogger(ggml_log_callback callback, void* user_data) {
        llama_log_get(&original_callback_, &original_user_data_);
        llama_log_set(callback, user_data);
    }

    ~ScopedLlamaLogger() {
        llama_log_set(original_callback_, original_user_data_);
    }

  private:
    ggml_log_callback original_callback_ = nullptr;
    void* original_user_data_ = nullptr;
};

zoo::core::ModelInfo make_synthetic_info(uint64_t file_size, int layers, int embd, int ctx_train,
                                         int heads = 0, int kv_heads = 0) {
    zoo::core::ModelInfo info;
    info.file_path = "/models/synthetic.gguf";
    info.file_size_bytes = file_size;
    info.layer_count = layers;
    info.embedding_dim = embd;
    info.head_count = heads;
    info.kv_head_count = kv_heads;
    info.context_length = ctx_train;
    return info;
}

zoo::core::SystemInfo make_system(uint64_t total_ram, bool gpu_supported, uint64_t total_vram) {
    zoo::core::SystemInfo sys;
    sys.total_ram_bytes = total_ram;
    sys.available_ram_bytes = total_ram;
    sys.logical_cpu_count = 8;
    sys.gpu_offload_supported = gpu_supported;
    if (gpu_supported && total_vram > 0) {
        zoo::core::GpuInfo gpu;
        gpu.name = "synthetic";
        gpu.total_vram_bytes = total_vram;
        gpu.free_vram_bytes = total_vram;
        sys.gpus.push_back(gpu);
    }
    return sys;
}

} // namespace

// ---- ModelInfo equality ----

TEST(ModelInfoTest, DefaultEquality) {
    zoo::core::ModelInfo a;
    zoo::core::ModelInfo b;
    EXPECT_EQ(a, b);
}

TEST(ModelInfoTest, DifferentNameNotEqual) {
    zoo::core::ModelInfo a;
    a.name = "model-a";
    zoo::core::ModelInfo b;
    b.name = "model-b";
    EXPECT_NE(a, b);
}

// ---- auto_configure(info, sys) heuristics ----

TEST(AutoConfigTest, EmptyPathReturnsError) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 8192);
    info.file_path = "";
    auto sys = make_system(64ULL * kGiB, true, 24ULL * kGiB);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_FALSE(config.has_value());
    EXPECT_EQ(config.error().code, zoo::ErrorCode::InvalidModelPath);
}

TEST(AutoConfigTest, SetsModelPath) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 4096);
    auto sys = make_system(32ULL * kGiB, false, 0);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->model_path, "/models/synthetic.gguf");
}

TEST(AutoConfigTest, FullGpuOffloadWhenVramExceedsModelSize) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 8192);
    auto sys = make_system(64ULL * kGiB, true, 24ULL * kGiB);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->n_gpu_layers, -1);
}

TEST(AutoConfigTest, NoGpuOffloadWhenUnsupported) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 8192);
    auto sys = make_system(64ULL * kGiB, false, 0);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->n_gpu_layers, 0);
}

TEST(AutoConfigTest, PartialGpuOffloadWhenVramTooSmall) {
    auto info = make_synthetic_info(8ULL * kGiB, 32, 4096, 8192);
    auto sys = make_system(64ULL * kGiB, true, 4ULL * kGiB);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_GT(config->n_gpu_layers, 0);
    EXPECT_LT(config->n_gpu_layers, info.layer_count);
}

TEST(AutoConfigTest, ContextCappedByTrainingContext) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 2048);
    auto sys = make_system(128ULL * kGiB, true, 24ULL * kGiB);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->context_size, 2048);
}

TEST(AutoConfigTest, ContextRespectsHardCap) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 131072);
    auto sys = make_system(256ULL * kGiB, true, 80ULL * kGiB);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_LE(config->context_size, 32768);
}

TEST(AutoConfigTest, ContextHasFloorOnTinyRam) {
    auto info = make_synthetic_info(8ULL * kGiB, 32, 4096, 8192);
    auto sys = make_system(8ULL * kGiB, false, 0); // RAM == file size, no headroom

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_GE(config->context_size, 512);
}

TEST(AutoConfigTest, MlockOnlyWhenRamComfortablyExceedsModel) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 8192);

    auto small_ram = make_system(8ULL * kGiB, false, 0); // 2x model size — boundary
    auto small_cfg = zoo::core::GgufInspector::auto_configure(info, small_ram);
    ASSERT_TRUE(small_cfg.has_value());
    EXPECT_FALSE(small_cfg->use_mlock);

    auto big_ram = make_system(64ULL * kGiB, false, 0);
    auto big_cfg = zoo::core::GgufInspector::auto_configure(info, big_ram);
    ASSERT_TRUE(big_cfg.has_value());
    EXPECT_TRUE(big_cfg->use_mlock);
}

TEST(AutoConfigTest, MmapAlwaysEnabled) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 8192);
    auto sys = make_system(8ULL * kGiB, false, 0);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_TRUE(config->use_mmap);
}

TEST(AutoConfigTest, NBatchTracksContextWithCap) {
    auto small_info = make_synthetic_info(1ULL * kGiB, 32, 4096, 1024);
    auto sys = make_system(64ULL * kGiB, false, 0);
    auto small_cfg = zoo::core::GgufInspector::auto_configure(small_info, sys);
    ASSERT_TRUE(small_cfg.has_value());
    EXPECT_EQ(small_cfg->n_batch, small_cfg->context_size); // ≤ 2048 cap → equals context

    auto big_info = make_synthetic_info(4ULL * kGiB, 32, 4096, 32768);
    auto big_cfg = zoo::core::GgufInspector::auto_configure(big_info, sys);
    ASSERT_TRUE(big_cfg.has_value());
    EXPECT_EQ(big_cfg->n_batch, 2048); // capped
    EXPECT_GE(big_cfg->context_size, big_cfg->n_batch);
}

TEST(AutoConfigTest, GqaModelGetsLargerContextThanMha) {
    constexpr uint64_t kFileSize = 8ULL * kGiB;
    constexpr int kLayers = 32;
    constexpr int kEmbd = 4096;
    constexpr int kCtxTrain = 32768;

    // Same model size; one is full MHA (heads == kv_heads), other is 4-to-1 GQA.
    auto mha = make_synthetic_info(kFileSize, kLayers, kEmbd, kCtxTrain, 32, 32);
    auto gqa = make_synthetic_info(kFileSize, kLayers, kEmbd, kCtxTrain, 32, 8);

    auto sys = make_system(20ULL * kGiB, false, 0); // tight RAM forces KV-budget binding

    auto mha_cfg = zoo::core::GgufInspector::auto_configure(mha, sys);
    auto gqa_cfg = zoo::core::GgufInspector::auto_configure(gqa, sys);
    ASSERT_TRUE(mha_cfg.has_value());
    ASSERT_TRUE(gqa_cfg.has_value());

    EXPECT_GT(gqa_cfg->context_size, mha_cfg->context_size);
}

TEST(AutoConfigTest, ContextUsesAvailableRamWhenSet) {
    auto info = make_synthetic_info(8ULL * kGiB, 32, 4096, 32768, 32, 8);
    auto sys_total = make_system(64ULL * kGiB, false, 0);
    auto sys_pressure = make_system(64ULL * kGiB, false, 0);
    sys_pressure.available_ram_bytes = 12ULL * kGiB; // most RAM in use elsewhere

    auto big = zoo::core::GgufInspector::auto_configure(info, sys_total);
    auto pressured = zoo::core::GgufInspector::auto_configure(info, sys_pressure);
    ASSERT_TRUE(big.has_value());
    ASSERT_TRUE(pressured.has_value());

    // Under memory pressure, the derived context window must shrink.
    EXPECT_LT(pressured->context_size, big->context_size);
}

TEST(AutoConfigTest, GpuOffloadUsesFreeVramWhenSet) {
    auto info = make_synthetic_info(4ULL * kGiB, 32, 4096, 8192);
    auto sys = make_system(64ULL * kGiB, true, 24ULL * kGiB);
    // Reserve most VRAM for other workloads.
    sys.gpus.front().free_vram_bytes = 1ULL * kGiB;

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_NE(config->n_gpu_layers, -1); // not full offload — VRAM is occupied
}

TEST(AutoConfigTest, ZeroLayerCountFallsBack) {
    auto info = make_synthetic_info(1ULL * kGiB, 0, 0, 4096);
    auto sys = make_system(32ULL * kGiB, true, 24ULL * kGiB);

    auto config = zoo::core::GgufInspector::auto_configure(info, sys);
    ASSERT_TRUE(config.has_value());
    EXPECT_GE(config->context_size, 512);
    EXPECT_LE(config->context_size, 32768);
}

// ---- Inspector regression coverage ----

// ---- load_model_config(json) ----

TEST(LoadModelConfigTest, ReturnsBaseConfigWhenAutoConfigureNotRequested) {
    nlohmann::json j = {{"model_path", "/models/example.gguf"}, {"context_size", 4096}};

    auto config = zoo::load_model_config(j);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->model_path, "/models/example.gguf");
    EXPECT_EQ(config->context_size, 4096);
}

TEST(LoadModelConfigTest, ReturnsBaseConfigWhenAutoConfigureFalse) {
    nlohmann::json j = {{"model_path", "/models/example.gguf"}, {"auto_configure", false}};

    auto config = zoo::load_model_config(j);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->model_path, "/models/example.gguf");
}

TEST(LoadModelConfigTest, AutoConfigureFailsForMissingFile) {
    nlohmann::json j = {{"model_path", "/does/not/exist.gguf"}, {"auto_configure", true}};

    auto config = zoo::load_model_config(j);
    ASSERT_FALSE(config.has_value());
    EXPECT_EQ(config.error().code, zoo::ErrorCode::GgufReadFailed);
}

TEST(GgufInspectorTest, RestoresGlobalLoggerAfterInspect) {
    const auto model_path = fixture_vocab_model_path();
    ASSERT_TRUE(std::filesystem::exists(model_path)) << model_path.string();

    void* const sentinel_user_data = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234));
    ScopedLlamaLogger logger(sentinel_log_callback, sentinel_user_data);

    auto result = zoo::core::GgufInspector::inspect(model_path.string());
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    ggml_log_callback current_callback = nullptr;
    void* current_user_data = nullptr;
    llama_log_get(&current_callback, &current_user_data);

    EXPECT_EQ(current_callback, sentinel_log_callback);
    EXPECT_EQ(current_user_data, sentinel_user_data);
}
