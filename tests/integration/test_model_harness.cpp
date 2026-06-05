/**
 * @file test_model_harness.cpp
 * @brief Integration coverage for the concrete Model harness.
 */

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#include "zoo/core/json.hpp"
#include "zoo/zoo.hpp"

namespace {

std::filesystem::path project_source_dir() {
    return std::filesystem::path{ZOO_PROJECT_SOURCE_DIR};
}

std::filesystem::path fixture_vocab_model_path() {
    return project_source_dir() / "tests/fixtures/ggml-vocab-gpt-2.gguf";
}

std::optional<std::filesystem::path> live_model_path() {
#ifdef ZOO_INTEGRATION_MODEL_PATH
    if (std::filesystem::path configured{ZOO_INTEGRATION_MODEL_PATH}; !configured.empty()) {
        return configured;
    }
#endif

    if (const char* env = std::getenv("ZOO_INTEGRATION_MODEL")) {
        if (*env != '\0') {
            return std::filesystem::path{env};
        }
    }

    return std::nullopt;
}

struct TestConfig {
    zoo::ModelConfig model;
    zoo::GenerationOptions generation;
};

TestConfig make_base_config(const std::filesystem::path& model_path) {
    TestConfig config;
    config.model.model_path = model_path.string();
    config.model.context_size = 2048;
    config.model.n_gpu_layers = 0;
    config.generation.max_tokens = 24;
    config.generation.sampling.temperature = 0.0f;
    config.generation.sampling.top_p = 1.0f;
    config.generation.sampling.top_k = 1;
    config.generation.sampling.seed = 7;
    return config;
}

class LiveModelIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        auto model_path = live_model_path();
        if (!model_path.has_value()) {
            GTEST_SKIP() << "Set ZOO_INTEGRATION_MODEL to run live generation smoke tests.";
        }

        if (!std::filesystem::exists(*model_path)) {
            GTEST_SKIP() << "Configured integration model does not exist: " << model_path->string();
        }

        model_path_ = *model_path;
    }

    TestConfig config() const {
        return make_base_config(model_path_);
    }

    std::filesystem::path model_path_;
};

} // namespace

TEST(ModelIntegrationTest, LoadRejectsIncompleteVendoredFixture) {
    const auto model_path = fixture_vocab_model_path();
    ASSERT_TRUE(std::filesystem::exists(model_path))
        << "Expected vendored llama.cpp vocabulary fixture at " << model_path.string();

    auto cfg = make_base_config(model_path);
    auto result = zoo::Model::load(cfg.model, cfg.generation);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ModelLoadFailed);
}

TEST_F(LiveModelIntegrationTest, AutoConfiguredCpuOverrideLoadsModel) {
    const nlohmann::json config_json = {{"model_path", model_path_.string()},
                                        {"auto_configure", true},
                                        {"context_size", 2048},
                                        {"n_gpu_layers", 0}};

    auto config_result = zoo::load_model_config(config_json);
    ASSERT_TRUE(config_result.has_value()) << config_result.error().to_string();

    const auto& model_config = *config_result;
    EXPECT_EQ(model_config.context_size, 2048);
    EXPECT_EQ(model_config.n_gpu_layers, 0);
    EXPECT_GT(model_config.n_batch, 0);
    EXPECT_LE(model_config.n_batch, model_config.context_size);

    std::error_code ec;
    const bool same_model = std::filesystem::equivalent(
        std::filesystem::path{model_config.model_path}, model_path_, ec);
    EXPECT_FALSE(ec) << ec.message();
    EXPECT_TRUE(same_model) << "Expected auto-configured path to match " << model_path_.string()
                            << ", got " << model_config.model_path;

    auto cfg = config();
    auto model_result = zoo::Model::load(model_config, cfg.generation);
    ASSERT_TRUE(model_result.has_value()) << model_result.error().to_string();
}

TEST_F(LiveModelIntegrationTest, ModelGeneratesAndTracksHistory) {
    const auto cfg = config();
    auto model_result = zoo::Model::load(cfg.model, cfg.generation);
    ASSERT_TRUE(model_result.has_value()) << model_result.error().to_string();

    auto& model = *model_result;
    model->set_system_prompt("Reply briefly.");

    auto response = model->generate("Say hello in one short sentence.");
    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());

    const auto history = model->get_history();
    ASSERT_GE(history.size(), 3u);
    EXPECT_EQ(history[0].role, zoo::Role::System);
    EXPECT_EQ(history[1].role, zoo::Role::User);
    EXPECT_EQ(history[history.size() - 1].role, zoo::Role::Assistant);
}

TEST_F(LiveModelIntegrationTest, CompleteDoesNotMutatePersistentHistory) {
    const auto cfg = config();
    auto model_result = zoo::Model::load(cfg.model, cfg.generation);
    ASSERT_TRUE(model_result.has_value()) << model_result.error().to_string();

    auto& model = *model_result;
    model->set_system_prompt("Reply briefly.");
    ASSERT_TRUE(model->generate("Say hello in one short sentence.").has_value());

    const auto before = model->get_history();
    ASSERT_GE(before.size(), 3u);

    std::string streamed;
    const std::array<zoo::MessageView, 2> scoped_messages = {
        zoo::MessageView{zoo::Role::System, "Reply in exactly three words."},
        zoo::MessageView{zoo::Role::User, "Say hello politely."},
    };
    auto on_token = [&](std::string_view token) {
        streamed.append(token);
        return zoo::TokenAction::Continue;
    };
    auto scoped =
        model->complete(zoo::ConversationView{std::span<const zoo::MessageView>(scoped_messages)},
                        {}, on_token);

    ASSERT_TRUE(scoped.has_value()) << scoped.error().to_string();
    EXPECT_FALSE(scoped->text.empty());
    EXPECT_FALSE(streamed.empty());
    EXPECT_EQ(model->get_history(), before);
}

TEST_F(LiveModelIntegrationTest, ExtractReturnsValidJsonMatchingSchema) {
    auto cfg = config();
    cfg.generation.max_tokens = 64;
    auto model_result = zoo::Model::load(cfg.model, cfg.generation);
    ASSERT_TRUE(model_result.has_value()) << model_result.error().to_string();
    auto& model = *model_result;
    model->set_system_prompt("Extract information as instructed.");

    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
        {"required", nlohmann::json::array({"name", "age"})},
        {"additionalProperties", false}};

    auto response = model->extract(schema, "Alice is 30 years old.");

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_TRUE(response->data["name"].is_string());
    EXPECT_TRUE(response->data["age"].is_number_integer());
}

TEST_F(LiveModelIntegrationTest, StreamingCallbackCanCancelGeneration) {
    auto cfg = config();
    cfg.generation.max_tokens = 256;
    auto model_result = zoo::Model::load(cfg.model, cfg.generation);
    ASSERT_TRUE(model_result.has_value()) << model_result.error().to_string();

    int streamed_tokens = 0;
    auto on_token = [&](std::string_view) {
        ++streamed_tokens;
        return zoo::TokenAction::Continue;
    };
    auto should_cancel = [&] { return streamed_tokens > 2; };
    auto response = (*model_result)
                        ->generate("Write a long paragraph about local inference.",
                                   zoo::GenerationOverride::inherit_defaults(),
                                   on_token, should_cancel);

    ASSERT_FALSE(response.has_value());
    EXPECT_EQ(response.error().code, zoo::ErrorCode::RequestCancelled);
    EXPECT_GT(streamed_tokens, 0);
}
