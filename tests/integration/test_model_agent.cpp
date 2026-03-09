/**
 * @file test_model_agent.cpp
 * @brief Integration coverage for the concrete Model and Agent layers.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

#include "zoo/agent.hpp"
#include "zoo/core/model.hpp"

namespace {

std::filesystem::path project_source_dir() {
    return std::filesystem::path{ZOO_PROJECT_SOURCE_DIR};
}

std::filesystem::path vendored_fixture_model_path() {
    return project_source_dir() / "extern/llama.cpp/models/ggml-vocab-gpt-2.gguf";
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

zoo::Config make_base_config(const std::filesystem::path& model_path) {
    zoo::Config config;
    config.model_path = model_path.string();
    config.context_size = 2048;
    config.max_tokens = 24;
    config.n_gpu_layers = 0;
    config.max_history_messages = 8;
    config.sampling.temperature = 0.0f;
    config.sampling.top_p = 1.0f;
    config.sampling.top_k = 1;
    config.sampling.seed = 7;
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

    zoo::Config config() const {
        return make_base_config(model_path_);
    }

    std::filesystem::path model_path_;
};

} // namespace

TEST(ModelIntegrationTest, LoadRejectsIncompleteVendoredFixture) {
    const auto model_path = vendored_fixture_model_path();
    ASSERT_TRUE(std::filesystem::exists(model_path))
        << "Expected vendored llama.cpp vocabulary fixture at " << model_path.string();

    auto result = zoo::core::Model::load(make_base_config(model_path));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ModelLoadFailed);
}

TEST(AgentIntegrationTest, CreatePropagatesModelLoadFailures) {
    const auto model_path = vendored_fixture_model_path();
    ASSERT_TRUE(std::filesystem::exists(model_path))
        << "Expected vendored llama.cpp vocabulary fixture at " << model_path.string();

    auto result = zoo::Agent::create(make_base_config(model_path));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ModelLoadFailed);
}

TEST_F(LiveModelIntegrationTest, ModelGeneratesAndTracksHistory) {
    auto model_result = zoo::core::Model::load(config());
    ASSERT_TRUE(model_result.has_value()) << model_result.error().to_string();

    auto& model = *model_result;
    model->set_system_prompt("Reply briefly.");

    auto response = model->generate("Say hello in one short sentence.");
    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());

    const auto history = model->get_history();
    ASSERT_GE(history.size(), 3u);
    EXPECT_EQ(history.front().role, zoo::Role::System);
    EXPECT_EQ(history[1].role, zoo::Role::User);
    EXPECT_EQ(history.back().role, zoo::Role::Assistant);
}

TEST_F(LiveModelIntegrationTest, AgentChatsAndStreams) {
    auto agent_result = zoo::Agent::create(config());
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    agent->set_system_prompt("Reply briefly.");

    std::string streamed;
    auto handle = agent->chat(zoo::Message::user("Say hello in one short sentence."),
                              [&](std::string_view token) { streamed.append(token); });

    auto response = handle.future.get();
    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());
    EXPECT_FALSE(streamed.empty());

    const auto history = agent->get_history();
    ASSERT_GE(history.size(), 3u);
    EXPECT_EQ(history.front().role, zoo::Role::System);
    EXPECT_EQ(history[1].role, zoo::Role::User);
    EXPECT_EQ(history.back().role, zoo::Role::Assistant);
}
