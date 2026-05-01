/**
 * @file test_model_agent.cpp
 * @brief Integration coverage for the concrete Model and Agent layers.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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

struct TestConfig {
    zoo::ModelConfig model;
    zoo::AgentConfig agent;
    zoo::GenerationOptions generation;
};

TestConfig make_base_config(const std::filesystem::path& model_path) {
    TestConfig config;
    config.model.model_path = model_path.string();
    config.model.context_size = 2048;
    config.model.n_gpu_layers = 0;
    config.agent.max_history_messages = 8;
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
    const auto model_path = vendored_fixture_model_path();
    ASSERT_TRUE(std::filesystem::exists(model_path))
        << "Expected vendored llama.cpp vocabulary fixture at " << model_path.string();

    auto cfg = make_base_config(model_path);
    auto result = zoo::core::Model::load(cfg.model, cfg.generation);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ModelLoadFailed);
}

TEST(AgentIntegrationTest, CreatePropagatesModelLoadFailures) {
    const auto model_path = vendored_fixture_model_path();
    ASSERT_TRUE(std::filesystem::exists(model_path))
        << "Expected vendored llama.cpp vocabulary fixture at " << model_path.string();

    auto cfg = make_base_config(model_path);
    auto result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ModelLoadFailed);
}

TEST_F(LiveModelIntegrationTest, ModelGeneratesAndTracksHistory) {
    const auto cfg = config();
    auto model_result = zoo::core::Model::load(cfg.model, cfg.generation);
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

TEST_F(LiveModelIntegrationTest, AgentChatsAndStreams) {
    const auto cfg = config();
    auto agent_result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    agent->set_system_prompt("Reply briefly.");

    std::string streamed;
    auto handle = agent->chat("Say hello in one short sentence.", {},
                              [&](std::string_view token) { streamed.append(token); });

    auto response = handle.await_result();
    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());
    EXPECT_FALSE(streamed.empty());

    const auto history = agent->get_history();
    ASSERT_GE(history.size(), 3u);
    EXPECT_EQ(history[0].role, zoo::Role::System);
    EXPECT_EQ(history[1].role, zoo::Role::User);
    EXPECT_EQ(history[history.size() - 1].role, zoo::Role::Assistant);
}

TEST_F(LiveModelIntegrationTest, AgentCompleteDoesNotMutatePersistentHistory) {
    const auto cfg = config();
    auto agent_result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    agent->set_system_prompt("Reply briefly.");

    auto persistent = agent->chat("Say hello in one short sentence.");
    auto persistent_response = persistent.await_result();
    ASSERT_TRUE(persistent_response.has_value()) << persistent_response.error().to_string();

    const auto before = agent->get_history();
    ASSERT_GE(before.size(), 3u);

    std::string streamed;
    const std::array<zoo::MessageView, 2> scoped_messages = {
        zoo::MessageView{zoo::Role::System, "Reply in exactly three words."},
        zoo::MessageView{zoo::Role::User, "Say hello politely."},
    };
    auto scoped =
        agent->complete(zoo::ConversationView{std::span<const zoo::MessageView>(scoped_messages)},
                        {}, [&](std::string_view token) { streamed.append(token); });

    auto scoped_response = scoped.await_result();
    ASSERT_TRUE(scoped_response.has_value()) << scoped_response.error().to_string();
    EXPECT_FALSE(scoped_response->text.empty());
    EXPECT_FALSE(streamed.empty());

    const auto after = agent->get_history();
    EXPECT_EQ(after, before);
}

TEST_F(LiveModelIntegrationTest, AgentWithToolsHandlesFencedCodePrompt) {
    auto cfg = config();
    cfg.generation.max_tokens = 96;

    auto agent_result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    ASSERT_TRUE(agent
                    ->register_tool("get_time", "Get the current date and time", {},
                                    []() { return std::string("2026-03-20 12:00:00"); })
                    .has_value());

    agent->set_system_prompt("You are a helpful assistant with access to tools.");

    auto handle =
        agent->chat("Write a short fenced Python hello-world example and keep the answer brief.");
    auto response = handle.await_result();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());
}

TEST_F(LiveModelIntegrationTest, AgentWithToolsInvokesToolForTimeQuery) {
    auto cfg = config();
    cfg.generation.max_tokens = 128;
    cfg.generation.record_tool_trace = true;

    auto agent_result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    ASSERT_TRUE(agent
                    ->register_tool("get_time", "Get the current date and time", {},
                                    []() { return std::string("2026-03-20 12:00:00"); })
                    .has_value());

    agent->set_system_prompt(
        "You are a helpful assistant. When the user asks for the current time, you MUST call the "
        "get_time tool. Do not guess.");

    auto handle = agent->chat("What is the current date and time right now?");
    auto response = handle.await_result();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    ASSERT_TRUE(response->tool_trace.has_value())
        << "Tool trace was not recorded; expected at least one invocation of get_time";

    const auto& invocations = response->tool_trace->invocations;
    ASSERT_FALSE(invocations.empty())
        << "Tool trace was empty; the model did not emit a parseable tool call";

    const bool called_get_time =
        std::any_of(invocations.begin(), invocations.end(),
                    [](const zoo::ToolInvocation& inv) { return inv.name == "get_time"; });
    EXPECT_TRUE(called_get_time) << "Expected at least one invocation named 'get_time'; got "
                                 << invocations.size() << " invocation(s) with first name='"
                                 << invocations.front().name << "'";
}
