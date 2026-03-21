/**
 * @file test_model_agent.cpp
 * @brief Integration coverage for the concrete Model and Agent layers.
 */

#include <gtest/gtest.h>

#include <algorithm>
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

void skip_unless_live_model_uses_generic_tool_calling(const zoo::Config& config) {
    auto model_result = zoo::core::Model::load(config);
    ASSERT_TRUE(model_result.has_value()) << model_result.error().to_string();

    const std::vector<zoo::CoreToolInfo> tools = {{
        "echo",
        "Echo text",
        R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false})",
    }};

    ASSERT_TRUE((*model_result)->set_tool_calling(tools));
    if (std::string_view((*model_result)->tool_calling_format_name()) != "Generic") {
        GTEST_SKIP() << "Configured integration model does not resolve to generic tool calling.";
    }
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

TEST_F(LiveModelIntegrationTest, AgentCompleteDoesNotMutatePersistentHistory) {
    auto agent_result = zoo::Agent::create(config());
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    agent->set_system_prompt("Reply briefly.");

    auto persistent = agent->chat(zoo::Message::user("Say hello in one short sentence."));
    auto persistent_response = persistent.future.get();
    ASSERT_TRUE(persistent_response.has_value()) << persistent_response.error().to_string();

    const auto before = agent->get_history();
    ASSERT_GE(before.size(), 3u);

    std::string streamed;
    auto scoped = agent->complete({zoo::Message::system("Reply in exactly three words."),
                                   zoo::Message::user("Say hello politely.")},
                                  [&](std::string_view token) { streamed.append(token); });

    auto scoped_response = scoped.future.get();
    ASSERT_TRUE(scoped_response.has_value()) << scoped_response.error().to_string();
    EXPECT_FALSE(scoped_response->text.empty());
    EXPECT_FALSE(streamed.empty());

    const auto after = agent->get_history();
    EXPECT_EQ(after, before);
}

TEST_F(LiveModelIntegrationTest, AgentWithToolsHandlesFencedCodePrompt) {
    auto cfg = config();
    cfg.max_tokens = 96;

    auto agent_result = zoo::Agent::create(cfg);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    ASSERT_TRUE(agent
                    ->register_tool("get_time", "Get the current date and time", {},
                                    []() { return std::string("2026-03-20 12:00:00"); })
                    .has_value());

    agent->set_system_prompt("You are a helpful assistant with access to tools.");

    auto handle = agent->chat(zoo::Message::user(
        "Write a short fenced Python hello-world example and keep the answer brief."));
    auto response = handle.future.get();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());
}

TEST_F(LiveModelIntegrationTest, AgentWithGenericToolsDoesNotExposeWrapperJson) {
    auto cfg = config();
    cfg.max_tokens = 96;

    skip_unless_live_model_uses_generic_tool_calling(cfg);

    auto agent_result = zoo::Agent::create(cfg);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    ASSERT_TRUE(agent
                    ->register_tool("get_time", "Get the current date and time", {},
                                    []() { return std::string("2026-03-20 12:00:00"); })
                    .has_value());

    agent->set_system_prompt("You are a helpful assistant with access to tools.");

    std::string streamed;
    auto handle = agent->chat(
        zoo::Message::user(
            "Write a short fenced Python hello-world example and keep the answer brief."),
        [&](std::string_view token) { streamed.append(token); });
    auto response = handle.future.get();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());
    EXPECT_FALSE(streamed.empty());
    EXPECT_EQ(response->text.find(R"({"response")"), std::string::npos);
    EXPECT_EQ(response->text.find(R"({"tool_call")"), std::string::npos);
    EXPECT_EQ(streamed.find(R"({"response")"), std::string::npos);
    EXPECT_EQ(streamed.find(R"({"tool_call")"), std::string::npos);
}

TEST_F(LiveModelIntegrationTest, AgentWithGenericToolsCompletesToolLoopWithoutToolRoleHistory) {
    auto cfg = config();
    cfg.max_tokens = 96;

    skip_unless_live_model_uses_generic_tool_calling(cfg);

    auto agent_result = zoo::Agent::create(cfg);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();

    auto& agent = *agent_result;
    ASSERT_TRUE(agent
                    ->register_tool("get_time", "Get the current date and time", {},
                                    []() { return std::string("2026-03-20 12:00:00"); })
                    .has_value());

    agent->set_system_prompt("You are a helpful assistant with access to tools.");

    auto handle = agent->chat(
        zoo::Message::user("Use tools to tell me the current date and time. Reply briefly."));
    auto response = handle.future.get();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());
    EXPECT_FALSE(response->tool_invocations.empty());
    EXPECT_EQ(response->text.find(R"({"response")"), std::string::npos);
    EXPECT_EQ(response->text.find(R"({"tool_call")"), std::string::npos);

    const auto history = agent->get_history();
    EXPECT_TRUE(std::none_of(history.begin(), history.end(), [](const zoo::Message& message) {
        return message.role == zoo::Role::Tool;
    }));
}
