/**
 * @file test_extract.cpp
 * @brief Integration coverage for Agent::extract() with a real GGUF model.
 */

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <optional>
#include <string>

#include "zoo/agent.hpp"

namespace {

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

class LiveExtractIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        auto model_path = live_model_path();
        if (!model_path.has_value()) {
            GTEST_SKIP() << "Set ZOO_INTEGRATION_MODEL to run live extraction smoke tests.";
        }
        if (!std::filesystem::exists(*model_path)) {
            GTEST_SKIP() << "Configured integration model does not exist: " << model_path->string();
        }
        model_path_ = *model_path;
    }

    struct TestConfig {
        zoo::ModelConfig model;
        zoo::AgentConfig agent;
        zoo::GenerationOptions generation;
    };

    TestConfig config(int max_tokens = 64) const {
        TestConfig cfg;
        cfg.model.model_path = model_path_.string();
        cfg.model.context_size = 2048;
        cfg.model.n_gpu_layers = 0;
        cfg.agent.max_history_messages = 8;
        cfg.generation.max_tokens = max_tokens;
        cfg.generation.sampling.temperature = 0.0f;
        cfg.generation.sampling.top_p = 1.0f;
        cfg.generation.sampling.top_k = 1;
        cfg.generation.sampling.seed = 7;
        return cfg;
    }

    std::filesystem::path model_path_;
};

} // namespace

// Verify that the model produces a valid JSON object with the correct key types.
TEST_F(LiveExtractIntegrationTest, ExtractReturnsValidJsonMatchingSchema) {
    const auto cfg = config();
    auto agent_result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();
    auto& agent = *agent_result;
    agent->set_system_prompt("You are a helpful assistant. Extract information as instructed.");

    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
        {"required", nlohmann::json::array({"name", "age"})},
        {"additionalProperties", false}};

    auto handle = agent->extract(schema, "Alice is 30 years old.");
    auto response = handle.await_result();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    const auto& data = response->data;
    EXPECT_TRUE(data.contains("name"));
    EXPECT_TRUE(data.contains("age"));
    EXPECT_TRUE(data["name"].is_string());
    EXPECT_TRUE(data["age"].is_number_integer());
}

// Regression: normal chat() returns a plain text response type, not a structured payload.
TEST_F(LiveExtractIntegrationTest, ChatReturnsPlainTextResponse) {
    const auto cfg = config(24);
    auto agent_result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();
    auto& agent = *agent_result;
    agent->set_system_prompt("Reply briefly.");

    auto handle = agent->chat("Say hello in one short sentence.");
    auto response = handle.await_result();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->text.empty());
    EXPECT_FALSE(response->tool_trace.has_value());
}

// Stateless extract must not append to agent history.
TEST_F(LiveExtractIntegrationTest, StatelessExtractDoesNotMutateHistory) {
    const auto cfg = config();
    auto agent_result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();
    auto& agent = *agent_result;
    agent->set_system_prompt("Reply briefly.");

    auto chat_handle = agent->chat("Say hello in one short sentence.");
    ASSERT_TRUE(chat_handle.await_result().has_value());

    const auto before = agent->get_history();

    nlohmann::json schema = {{"type", "object"},
                             {"properties", {{"sentiment", {{"type", "string"}}}}},
                             {"required", nlohmann::json::array({"sentiment"})},
                             {"additionalProperties", false}};

    const std::array<zoo::MessageView, 2> messages = {
        zoo::MessageView{zoo::Role::System, "Classify the sentiment of the text."},
        zoo::MessageView{zoo::Role::User, "I really enjoyed this film."},
    };
    auto extract_handle =
        agent->extract(schema, zoo::ConversationView{std::span<const zoo::MessageView>(messages)});
    ASSERT_TRUE(extract_handle.await_result().has_value());

    EXPECT_EQ(agent->get_history(), before);
}

// Streaming callback must fire during extraction and extracted_data must still resolve.
TEST_F(LiveExtractIntegrationTest, ExtractStreamsTokensAndReturnsExtractedData) {
    const auto cfg = config();
    auto agent_result = zoo::Agent::create(cfg.model, cfg.agent, cfg.generation);
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();
    auto& agent = *agent_result;
    agent->set_system_prompt("You are a helpful assistant. Extract information as instructed.");

    nlohmann::json schema = {{"type", "object"},
                             {"properties", {{"count", {{"type", "integer"}}}}},
                             {"required", nlohmann::json::array({"count"})},
                             {"additionalProperties", false}};

    std::string streamed;
    auto handle = agent->extract(schema, "There are 7 apples on the shelf.", {},
                                 [&](std::string_view token) { streamed.append(token); });
    auto response = handle.await_result();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(streamed.empty());
    EXPECT_TRUE(response->data.contains("count"));
    EXPECT_TRUE(response->data["count"].is_number_integer());
}
