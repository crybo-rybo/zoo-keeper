/**
 * @file test_extract.cpp
 * @brief Integration coverage for Agent::extract() with a real GGUF model.
 */

#include <gtest/gtest.h>

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

    zoo::Config config(int max_tokens = 64) const {
        zoo::Config cfg;
        cfg.model_path = model_path_.string();
        cfg.context_size = 2048;
        cfg.max_tokens = max_tokens;
        cfg.n_gpu_layers = 0;
        cfg.max_history_messages = 8;
        cfg.sampling.temperature = 0.0f;
        cfg.sampling.top_p = 1.0f;
        cfg.sampling.top_k = 1;
        cfg.sampling.seed = 7;
        return cfg;
    }

    std::filesystem::path model_path_;
};

} // namespace

// Verify that the model produces a valid JSON object with the correct key types.
TEST_F(LiveExtractIntegrationTest, ExtractReturnsValidJsonMatchingSchema) {
    auto agent_result = zoo::Agent::create(config());
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();
    auto& agent = *agent_result;
    agent->set_system_prompt("You are a helpful assistant. Extract information as instructed.");

    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
        {"required", nlohmann::json::array({"name", "age"})},
        {"additionalProperties", false}};

    auto handle = agent->extract(schema, zoo::Message::user("Alice is 30 years old."));
    auto response = handle.future.get();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    ASSERT_TRUE(response->extracted_data.has_value());

    const auto& data = *response->extracted_data;
    EXPECT_TRUE(data.contains("name"));
    EXPECT_TRUE(data.contains("age"));
    EXPECT_TRUE(data["name"].is_string());
    EXPECT_TRUE(data["age"].is_number_integer());
}

// Regression: normal chat() must leave extracted_data as nullopt.
TEST_F(LiveExtractIntegrationTest, ExtractedDataIsNulloptForNormalChat) {
    auto agent_result = zoo::Agent::create(config(24));
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();
    auto& agent = *agent_result;
    agent->set_system_prompt("Reply briefly.");

    auto handle = agent->chat(zoo::Message::user("Say hello in one short sentence."));
    auto response = handle.future.get();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(response->extracted_data.has_value());
}

// Stateless extract must not append to agent history.
TEST_F(LiveExtractIntegrationTest, StatelessExtractDoesNotMutateHistory) {
    auto agent_result = zoo::Agent::create(config());
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();
    auto& agent = *agent_result;
    agent->set_system_prompt("Reply briefly.");

    auto chat_handle = agent->chat(zoo::Message::user("Say hello in one short sentence."));
    ASSERT_TRUE(chat_handle.future.get().has_value());

    const auto before = agent->get_history();

    nlohmann::json schema = {{"type", "object"},
                             {"properties", {{"sentiment", {{"type", "string"}}}}},
                             {"required", nlohmann::json::array({"sentiment"})},
                             {"additionalProperties", false}};

    auto extract_handle =
        agent->extract(schema, {zoo::Message::system("Classify the sentiment of the text."),
                                zoo::Message::user("I really enjoyed this film.")});
    ASSERT_TRUE(extract_handle.future.get().has_value());

    EXPECT_EQ(agent->get_history(), before);
}

// Streaming callback must fire during extraction and extracted_data must still resolve.
TEST_F(LiveExtractIntegrationTest, ExtractStreamsTokensAndReturnsExtractedData) {
    auto agent_result = zoo::Agent::create(config());
    ASSERT_TRUE(agent_result.has_value()) << agent_result.error().to_string();
    auto& agent = *agent_result;
    agent->set_system_prompt("You are a helpful assistant. Extract information as instructed.");

    nlohmann::json schema = {{"type", "object"},
                             {"properties", {{"count", {{"type", "integer"}}}}},
                             {"required", nlohmann::json::array({"count"})},
                             {"additionalProperties", false}};

    std::string streamed;
    auto handle = agent->extract(schema, zoo::Message::user("There are 7 apples on the shelf."),
                                 [&](std::string_view token) { streamed.append(token); });
    auto response = handle.future.get();

    ASSERT_TRUE(response.has_value()) << response.error().to_string();
    EXPECT_FALSE(streamed.empty());
    ASSERT_TRUE(response->extracted_data.has_value());
    EXPECT_TRUE((*response->extracted_data).contains("count"));
    EXPECT_TRUE((*response->extracted_data)["count"].is_number_integer());
}
