#include <gtest/gtest.h>
#include "zoo/core/model.hpp"
#include "mocks/mock_backend.hpp"

class ModelTest : public ::testing::Test {
protected:
    std::unique_ptr<zoo::testing::MockBackend> make_mock() {
        return std::make_unique<zoo::testing::MockBackend>();
    }

    std::unique_ptr<zoo::core::Model> make_model(
        std::unique_ptr<zoo::testing::MockBackend> mock = nullptr
    ) {
        zoo::Config config;
        config.model_path = "/path/to/model.gguf";
        config.max_tokens = 512;

        if (!mock) mock = make_mock();
        auto result = zoo::core::Model::load(config, std::move(mock));
        EXPECT_TRUE(result.has_value());
        return std::move(*result);
    }
};

TEST_F(ModelTest, LoadSuccess) {
    auto model = make_model();
    EXPECT_NE(model, nullptr);
}

TEST_F(ModelTest, LoadFailsOnInvalidConfig) {
    zoo::Config config; // empty model_path
    auto result = zoo::core::Model::load(config);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelPath);
}

TEST_F(ModelTest, LoadFailsOnBackendInit) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    auto mock = make_mock();
    mock->should_fail_initialize = true;
    auto result = zoo::core::Model::load(config, std::move(mock));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::BackendInitFailed);
}

TEST_F(ModelTest, GenerateBasic) {
    auto mock = make_mock();
    mock->default_response = "Hello back!";
    auto model = make_model(std::move(mock));

    auto result = model->generate("Hello");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "Hello back!");
}

TEST_F(ModelTest, GenerateAddsToHistory) {
    auto model = make_model();
    auto result = model->generate("Hello");
    ASSERT_TRUE(result.has_value());

    auto history = model->get_history();
    ASSERT_GE(history.size(), 2u);
    EXPECT_EQ(history[0].role, zoo::Role::User);
    EXPECT_EQ(history[0].content, "Hello");
    EXPECT_EQ(history[1].role, zoo::Role::Assistant);
}

TEST_F(ModelTest, SystemPrompt) {
    auto model = make_model();
    model->set_system_prompt("You are helpful.");

    auto history = model->get_history();
    ASSERT_GE(history.size(), 1u);
    EXPECT_EQ(history[0].role, zoo::Role::System);
    EXPECT_EQ(history[0].content, "You are helpful.");
}

TEST_F(ModelTest, SystemPromptFromConfig) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    config.system_prompt = "Be concise.";

    auto result = zoo::core::Model::load(config, make_mock());
    ASSERT_TRUE(result.has_value());
    auto model = std::move(*result);

    auto history = model->get_history();
    ASSERT_GE(history.size(), 1u);
    EXPECT_EQ(history[0].role, zoo::Role::System);
    EXPECT_EQ(history[0].content, "Be concise.");
}

TEST_F(ModelTest, ClearHistory) {
    auto model = make_model();
    model->generate("Hello");
    EXPECT_FALSE(model->get_history().empty());
    model->clear_history();
    EXPECT_TRUE(model->get_history().empty());
    EXPECT_EQ(model->estimated_tokens(), 0);
}

TEST_F(ModelTest, AddMessageValidation) {
    auto model = make_model();
    // Tool as first message should fail
    auto result = model->add_message(zoo::Message::tool("result", "id"));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidMessageSequence);
}

TEST_F(ModelTest, ConsecutiveSameRoleFails) {
    auto model = make_model();
    model->add_message(zoo::Message::user("Hello"));
    auto result = model->add_message(zoo::Message::user("Again"));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidMessageSequence);
}

TEST_F(ModelTest, GenerateFailsRollsBackHistory) {
    auto mock = make_mock();
    mock->should_fail_generate = true;
    auto model = make_model(std::move(mock));

    auto result = model->generate("Hello");
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(model->get_history().empty());
}

TEST_F(ModelTest, ContextSize) {
    auto model = make_model();
    EXPECT_EQ(model->context_size(), 8192);
}

TEST_F(ModelTest, StreamingCallback) {
    auto mock = make_mock();
    mock->default_response = "word1 word2";
    auto model = make_model(std::move(mock));

    std::vector<std::string> tokens;
    auto result = model->generate("Hello", [&tokens](std::string_view t) {
        tokens.push_back(std::string(t));
    });
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(tokens.empty());
}
