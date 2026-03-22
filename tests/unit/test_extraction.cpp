/**
 * @file test_extraction.cpp
 * @brief Unit tests for runtime extraction using a fake backend.
 */

#include "zoo/internal/agent/runtime.hpp"
#include <gtest/gtest.h>

#include <deque>
#include <mutex>
#include <string>
#include <utility>

namespace {

using zoo::AgentConfig;
using zoo::CancellationCallback;
using zoo::Error;
using zoo::ErrorCode;
using zoo::Expected;
using zoo::ExtractionResponse;
using zoo::GenerationOptions;
using zoo::HistorySnapshot;
using zoo::Message;
using zoo::MessageView;
using zoo::ModelConfig;
using zoo::Role;
using zoo::TokenAction;
using zoo::TokenCallback;
using zoo::internal::agent::AgentBackend;
using zoo::internal::agent::AgentRuntime;
using zoo::internal::agent::GenerationResult;
using zoo::internal::agent::ParsedToolResponse;

class FakeBackend final : public AgentBackend {
  public:
    using GenerationAction =
        std::function<Expected<GenerationResult>(TokenCallback, const CancellationCallback&)>;

    void push_generation(GenerationAction action) {
        std::lock_guard<std::mutex> lock(mutex_);
        generations_.push_back(std::move(action));
    }

    Expected<void> add_message(MessageView message) override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.push_back(Message::from_view(message));
        return {};
    }

    Expected<GenerationResult> generate_from_history(const GenerationOptions&,
                                                     TokenCallback on_token,
                                                     CancellationCallback should_cancel) override {
        GenerationAction action;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (generations_.empty()) {
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "No scripted generation available"});
            }
            action = std::move(generations_.front());
            generations_.pop_front();
        }
        return action(on_token, should_cancel);
    }

    void finalize_response() override {}

    void set_system_prompt(std::string_view prompt) override {
        std::lock_guard<std::mutex> lock(mutex_);
        Message system_message = Message::system(std::string(prompt));
        if (!history_.empty() && history_.front().role == Role::System) {
            history_.front() = std::move(system_message);
        } else {
            history_.insert(history_.begin(), std::move(system_message));
        }
    }

    HistorySnapshot get_history() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return HistorySnapshot{history_};
    }

    void clear_history() override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.clear();
    }

    void replace_history(HistorySnapshot snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_ = std::move(snapshot.messages);
    }

    HistorySnapshot swap_history(HistorySnapshot snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        HistorySnapshot previous{std::move(history_)};
        history_ = std::move(snapshot.messages);
        return previous;
    }

    bool set_tool_calling(const std::vector<zoo::CoreToolInfo>&) override {
        return true;
    }

    bool set_schema_grammar(const std::string&) override {
        return true;
    }

    void clear_tool_grammar() override {}

    ParsedToolResponse parse_tool_response(std::string_view text) const override {
        return ParsedToolResponse{std::string(text), {}};
    }

    const char* tool_calling_format_name() const noexcept override {
        return "fake";
    }

  private:
    mutable std::mutex mutex_;
    std::deque<GenerationAction> generations_;
    std::vector<Message> history_;
};

ModelConfig make_model_config() {
    ModelConfig config;
    config.model_path = "unused.gguf";
    return config;
}

AgentConfig make_agent_config() {
    AgentConfig config;
    config.request_queue_capacity = 4;
    return config;
}

nlohmann::json simple_schema() {
    return {{"type", "object"},
            {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
            {"required", nlohmann::json::array({"name", "age"})},
            {"additionalProperties", false}};
}

TEST(ExtractionRuntimeTest, ExtractReturnsParsedJson) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"name":"Alice","age":30})", 0, false, "", {}});
    });

    auto handle = runtime.extract(simple_schema(), "Alice is 30");
    auto result = handle.await_result();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->data["name"], "Alice");
    EXPECT_EQ(result->data["age"], 30);
}

TEST(ExtractionRuntimeTest, InvalidSchemaFailsImmediately) {
    auto backend = std::make_unique<FakeBackend>();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    nlohmann::json bad_schema = {{"type", "array"}};
    auto handle = runtime.extract(bad_schema, "extract");
    auto result = handle.await_result();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidOutputSchema);
}

TEST(ExtractionRuntimeTest, StatelessExtractDoesNotMutateHistory) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"hello", 0, false, "", {}});
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"name":"Bob","age":42})", 0, false, "", {}});
    });

    ASSERT_TRUE(runtime.chat("hello").await_result().has_value());
    const auto before = runtime.get_history();

    const std::array<Message, 2> scoped_messages = {Message::system("Extract entities."),
                                                    Message::user("Bob is 42")};
    auto handle = runtime.extract(simple_schema(),
                                  zoo::ConversationView{std::span<const Message>(scoped_messages)});
    auto result = handle.await_result();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->data["name"], "Bob");
    EXPECT_EQ(runtime.get_history(), before);
}

TEST(ExtractionRuntimeTest, ExtractStreamsTokens) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    backend_ptr->push_generation([](TokenCallback on_token, const CancellationCallback&) {
        if (on_token) {
            EXPECT_EQ(on_token(R"({"name":)"), TokenAction::Continue);
            EXPECT_EQ(on_token(R"("Alice","age":30})"), TokenAction::Continue);
        }
        return Expected<GenerationResult>(
            GenerationResult{R"({"name":"Alice","age":30})", 0, false, "", {}});
    });

    std::string streamed;
    auto handle = runtime.extract(simple_schema(), "Alice is 30", GenerationOptions{},
                                  [&](std::string_view token) { streamed.append(token); });
    auto result = handle.await_result();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_FALSE(streamed.empty());
    EXPECT_EQ(result->data["name"], "Alice");
}

} // namespace
