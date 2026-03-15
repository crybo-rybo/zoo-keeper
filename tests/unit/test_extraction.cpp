/**
 * @file test_extraction.cpp
 * @brief Unit tests for the extraction flow using FakeBackend.
 */

#include "zoo/internal/agent/runtime.hpp"
#include <gtest/gtest.h>

#include <chrono>
#include <deque>
#include <future>
#include <mutex>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;

using zoo::CancellationCallback;
using zoo::Config;
using zoo::Error;
using zoo::ErrorCode;
using zoo::Expected;
using zoo::Message;
using zoo::TokenAction;
using zoo::TokenCallback;
using zoo::internal::agent::AgentBackend;
using zoo::internal::agent::AgentRuntime;
using zoo::internal::agent::GenerationResult;

class FakeBackend final : public AgentBackend {
  public:
    using GenerationAction = std::function<Expected<GenerationResult>(std::optional<TokenCallback>,
                                                                      const CancellationCallback&)>;

    void push_generation(GenerationAction action) {
        std::lock_guard<std::mutex> lock(mutex_);
        generations_.push_back(std::move(action));
    }

    void set_tool_grammar_supported(bool supported) {
        std::lock_guard<std::mutex> lock(mutex_);
        tool_grammar_supported_ = supported;
    }

    std::vector<std::string> operations() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return operations_;
    }

    std::string last_tool_grammar() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_tool_grammar_;
    }

    std::string last_schema_grammar_set() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_schema_grammar_set_;
    }

    Expected<void> add_message(const Message& message) override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.push_back(message);
        operations_.push_back("add:" + std::string(zoo::role_to_string(message.role)) + ":" +
                              message.content);
        return {};
    }

    Expected<GenerationResult> generate_from_history(std::optional<TokenCallback> on_token,
                                                     CancellationCallback should_cancel) override {
        GenerationAction action;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            operations_.push_back("generate");
            if (generations_.empty()) {
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "No scripted generation available"});
            }
            action = std::move(generations_.front());
            generations_.pop_front();
        }
        return action(std::move(on_token), should_cancel);
    }

    void finalize_response() override {
        std::lock_guard<std::mutex> lock(mutex_);
        operations_.push_back("finalize");
    }

    void set_system_prompt(const std::string& prompt) override {
        std::lock_guard<std::mutex> lock(mutex_);
        Message system_message = Message::system(prompt);
        if (!history_.empty() && history_.front().role == zoo::Role::System) {
            history_.front() = std::move(system_message);
        } else {
            history_.insert(history_.begin(), std::move(system_message));
        }
        operations_.push_back("set_system_prompt:" + prompt);
    }

    std::vector<Message> get_history() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        operations_.push_back("get_history");
        return history_;
    }

    void clear_history() override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.clear();
        operations_.push_back("clear_history");
    }

    bool set_tool_grammar(const std::string& grammar_str) override {
        std::lock_guard<std::mutex> lock(mutex_);
        last_tool_grammar_ = grammar_str;
        operations_.push_back("set_tool_grammar");
        return tool_grammar_supported_;
    }

    bool set_schema_grammar(const std::string& grammar_str) override {
        std::lock_guard<std::mutex> lock(mutex_);
        last_schema_grammar_set_ = grammar_str;
        operations_.push_back("set_schema_grammar");
        return true;
    }

    void clear_tool_grammar() override {
        std::lock_guard<std::mutex> lock(mutex_);
        last_tool_grammar_.clear();
        operations_.push_back("clear_tool_grammar");
    }

    void replace_messages(std::vector<Message> messages) override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_ = std::move(messages);
        operations_.push_back("replace_messages");
    }

  private:
    mutable std::mutex mutex_;
    mutable std::vector<std::string> operations_;
    std::deque<GenerationAction> generations_;
    std::vector<Message> history_;
    std::string last_tool_grammar_;
    std::string last_schema_grammar_set_;
    bool tool_grammar_supported_ = true;
};

Config make_config() {
    Config config;
    config.request_queue_capacity = 4;
    config.max_tool_iterations = 5;
    config.max_tool_retries = 2;
    return config;
}

nlohmann::json simple_schema() {
    return {{"type", "object"},
            {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
            {"required", nlohmann::json::array({"name", "age"})},
            {"additionalProperties", false}};
}

TEST(ExtractionTest, ValidJsonReturnsExtractedData) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"name": "Alice", "age": 30})", 10, false});
    });

    auto handle = runtime.extract(simple_schema(), Message::user("extract person"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    ASSERT_TRUE(result->extracted_data.has_value());
    EXPECT_EQ((*result->extracted_data)["name"], "Alice");
    EXPECT_EQ((*result->extracted_data)["age"], 30);
    EXPECT_EQ(result->text, R"({"name": "Alice", "age": 30})");
}

TEST(ExtractionTest, InvalidSchemaRejectedUpfront) {
    auto backend = std::make_unique<FakeBackend>();
    AgentRuntime runtime(make_config(), std::move(backend));

    nlohmann::json bad_schema = {{"type", "array"}}; // Not object type

    auto handle = runtime.extract(bad_schema, Message::user("extract"));
    auto result = handle.future.get();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidOutputSchema);
}

TEST(ExtractionTest, SchemaGrammarIsSetDuringExtraction) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"name": "Bob", "age": 25})", 10, false});
    });

    auto handle = runtime.extract(simple_schema(), Message::user("extract person"));
    auto result = handle.future.get();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    const auto ops = backend_ptr->operations();
    bool found_schema_grammar = false;
    for (const auto& op : ops) {
        if (op == "set_schema_grammar") {
            found_schema_grammar = true;
            break;
        }
    }
    EXPECT_TRUE(found_schema_grammar);

    // Schema grammar string should contain schema-0 prefix
    EXPECT_NE(backend_ptr->last_schema_grammar_set().find("schema-0"), std::string::npos);
}

TEST(ExtractionTest, ToolGrammarRestoredAfterExtraction) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    // Register a tool to set up tool grammar
    auto definition = zoo::tools::detail::make_tool_definition(
        "greet", "Greet someone", std::vector<std::string>{"name"},
        [](std::string name) { return "Hi " + name; });
    ASSERT_TRUE(definition.has_value());
    ASSERT_TRUE(runtime.register_tool(std::move(*definition)).has_value());

    // Tool grammar should be active
    EXPECT_FALSE(backend_ptr->last_tool_grammar().empty());

    // Now do an extraction
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"name": "Alice", "age": 30})", 10, false});
    });

    auto handle = runtime.extract(simple_schema(), Message::user("extract"));
    auto result = handle.future.get();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    // After extraction, tool grammar should be restored
    const auto ops = backend_ptr->operations();
    // Should see: set_tool_grammar (initial), set_schema_grammar (extraction),
    // clear_tool_grammar (guard), set_tool_grammar (restore)
    int set_tool_grammar_count = 0;
    for (const auto& op : ops) {
        if (op == "set_tool_grammar") {
            ++set_tool_grammar_count;
        }
    }
    EXPECT_GE(set_tool_grammar_count, 2); // At least initial + restore
}

TEST(ExtractionTest, StreamingCallbackReceivesTokens) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    backend_ptr->push_generation(
        [](std::optional<TokenCallback> on_token, const CancellationCallback&) {
            if (on_token) {
                (*on_token)("{");
                (*on_token)(R"("name": "Alice")");
                (*on_token)("}");
            }
            return Expected<GenerationResult>(GenerationResult{R"({"name": "Alice"})", 10, false});
        });

    nlohmann::json schema = {{"type", "object"},
                             {"properties", {{"name", {{"type", "string"}}}}},
                             {"required", nlohmann::json::array({"name"})},
                             {"additionalProperties", false}};

    std::string streamed;
    auto handle = runtime.extract(schema, Message::user("extract"),
                                  [&streamed](std::string_view token) { streamed += token; });
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_FALSE(streamed.empty());
}

TEST(ExtractionTest, CancellationDuringExtraction) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();

    backend_ptr->push_generation(
        [entered](std::optional<TokenCallback>, const CancellationCallback& should_cancel) {
            entered->set_value();
            while (!should_cancel()) {
                std::this_thread::sleep_for(1ms);
            }
            return Expected<GenerationResult>(
                std::unexpected(Error{ErrorCode::RequestCancelled, "cancelled"}));
        });

    auto handle = runtime.extract(simple_schema(), Message::user("extract"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    runtime.cancel(handle.id);

    auto result = handle.future.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::RequestCancelled);
}

TEST(ExtractionTest, StatelessExtractionUsesProvidedMessages) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    // First, add persistent state
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"persistent reply", 0, false});
    });
    auto first = runtime.chat(Message::user("persistent user"));
    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());

    // Now do stateless extraction
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"name": "Bob", "age": 42})", 10, false});
    });

    auto handle =
        runtime.extract(simple_schema(), std::vector<Message>{Message::system("extract entities"),
                                                              Message::user("Bob is 42")});
    auto result = handle.future.get();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    ASSERT_TRUE(result->extracted_data.has_value());
    EXPECT_EQ((*result->extracted_data)["name"], "Bob");

    // Persistent history should be restored
    auto history = runtime.get_history();
    ASSERT_GE(history.size(), 2u);
    EXPECT_EQ(history[0].content, "persistent user");
    EXPECT_EQ(history[1].content, "persistent reply");
}

TEST(ExtractionTest, NormalChatHasNulloptExtractedData) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"normal reply", 0, false});
    });

    auto handle = runtime.chat(Message::user("hello"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->extracted_data.has_value());
}

TEST(ExtractionTest, ExtractionRequestRoutesWithoutToolLoop) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    // Register a tool to ensure we have a tool loop normally
    auto definition = zoo::tools::detail::make_tool_definition("double_value", "Doubles a number",
                                                               std::vector<std::string>{"value"},
                                                               [](int value) { return value * 2; });
    ASSERT_TRUE(definition.has_value());
    ASSERT_TRUE(runtime.register_tool(std::move(*definition)).has_value());

    // Push exactly one generation -- extraction should use single pass
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"name": "test", "age": 1})", 10, false});
    });

    auto handle = runtime.extract(simple_schema(), Message::user("extract"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    ASSERT_TRUE(result->extracted_data.has_value());
    // Only one generation call (no tool loop iterations)
    int gen_count = 0;
    for (const auto& op : backend_ptr->operations()) {
        if (op == "generate") {
            ++gen_count;
        }
    }
    // Exactly one generate from extraction (plus one from tool grammar setup)
    // The extraction itself should only call generate once
    EXPECT_EQ(gen_count, 1);
}

TEST(ExtractionTest, ValidJsonButWrongTypesReturnsError) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"name": 42, "age": "not_a_number"})", 10, false});
    });

    auto handle = runtime.extract(simple_schema(), Message::user("extract"));
    auto result = handle.future.get();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ExtractionFailed);
}

TEST(ExtractionTest, MalformedJsonOutputReturnsError) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"not valid json at all", 10, false});
    });

    auto handle = runtime.extract(simple_schema(), Message::user("extract"));
    auto result = handle.future.get();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ExtractionFailed);
}

} // namespace
