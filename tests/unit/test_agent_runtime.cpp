/**
 * @file test_agent_runtime.cpp
 * @brief Unit tests for the internal agent runtime using a fake backend.
 */

#include "zoo/internal/agent/runtime.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

namespace {

using namespace std::chrono_literals;

using zoo::CancellationCallback;
using zoo::Config;
using zoo::Error;
using zoo::ErrorCode;
using zoo::Expected;
using zoo::Message;
using zoo::TokenCallback;
using zoo::internal::agent::AgentBackend;
using zoo::internal::agent::AgentRuntime;
using zoo::internal::agent::GenerationResult;
using zoo::internal::agent::ParsedToolResponse;

class FakeBackend final : public AgentBackend {
  public:
    using GenerationAction = std::function<Expected<GenerationResult>(std::optional<TokenCallback>,
                                                                      const CancellationCallback&)>;

    void push_generation(GenerationAction action) {
        std::lock_guard<std::mutex> lock(mutex_);
        generations_.push_back(std::move(action));
    }

    std::vector<std::string> operations() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return operations_;
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

    bool set_tool_calling(const std::vector<zoo::CoreToolInfo>& tools) override {
        std::lock_guard<std::mutex> lock(mutex_);
        tool_calling_supported_ = !tools.empty();
        operations_.push_back("set_tool_calling");
        return tool_calling_supported_;
    }

    ParsedToolResponse parse_tool_response(const std::string& text) const override {
        ParsedToolResponse result;

        if (generic_tool_format_) {
            // Simulate common_chat_parse_generic(): the model wraps all output
            // in JSON — either {"response": "..."} or {"tool_call": {...}}.
            auto j = nlohmann::json::parse(text, nullptr, false);
            if (!j.is_discarded() && j.is_object()) {
                if (j.contains("tool_call")) {
                    const auto& tc_json = j["tool_call"];
                    zoo::ToolCallInfo tc;
                    tc.id = tc_json.value("id", "");
                    tc.name = tc_json.value("name", "");
                    if (tc_json.contains("arguments")) {
                        tc.arguments_json = tc_json["arguments"].dump();
                    }
                    result.tool_calls.push_back(std::move(tc));
                } else if (j.contains("response")) {
                    result.content = j["response"].get<std::string>();
                }
            } else {
                result.content = text;
            }
            return result;
        }

        const std::string open_tag = "<tool_call>";
        const std::string close_tag = "</tool_call>";
        auto start = text.find(open_tag);
        auto end = text.find(close_tag);
        if (start != std::string::npos && end != std::string::npos) {
            auto json_str = text.substr(start + open_tag.size(), end - start - open_tag.size());
            auto j = nlohmann::json::parse(json_str, nullptr, false);
            if (!j.is_discarded()) {
                zoo::ToolCallInfo tc;
                tc.id = j.value("id", "");
                tc.name = j.value("name", "");
                if (j.contains("arguments")) {
                    tc.arguments_json = j["arguments"].dump();
                }
                result.tool_calls.push_back(std::move(tc));
            }
            // Content is everything outside the tool call tags
            result.content = text.substr(0, start);
            if (end + close_tag.size() < text.size()) {
                result.content += text.substr(end + close_tag.size());
            }
        } else {
            result.content = text;
        }
        return result;
    }

    const char* tool_calling_format_name() const noexcept override {
        return generic_tool_format_ ? "Generic" : "fake";
    }

    bool is_generic_tool_format() const noexcept override {
        return generic_tool_format_;
    }

    void set_generic_tool_format(bool value) {
        generic_tool_format_ = value;
    }

    bool set_schema_grammar(const std::string& grammar_str) override {
        std::lock_guard<std::mutex> lock(mutex_);
        operations_.push_back("set_schema_grammar");
        (void)grammar_str;
        return true;
    }

    void clear_tool_grammar() override {
        std::lock_guard<std::mutex> lock(mutex_);
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
    bool tool_calling_supported_ = true;
    bool generic_tool_format_ = false;
};

Config make_config() {
    Config config;
    config.request_queue_capacity = 4;
    config.max_tool_iterations = 5;
    config.max_tool_retries = 2;
    return config;
}

template <typename Func>
void register_single_int_tool(AgentRuntime& runtime, const std::string& name,
                              const std::string& description, Func func) {
    auto definition = zoo::tools::detail::make_tool_definition(
        name, description, std::vector<std::string>{"value"}, std::move(func));
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();
    ASSERT_TRUE(runtime.register_tool(std::move(*definition)).has_value());
}

GenerationResult tool_call_generation(const std::string& tool_name, const nlohmann::json& arguments,
                                      std::string id = "call-1") {
    nlohmann::json payload = {{"id", std::move(id)}, {"name", tool_name}, {"arguments", arguments}};
    return GenerationResult{"<tool_call>" + payload.dump() + "</tool_call>", 0, true};
}

size_t index_of(const std::vector<std::string>& operations, const std::string& value) {
    auto it = std::find(operations.begin(), operations.end(), value);
    EXPECT_NE(it, operations.end());
    return static_cast<size_t>(std::distance(operations.begin(), it));
}

TEST(AgentRuntimeTest, QueueFullFailsAdditionalQueuedRequest) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.request_queue_capacity = 1;
    AgentRuntime runtime(config, std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](std::optional<TokenCallback>, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"first reply", 0, false});
        });
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"second reply", 0, false});
    });

    auto first = runtime.chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto second = runtime.chat(Message::user("second"));
    auto third = runtime.chat(Message::user("third"));

    auto third_result = third.future.get();
    ASSERT_FALSE(third_result.has_value());
    EXPECT_EQ(third_result.error().code, ErrorCode::QueueFull);

    release->set_value();

    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());
    EXPECT_EQ(first_result->text, "first reply");

    auto second_result = second.future.get();
    ASSERT_TRUE(second_result.has_value());
    EXPECT_EQ(second_result->text, "second reply");
}

TEST(AgentRuntimeTest, CancelBeforeProcessingBeginsFailsQueuedRequest) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](std::optional<TokenCallback>, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"first reply", 0, false});
        });

    auto first = runtime.chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto second = runtime.chat(Message::user("second"));
    runtime.cancel(second.id);

    release->set_value();

    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());

    auto second_result = second.future.get();
    ASSERT_FALSE(second_result.has_value());
    EXPECT_EQ(second_result.error().code, ErrorCode::RequestCancelled);
}

TEST(AgentRuntimeTest, CancelDuringGenerationPropagatesCancellationError) {
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
                std::unexpected(Error{ErrorCode::RequestCancelled, "cancelled during generation"}));
        });

    auto handle = runtime.chat(Message::user("cancel me"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    runtime.cancel(handle.id);

    auto result = handle.future.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::RequestCancelled);
}

TEST(AgentRuntimeTest, StopDrainsQueuedRequestsWithAgentNotRunning) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.request_queue_capacity = 3;
    AgentRuntime runtime(config, std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](std::optional<TokenCallback>, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"first reply", 0, false});
        });

    auto first = runtime.chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto second = runtime.chat(Message::user("second"));
    auto third = runtime.chat(Message::user("third"));

    auto stop_future = std::async(std::launch::async, [&runtime] { runtime.stop(); });
    EXPECT_EQ(stop_future.wait_for(50ms), std::future_status::timeout);

    release->set_value();
    stop_future.get();

    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());

    auto second_result = second.future.get();
    ASSERT_FALSE(second_result.has_value());
    EXPECT_EQ(second_result.error().code, ErrorCode::AgentNotRunning);

    auto third_result = third.future.get();
    ASSERT_FALSE(third_result.has_value());
    EXPECT_EQ(third_result.error().code, ErrorCode::AgentNotRunning);
}

TEST(AgentRuntimeTest, ToolValidationRetryExhaustionReturnsToolRetriesExhausted) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.max_tool_retries = 1;
    AgentRuntime runtime(config, std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });
    // Verify that tool calling was configured after registration
    {
        const auto ops = backend_ptr->operations();
        EXPECT_NE(std::find(ops.begin(), ops.end(), "set_tool_calling"), ops.end());
    }

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            tool_call_generation("double_value", nlohmann::json{{"value", "wrong"}}));
    });
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            tool_call_generation("double_value", nlohmann::json{{"value", "wrong"}}));
    });

    auto handle = runtime.chat(Message::user("use the tool"));
    auto result = handle.future.get();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolRetriesExhausted);
}

TEST(AgentRuntimeTest, ToolLoopLimitExhaustionReturnsToolLoopLimitReached) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.max_tool_iterations = 2;
    AgentRuntime runtime(config, std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            tool_call_generation("double_value", nlohmann::json{{"value", 1}}, "call-1"));
    });
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            tool_call_generation("double_value", nlohmann::json{{"value", 2}}, "call-2"));
    });

    auto handle = runtime.chat(Message::user("use the tool twice"));
    auto result = handle.future.get();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolLoopLimitReached);
}

TEST(AgentRuntimeTest, SuccessfulToolCallPopulatesToolInvocations) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            tool_call_generation("double_value", nlohmann::json{{"value", 5}}));
    });
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"The result is 10", 0, false});
    });

    auto handle = runtime.chat(Message::user("double 5"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "The result is 10");
    ASSERT_EQ(result->tool_invocations.size(), 1u);

    const auto& inv = result->tool_invocations[0];
    EXPECT_EQ(inv.id, "call-1");
    EXPECT_EQ(inv.name, "double_value");
    EXPECT_EQ(inv.status, zoo::ToolInvocationStatus::Succeeded);
    ASSERT_TRUE(inv.result_json.has_value());
    EXPECT_EQ(*inv.result_json, "{\"result\":10}");
    EXPECT_FALSE(inv.error.has_value());
}

TEST(AgentRuntimeTest, ToolEnabledFencedCodeReplyCompletesWithoutSpuriousToolCalls) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{"```python\nprint(\"hello world\")\n```", 6, false});
    });

    auto handle = runtime.chat(Message::user("Write a fenced Python hello world example."));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "```python\nprint(\"hello world\")\n```");
    EXPECT_TRUE(result->tool_invocations.empty());

    const auto ops = backend_ptr->operations();
    EXPECT_LT(index_of(ops, "set_tool_calling"), index_of(ops, "generate"));
    EXPECT_LT(index_of(ops, "generate"), index_of(ops, "finalize"));
}

TEST(AgentRuntimeTest, ValidationFailureThenSuccessPopulatesToolInvocations) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.max_tool_retries = 2;
    AgentRuntime runtime(config, std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });

    // First attempt: bad arguments (string instead of int)
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            tool_call_generation("double_value", nlohmann::json{{"value", "wrong"}}, "call-bad"));
    });
    // Second attempt: valid arguments
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            tool_call_generation("double_value", nlohmann::json{{"value", 7}}, "call-good"));
    });
    // Final text response
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"The result is 14", 0, false});
    });

    auto handle = runtime.chat(Message::user("double 7"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "The result is 14");
    ASSERT_EQ(result->tool_invocations.size(), 2u);

    const auto& failed = result->tool_invocations[0];
    EXPECT_EQ(failed.id, "call-bad");
    EXPECT_EQ(failed.name, "double_value");
    EXPECT_EQ(failed.status, zoo::ToolInvocationStatus::ValidationFailed);
    EXPECT_FALSE(failed.result_json.has_value());
    ASSERT_TRUE(failed.error.has_value());

    const auto& succeeded = result->tool_invocations[1];
    EXPECT_EQ(succeeded.id, "call-good");
    EXPECT_EQ(succeeded.name, "double_value");
    EXPECT_EQ(succeeded.status, zoo::ToolInvocationStatus::Succeeded);
    ASSERT_TRUE(succeeded.result_json.has_value());
    EXPECT_EQ(*succeeded.result_json, "{\"result\":14}");
    EXPECT_FALSE(succeeded.error.has_value());
}

TEST(AgentRuntimeTest, ExecutionFailurePopulatesToolInvocations) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    // Register a tool whose handler always returns an execution error.
    zoo::tools::ToolDefinition def;
    def.metadata.name = "failing_tool";
    def.metadata.description = "A tool that always fails";
    def.metadata.parameters_schema = {{"type", "object"},
                                      {"properties", {{"value", {{"type", "integer"}}}}},
                                      {"required", nlohmann::json::array({"value"})}};
    def.metadata.parameters = {
        zoo::tools::ToolParameter{"value", zoo::tools::ToolValueType::Integer, true, "", {}}};
    def.handler = [](const nlohmann::json&) -> Expected<nlohmann::json> {
        return std::unexpected(Error{ErrorCode::ToolExecutionFailed, "intentional failure"});
    };
    ASSERT_TRUE(runtime.register_tool(std::move(def)).has_value());

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            tool_call_generation("failing_tool", nlohmann::json{{"value", 42}}));
    });
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"Sorry, the tool failed", 0, false});
    });

    auto handle = runtime.chat(Message::user("use failing tool"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "Sorry, the tool failed");
    ASSERT_EQ(result->tool_invocations.size(), 1u);

    const auto& inv = result->tool_invocations[0];
    EXPECT_EQ(inv.id, "call-1");
    EXPECT_EQ(inv.name, "failing_tool");
    EXPECT_EQ(inv.status, zoo::ToolInvocationStatus::ExecutionFailed);
    EXPECT_FALSE(inv.result_json.has_value());
    ASSERT_TRUE(inv.error.has_value());
    EXPECT_EQ(inv.error->code, ErrorCode::ToolExecutionFailed);
    EXPECT_EQ(inv.error->message, "intentional failure");
}

TEST(AgentRuntimeTest, SetSystemPromptSerializesBeforeQueuedRequests) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.request_queue_capacity = 2;
    AgentRuntime runtime(config, std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](std::optional<TokenCallback>, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"first reply", 0, false});
        });
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"second reply", 0, false});
    });

    auto first = runtime.chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    auto second = runtime.chat(Message::user("second"));

    auto prompt_future =
        std::async(std::launch::async, [&runtime] { runtime.set_system_prompt("updated"); });

    EXPECT_EQ(prompt_future.wait_for(50ms), std::future_status::timeout);

    release->set_value();

    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());
    prompt_future.get();
    auto second_result = second.future.get();
    ASSERT_TRUE(second_result.has_value());

    const auto operations = backend_ptr->operations();
    EXPECT_LT(index_of(operations, "set_system_prompt:updated"),
              index_of(operations, "add:user:second"));
}

TEST(AgentRuntimeTest, GetHistorySerializesBeforeQueuedRequests) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.request_queue_capacity = 2;
    AgentRuntime runtime(config, std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](std::optional<TokenCallback>, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"first reply", 0, false});
        });
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"second reply", 0, false});
    });

    auto first = runtime.chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    auto second = runtime.chat(Message::user("second"));

    auto history_future =
        std::async(std::launch::async, [&runtime] { return runtime.get_history(); });
    EXPECT_EQ(history_future.wait_for(50ms), std::future_status::timeout);

    release->set_value();

    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());
    auto history = history_future.get();
    auto second_result = second.future.get();
    ASSERT_TRUE(second_result.has_value());

    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].content, "first");
    EXPECT_EQ(history[1].content, "first reply");

    const auto operations = backend_ptr->operations();
    EXPECT_LT(index_of(operations, "get_history"), index_of(operations, "add:user:second"));
}

TEST(AgentRuntimeTest, ClearHistorySerializesBeforeQueuedRequests) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.request_queue_capacity = 2;
    AgentRuntime runtime(config, std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](std::optional<TokenCallback>, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"first reply", 0, false});
        });
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"second reply", 0, false});
    });

    auto first = runtime.chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    auto second = runtime.chat(Message::user("second"));

    auto clear_future = std::async(std::launch::async, [&runtime] { runtime.clear_history(); });
    EXPECT_EQ(clear_future.wait_for(50ms), std::future_status::timeout);

    release->set_value();

    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());
    clear_future.get();
    auto second_result = second.future.get();
    ASSERT_TRUE(second_result.has_value());

    auto history = runtime.get_history();
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].content, "second");
    EXPECT_EQ(history[1].content, "second reply");

    const auto operations = backend_ptr->operations();
    EXPECT_LT(index_of(operations, "clear_history"), index_of(operations, "add:user:second"));
}

TEST(AgentRuntimeTest, CompleteUsesScopedHistoryAndRestoresPersistentHistory) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_config(), std::move(backend));

    runtime.set_system_prompt("persistent prompt");

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"persistent reply", 0, false});
    });

    auto first = runtime.chat(Message::user("persistent user"));
    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());
    EXPECT_EQ(first_result->text, "persistent reply");

    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"scoped reply", 0, false});
    });

    auto scoped =
        runtime.complete({Message::system("request prompt"), Message::user("request user")});
    auto scoped_result = scoped.future.get();
    ASSERT_TRUE(scoped_result.has_value());
    EXPECT_EQ(scoped_result->text, "scoped reply");

    const auto history = runtime.get_history();
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history[0].role, zoo::Role::System);
    EXPECT_EQ(history[0].content, "persistent prompt");
    EXPECT_EQ(history[1].role, zoo::Role::User);
    EXPECT_EQ(history[1].content, "persistent user");
    EXPECT_EQ(history[2].role, zoo::Role::Assistant);
    EXPECT_EQ(history[2].content, "persistent reply");
}

TEST(AgentRuntimeTest, CompleteRejectsEmptyMessageHistory) {
    auto backend = std::make_unique<FakeBackend>();
    AgentRuntime runtime(make_config(), std::move(backend));

    auto handle = runtime.complete({});
    auto result = handle.future.get();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

// ---------- Generic tool-calling format tests ----------

TEST(AgentRuntimeTest, GenericFormatUnwrapsResponseJson) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    backend_ptr->set_generic_tool_format(true);
    AgentRuntime runtime(make_config(), std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });

    // Generic format: model outputs {"response": "Hello"} with tool_call_detected=false
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{R"({"response": "Hello"})", 10, false});
    });

    auto handle = runtime.chat(Message::user("say hello"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "Hello");
    EXPECT_TRUE(result->tool_invocations.empty());
}

TEST(AgentRuntimeTest, GenericFormatDoesNotStreamRawJson) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    backend_ptr->set_generic_tool_format(true);
    AgentRuntime runtime(make_config(), std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });

    backend_ptr->push_generation([](std::optional<TokenCallback> on_token,
                                    const CancellationCallback&) {
        // Simulate token-by-token generation of the JSON wrapper
        if (on_token) {
            (*on_token)("{");
            (*on_token)(R"("response")");
            (*on_token)(":");
            (*on_token)(R"( "Hello")");
            (*on_token)("}");
        }
        return Expected<GenerationResult>(GenerationResult{R"({"response": "Hello"})", 10, false});
    });

    std::string streamed;
    auto handle = runtime.chat(Message::user("say hello"),
                               [&streamed](std::string_view token) { streamed += token; });
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    // The streamed output should be the unwrapped "Hello", not the raw JSON
    EXPECT_EQ(streamed, "Hello");
    EXPECT_EQ(result->text, "Hello");
}

TEST(AgentRuntimeTest, GenericFormatToolCallUsesUserFollowUp) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    backend_ptr->set_generic_tool_format(true);
    AgentRuntime runtime(make_config(), std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });

    // First generation: generic tool call
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        nlohmann::json payload = {
            {"tool_call",
             {{"id", "call-1"}, {"name", "double_value"}, {"arguments", {{"value", 5}}}}}};
        return Expected<GenerationResult>(GenerationResult{payload.dump(), 10, true});
    });
    // Second generation: final text response
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"response": "The result is 10"})", 10, false});
    });

    auto handle = runtime.chat(Message::user("double 5"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "The result is 10");
    ASSERT_EQ(result->tool_invocations.size(), 1u);

    // Verify the tool result was sent as a user message, not a tool message
    const auto ops = backend_ptr->operations();
    bool found_user_tool_result = false;
    for (const auto& op : ops) {
        if (op.find("add:user:Tool result for `double_value`") != std::string::npos) {
            found_user_tool_result = true;
            break;
        }
    }
    EXPECT_TRUE(found_user_tool_result)
        << "Expected user follow-up with tool result, not tool message";

    // Ensure no tool-role messages were added
    for (const auto& op : ops) {
        EXPECT_EQ(op.find("add:tool:"), std::string::npos)
            << "Generic format should not use Role::Tool, found: " << op;
    }
}

TEST(AgentRuntimeTest, GenericFormatValidationFailureUsesUserFollowUp) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    backend_ptr->set_generic_tool_format(true);
    Config config = make_config();
    config.max_tool_retries = 2;
    AgentRuntime runtime(config, std::move(backend));

    register_single_int_tool(runtime, "double_value", "Doubles a number",
                             [](int value) { return value * 2; });

    // First generation: invalid arguments
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        nlohmann::json payload = {
            {"tool_call",
             {{"id", "call-bad"}, {"name", "double_value"}, {"arguments", {{"value", "wrong"}}}}}};
        return Expected<GenerationResult>(GenerationResult{payload.dump(), 10, true});
    });
    // Second generation: valid arguments
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        nlohmann::json payload = {
            {"tool_call",
             {{"id", "call-good"}, {"name", "double_value"}, {"arguments", {{"value", 7}}}}}};
        return Expected<GenerationResult>(GenerationResult{payload.dump(), 10, true});
    });
    // Final text response
    backend_ptr->push_generation([](std::optional<TokenCallback>, const CancellationCallback&) {
        return Expected<GenerationResult>(
            GenerationResult{R"({"response": "The result is 14"})", 10, false});
    });

    auto handle = runtime.chat(Message::user("double 7"));
    auto result = handle.future.get();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "The result is 14");
    ASSERT_EQ(result->tool_invocations.size(), 2u);

    // Verify the validation failure was sent as a user message
    const auto ops = backend_ptr->operations();
    bool found_user_validation_failure = false;
    for (const auto& op : ops) {
        if (op.find("add:user:Tool call validation failed for `double_value`") !=
            std::string::npos) {
            found_user_validation_failure = true;
            break;
        }
    }
    EXPECT_TRUE(found_user_validation_failure)
        << "Expected user follow-up with validation failure, not tool message";

    // Ensure no tool-role messages were added
    for (const auto& op : ops) {
        EXPECT_EQ(op.find("add:tool:"), std::string::npos)
            << "Generic format should not use Role::Tool, found: " << op;
    }
}

} // namespace
