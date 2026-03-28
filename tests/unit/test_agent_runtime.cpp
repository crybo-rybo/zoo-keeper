/**
 * @file test_agent_runtime.cpp
 * @brief Unit tests for the internal agent runtime using a fake backend.
 */

#include "agent/runtime.hpp"
#include <gtest/gtest.h>

#include <chrono>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

namespace {

using namespace std::chrono_literals;

using zoo::AgentConfig;
using zoo::CancellationCallback;
using zoo::Error;
using zoo::ErrorCode;
using zoo::Expected;
using zoo::GenerationOptions;
using zoo::HistorySnapshot;
using zoo::Message;
using zoo::MessageView;
using zoo::ModelConfig;
using zoo::Role;
using zoo::TextResponse;
using zoo::TokenAction;
using zoo::TokenCallback;
using zoo::ToolInvocationStatus;
using zoo::internal::agent::AgentBackend;
using zoo::internal::agent::AgentRuntime;
using zoo::internal::agent::GenerationResult;
using zoo::internal::agent::HistoryMode;
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

    void trim_history(size_t max_non_system_messages) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t system_offset =
            (!history_.empty() && history_.front().role == zoo::Role::System) ? 1u : 0u;
        if (history_.size() <= system_offset + max_non_system_messages) {
            return;
        }
        size_t erase_count = history_.size() - system_offset - max_non_system_messages;
        history_.erase(history_.begin() + static_cast<std::ptrdiff_t>(system_offset),
                       history_.begin() + static_cast<std::ptrdiff_t>(system_offset + erase_count));
    }

    bool set_tool_calling(const std::vector<zoo::CoreToolInfo>& tools) override {
        std::lock_guard<std::mutex> lock(mutex_);
        tool_calling_supported_ = !tools.empty();
        return tool_calling_supported_;
    }

    ParsedToolResponse parse_tool_response(std::string_view text) const override {
        ParsedToolResponse result;

        const std::string open_tag = "<tool_call>";
        const std::string close_tag = "</tool_call>";
        const std::string value(text);
        auto start = value.find(open_tag);
        auto end = value.find(close_tag);
        if (start != std::string::npos && end != std::string::npos) {
            auto json_str = value.substr(start + open_tag.size(), end - start - open_tag.size());
            auto j = nlohmann::json::parse(json_str, nullptr, false);
            if (!j.is_discarded()) {
                zoo::OwnedToolCall tc;
                tc.id = j.value("id", "");
                tc.name = j.value("name", "");
                if (j.contains("arguments")) {
                    tc.arguments_json = j["arguments"].dump();
                }
                result.tool_calls.push_back(std::move(tc));
            }
            result.content = value.substr(0, start);
            if (end + close_tag.size() < value.size()) {
                result.content += value.substr(end + close_tag.size());
            }
        } else {
            result.content = value;
        }
        return result;
    }

    const char* tool_calling_format_name() const noexcept override {
        return "fake";
    }

    bool set_schema_grammar(const std::string&) override {
        return true;
    }

    void clear_tool_grammar() override {}

  private:
    mutable std::mutex mutex_;
    std::deque<GenerationAction> generations_;
    std::vector<Message> history_;
    bool tool_calling_supported_ = true;
};

ModelConfig make_model_config() {
    ModelConfig config;
    config.model_path = "unused.gguf";
    return config;
}

AgentConfig make_agent_config(size_t capacity = 4, size_t max_history_messages = 64) {
    AgentConfig config;
    config.request_queue_capacity = capacity;
    config.max_history_messages = max_history_messages;
    config.max_tool_iterations = 5;
    config.max_tool_retries = 2;
    return config;
}

GenerationResult tool_call_generation(const std::string& tool_name, const nlohmann::json& arguments,
                                      std::string id = "call-1") {
    nlohmann::json payload = {{"id", std::move(id)}, {"name", tool_name}, {"arguments", arguments}};
    return GenerationResult{"<tool_call>" + payload.dump() + "</tool_call>", 0, true, "", {}};
}

TEST(AgentRuntimeTest, QueueFullFailsAdditionalRequestWhileSlotIsOccupied) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(1), GenerationOptions{},
                         std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](TokenCallback, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"first reply", 0, false, "", {}});
        });

    auto first = runtime.chat("first");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto second = runtime.chat("second");
    auto second_result = second.await_result();
    ASSERT_FALSE(second_result.has_value());
    EXPECT_EQ(second_result.error().code, ErrorCode::QueueFull);

    release->set_value();

    auto first_result = first.await_result();
    ASSERT_TRUE(first_result.has_value());
    EXPECT_EQ(first_result->text, "first reply");
}

TEST(AgentRuntimeTest, CancelBeforeProcessingBeginsFailsQueuedRequest) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(2), GenerationOptions{},
                         std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](TokenCallback, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"first reply", 0, false, "", {}});
        });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"second reply", 0, false, "", {}});
    });

    auto first = runtime.chat("first");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto second = runtime.chat("second");
    runtime.cancel(second.id());

    release->set_value();

    ASSERT_TRUE(first.await_result().has_value());

    auto second_result = second.await_result();
    ASSERT_FALSE(second_result.has_value());
    EXPECT_EQ(second_result.error().code, ErrorCode::RequestCancelled);
}

TEST(AgentRuntimeTest, CompleteDoesNotMutatePersistentHistory) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"persistent reply", 0, false, "", {}});
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"scoped reply", 0, false, "", {}});
    });

    auto persistent = runtime.chat("persistent user");
    ASSERT_TRUE(persistent.await_result().has_value());

    const auto before = runtime.get_history();
    ASSERT_FALSE(before.empty());

    const std::array<Message, 2> scoped_messages = {Message::system("request prompt"),
                                                    Message::user("request user")};
    auto scoped = runtime.complete(zoo::ConversationView{std::span<const Message>(scoped_messages)},
                                   GenerationOptions{});
    auto scoped_result = scoped.await_result();
    ASSERT_TRUE(scoped_result.has_value());
    EXPECT_EQ(scoped_result->text, "scoped reply");

    EXPECT_EQ(runtime.get_history(), before);
}

TEST(AgentRuntimeTest, ChatStreamingCallbackSurvivesTokenStreaming) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    backend_ptr->push_generation([](TokenCallback on_token, const CancellationCallback&) {
        if (on_token) {
            EXPECT_EQ(on_token("Once "), TokenAction::Continue);
            EXPECT_EQ(on_token("upon "), TokenAction::Continue);
            EXPECT_EQ(on_token("a time"), TokenAction::Continue);
        }
        return Expected<GenerationResult>(GenerationResult{"Once upon a time", 7, false, "", {}});
    });

    std::string streamed;
    auto handle = runtime.chat("Tell me a story", GenerationOptions{}, [&](std::string_view token) {
        streamed.append(token.data(), token.size());
    });

    auto result = handle.await_result();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "Once upon a time");
    EXPECT_EQ(streamed, "Once upon a time");
    EXPECT_EQ(result->usage.prompt_tokens, 7);
    EXPECT_EQ(result->usage.completion_tokens, 3);
}

TEST(AgentRuntimeTest, StatefulRequestsTrimRetainedHistoryToConfiguredLimit) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(4, 2), GenerationOptions{},
                         std::move(backend));

    runtime.set_system_prompt("Keep only the latest turn.");

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"first reply", 0, false, "", {}});
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"second reply", 0, false, "", {}});
    });

    ASSERT_TRUE(runtime.chat("first user").await_result().has_value());
    ASSERT_TRUE(runtime.chat("second user").await_result().has_value());

    const auto history = runtime.get_history();
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history[0].role, Role::System);
    EXPECT_EQ(history[1].content, "second user");
    EXPECT_EQ(history[2].content, "second reply");
}

TEST(AgentRuntimeTest, ToolLoopReturnsTraceOnlyWhenRequested) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto definition = zoo::tools::detail::make_tool_definition("double", "Double a number",
                                                               std::vector<std::string>{"value"},
                                                               [](int value) { return value * 2; });
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();
    ASSERT_TRUE(runtime.register_tool(std::move(*definition)).has_value());

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(tool_call_generation("double", {{"value", 5}}));
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"10", 0, false, "", {}});
    });

    GenerationOptions options;
    options.record_tool_trace = true;
    auto handle = runtime.chat("double 5", options);
    auto result = handle.await_result();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "10");
    ASSERT_TRUE(result->tool_trace.has_value());
    ASSERT_EQ(result->tool_trace->invocations.size(), 1u);
    EXPECT_EQ(result->tool_trace->invocations[0].status, ToolInvocationStatus::Succeeded);
}

TEST(AgentRuntimeTest, SetSystemPromptUpdatesHistoryThroughCommandLane) {
    auto backend = std::make_unique<FakeBackend>();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    runtime.set_system_prompt("Be concise.");

    const auto history = runtime.get_history();
    ASSERT_EQ(history.size(), 1u);
    EXPECT_EQ(history[0].role, Role::System);
    EXPECT_EQ(history[0].content, "Be concise.");
}

TEST(AgentRuntimeTest, RegisterToolsBatchRegistersAllToolsWithSingleUpdate) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto def1 = zoo::tools::detail::make_tool_definition("add", "Add two numbers",
                                                         std::vector<std::string>{"a", "b"},
                                                         [](int a, int b) { return a + b; });
    auto def2 = zoo::tools::detail::make_tool_definition(
        "greet", "Greet someone", std::vector<std::string>{"name"},
        [](std::string name) { return "Hello, " + name + "!"; });
    ASSERT_TRUE(def1.has_value());
    ASSERT_TRUE(def2.has_value());

    std::vector<zoo::tools::ToolDefinition> definitions;
    definitions.push_back(std::move(*def1));
    definitions.push_back(std::move(*def2));

    auto result = runtime.register_tools(std::move(definitions));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(runtime.tool_count(), 2u);

    // Verify tools are usable via a tool-calling flow
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(tool_call_generation("add", {{"a", 3}, {"b", 4}}));
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"7", 0, false, "", {}});
    });

    GenerationOptions options;
    options.record_tool_trace = true;
    auto handle = runtime.chat("add 3 and 4", options);
    auto chat_result = handle.await_result();
    ASSERT_TRUE(chat_result.has_value()) << chat_result.error().to_string();
    EXPECT_EQ(chat_result->text, "7");
    ASSERT_TRUE(chat_result->tool_trace.has_value());
    EXPECT_EQ(chat_result->tool_trace->invocations[0].status, ToolInvocationStatus::Succeeded);
}

TEST(AgentRuntimeTest, StreamingCallbackRunsOffInferenceThread) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    std::thread::id inference_thread_id;
    std::thread::id callback_thread_id;

    backend_ptr->push_generation(
        [&inference_thread_id](TokenCallback on_token, const CancellationCallback&) {
            inference_thread_id = std::this_thread::get_id();
            if (on_token) {
                on_token("hello");
            }
            return Expected<GenerationResult>(GenerationResult{"hello", 5, false, "", {}});
        });

    auto handle = runtime.chat("test", GenerationOptions{}, [&](std::string_view) {
        callback_thread_id = std::this_thread::get_id();
    });

    auto result = handle.await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    EXPECT_NE(inference_thread_id, std::thread::id{});
    EXPECT_NE(callback_thread_id, std::thread::id{});
    EXPECT_NE(callback_thread_id, inference_thread_id);
}

} // namespace
