/**
 * @file test_agent_runtime.cpp
 * @brief Unit tests for the internal agent runtime using a fake backend.
 */

#include "agent/runtime.hpp"
#include "agent/runtime_helpers.hpp"
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

using namespace std::chrono_literals;

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
using zoo::RequestHandle;
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
using zoo::internal::agent::RequestHistoryScope;
using zoo::internal::agent::ScopeExit;

struct UnsupportedRequestResult {};

static_assert(requires { typename RequestHandle<TextResponse>; });
static_assert(requires { typename RequestHandle<ExtractionResponse>; });
static_assert(!zoo::internal::agent::RequestHandleResult<UnsupportedRequestResult>);

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

    Expected<GenerationResult> generate_from_history(const GenerationOptions& options,
                                                     TokenCallback on_token,
                                                     CancellationCallback should_cancel) override {
        GenerationAction action;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_options_ = options;
            if (generations_.empty()) {
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "No scripted generation available"});
            }
            action = std::move(generations_.front());
            generations_.pop_front();
        }
        return action(on_token, should_cancel);
    }

    GenerationOptions last_generation_options() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_options_;
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
    GenerationOptions last_options_;
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

nlohmann::json simple_extraction_schema() {
    return {{"type", "object"},
            {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
            {"required", nlohmann::json::array({"name", "age"})},
            {"additionalProperties", false}};
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

TEST(AgentRuntimeTest, CancelDuringGenerationPropagatesRequestCancelled) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();

    backend_ptr->push_generation([entered](TokenCallback,
                                           const CancellationCallback& should_cancel) {
        entered->set_value();
        for (int attempt = 0; attempt < 100 && !should_cancel(); ++attempt) {
            std::this_thread::sleep_for(5ms);
        }
        if (should_cancel()) {
            return Expected<GenerationResult>(
                std::unexpected(Error{ErrorCode::RequestCancelled, "cancelled by test"}));
        }
        return Expected<GenerationResult>(GenerationResult{"unexpected reply", 0, false, "", {}});
    });

    auto handle = runtime.chat("cancel me");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    runtime.cancel(handle.id());

    auto result = handle.await_result(1s);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::RequestCancelled);
}

TEST(AgentRuntimeTest, RequestHandleCancelCancelsRunningRequest) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();

    backend_ptr->push_generation([entered](TokenCallback,
                                           const CancellationCallback& should_cancel) {
        entered->set_value();
        for (int attempt = 0; attempt < 100 && !should_cancel(); ++attempt) {
            std::this_thread::sleep_for(5ms);
        }
        if (should_cancel()) {
            return Expected<GenerationResult>(
                std::unexpected(Error{ErrorCode::RequestCancelled, "cancelled by handle"}));
        }
        return Expected<GenerationResult>(GenerationResult{"unexpected reply", 0, false, "", {}});
    });

    auto handle = runtime.chat("cancel me");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    handle.cancel();

    auto result = handle.await_result(1s);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::RequestCancelled);
}

TEST(AgentRuntimeTest, AwaitResultTimeoutDoesNotConsumeHandle) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](TokenCallback, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"eventual reply", 0, false, "", {}});
        });

    auto handle = runtime.chat("wait for me");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto timed_out = handle.await_result(20ms);
    ASSERT_FALSE(timed_out.has_value());
    EXPECT_EQ(timed_out.error().code, ErrorCode::RequestTimeout);
    EXPECT_TRUE(handle.valid());
    EXPECT_FALSE(handle.ready());

    release->set_value();

    auto result = handle.await_result(1s);
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "eventual reply");
    EXPECT_FALSE(handle.valid());
}

TEST(AgentRuntimeTest, DroppingRunningRequestHandleDoesNotLeakSlotAfterCompletion) {
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
            return Expected<GenerationResult>(GenerationResult{"orphaned reply", 0, false, "", {}});
        });

    {
        auto handle = runtime.chat("orphan this request");
        ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    }

    release->set_value();

    auto history = runtime.get_history(1s);
    ASSERT_TRUE(history.has_value()) << history.error().to_string();

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"second reply", 0, false, "", {}});
    });

    auto second = runtime.chat("slot should be reusable");
    auto result = second.await_result(1s);
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "second reply");
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

TEST(AgentRuntimeTest, ChatStreamingTokenCallbackCanStopGeneration) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    backend_ptr->push_generation([](TokenCallback on_token, const CancellationCallback&) {
        if (on_token) {
            EXPECT_EQ(on_token("enough"), TokenAction::Stop);
        }
        return Expected<GenerationResult>(GenerationResult{"enough", 4, false, "", {}});
    });

    std::string streamed;
    auto handle =
        runtime.chat("Tell me something short", GenerationOptions{}, [&](std::string_view token) {
            streamed.append(token.data(), token.size());
            return TokenAction::Stop;
        });

    auto result = handle.await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "enough");
    EXPECT_EQ(streamed, "enough");
    EXPECT_EQ(result->usage.completion_tokens, 1);
}

TEST(AgentRuntimeTest, GenerationOverrideInheritsConfiguredDefaults) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    GenerationOptions defaults;
    defaults.max_tokens = 17;
    AgentRuntime runtime(make_model_config(), make_agent_config(), defaults, std::move(backend));

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"ok", 0, false, "", {}});
    });

    auto result =
        runtime.chat("inherit", zoo::GenerationOverride::inherit_defaults()).await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(backend_ptr->last_generation_options().max_tokens, 17);
}

TEST(AgentRuntimeTest, GenerationOverrideUsesExplicitNonDefaultOptions) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    GenerationOptions defaults;
    defaults.max_tokens = 17;
    AgentRuntime runtime(make_model_config(), make_agent_config(), defaults, std::move(backend));

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"ok", 0, false, "", {}});
    });

    GenerationOptions explicit_options;
    explicit_options.max_tokens = 3;
    auto result =
        runtime.chat("explicit", zoo::GenerationOverride::explicit_options(explicit_options))
            .await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(backend_ptr->last_generation_options().max_tokens, 3);
}

TEST(AgentRuntimeTest, GenerationOverrideCanRequestBuiltInDefaults) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    GenerationOptions defaults;
    defaults.max_tokens = 17;
    AgentRuntime runtime(make_model_config(), make_agent_config(), defaults, std::move(backend));

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"ok", 0, false, "", {}});
    });

    auto result =
        runtime.chat("builtin", zoo::GenerationOverride::explicit_options(GenerationOptions{}))
            .await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(backend_ptr->last_generation_options().max_tokens, -1);
}

TEST(AgentRuntimeTest, ChatStreamingCallbackFailureFailsRequest) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    backend_ptr->push_generation([](TokenCallback on_token, const CancellationCallback&) {
        if (on_token) {
            EXPECT_EQ(on_token("boom"), TokenAction::Continue);
        }
        return Expected<GenerationResult>(GenerationResult{"boom", 1, false, "", {}});
    });

    auto handle = runtime.chat("trigger callback", GenerationOptions{},
                               [](std::string_view) { throw std::runtime_error("callback boom"); });
    auto result = handle.await_result();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InferenceFailed);
    EXPECT_NE(result.error().message.find("callback boom"), std::string::npos);
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

TEST(AgentRuntimeTest, ToolCallingWorksAfterSchemaExtractionRestoresToolGrammar) {
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
        return Expected<GenerationResult>(
            GenerationResult{R"({"name":"Alice","age":30})", 0, false, "", {}});
    });
    auto extraction = runtime.extract(simple_extraction_schema(), "Alice is 30").await_result();
    ASSERT_TRUE(extraction.has_value()) << extraction.error().to_string();
    EXPECT_EQ(extraction->data["name"], "Alice");

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(tool_call_generation("double", {{"value", 5}}));
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"10", 0, false, "", {}});
    });

    GenerationOptions options;
    options.record_tool_trace = true;
    auto result = runtime.chat("double 5", options).await_result();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "10");
    ASSERT_TRUE(result->tool_trace.has_value());
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

TEST(AgentRuntimeTest, TryCommandMethodsReportStoppedAgent) {
    auto backend = std::make_unique<FakeBackend>();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));
    runtime.stop();

    auto set_result = runtime.try_set_system_prompt("ignored");
    ASSERT_FALSE(set_result.has_value());
    EXPECT_EQ(set_result.error().code, ErrorCode::AgentNotRunning);

    auto history_result = runtime.try_get_history();
    ASSERT_FALSE(history_result.has_value());
    EXPECT_EQ(history_result.error().code, ErrorCode::AgentNotRunning);

    auto clear_result = runtime.try_clear_history();
    ASSERT_FALSE(clear_result.has_value());
    EXPECT_EQ(clear_result.error().code, ErrorCode::AgentNotRunning);
}

TEST(AgentRuntimeTest, StopResolvesQueuedRequestAndCommandWithoutDeadlock) {
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

    auto first = runtime.chat("running request");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    auto queued = runtime.chat("queued request");

    auto command_future =
        std::async(std::launch::async, [&runtime] { return runtime.try_get_history(); });
    ASSERT_EQ(command_future.wait_for(50ms), std::future_status::timeout);

    auto stop_future = std::async(std::launch::async, [&runtime] { runtime.stop(); });
    ASSERT_EQ(stop_future.wait_for(50ms), std::future_status::timeout);

    release->set_value();

    ASSERT_EQ(stop_future.wait_for(1s), std::future_status::ready);

    auto first_result = first.await_result(1s);
    ASSERT_TRUE(first_result.has_value()) << first_result.error().to_string();

    auto queued_result = queued.await_result(1s);
    ASSERT_FALSE(queued_result.has_value());
    EXPECT_EQ(queued_result.error().code, ErrorCode::AgentNotRunning);

    auto command_result = command_future.get();
    ASSERT_FALSE(command_result.has_value());
    EXPECT_EQ(command_result.error().code, ErrorCode::AgentNotRunning);
}

TEST(AgentRuntimeTest, SetSystemPromptTimeoutReturnsRequestTimeoutWhenCommandLaneIsBusy) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](TokenCallback, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"done", 0, false, "", {}});
        });

    auto request = runtime.chat("occupy inference thread");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto timed_out = runtime.set_system_prompt("Do not wait forever.", 20ms);
    ASSERT_FALSE(timed_out.has_value());
    EXPECT_EQ(timed_out.error().code, ErrorCode::RequestTimeout);

    release->set_value();
    ASSERT_TRUE(request.await_result(1s).has_value());

    auto history = runtime.get_history(1s);
    ASSERT_TRUE(history.has_value()) << history.error().to_string();
    ASSERT_GE(history->size(), 1u);
    EXPECT_EQ((*history)[0].role, Role::System);
    EXPECT_EQ((*history)[0].content, "Do not wait forever.");
}

TEST(AgentRuntimeTest, AddSystemMessageAppendsWithoutReplacingExistingSystemPrompt) {
    auto backend = std::make_unique<FakeBackend>();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    runtime.set_system_prompt("You are a helpful NPC.");

    auto result = runtime.add_system_message("Mood: suspicious. Trust: low.");
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    const auto history = runtime.get_history();
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].role, Role::System);
    EXPECT_EQ(history[0].content, "You are a helpful NPC.");
    EXPECT_EQ(history[1].role, Role::System);
    EXPECT_EQ(history[1].content, "Mood: suspicious. Trust: low.");
}

TEST(AgentRuntimeTest, AddSystemMessageTimeoutReturnsRequestTimeoutWhenBusy) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](TokenCallback, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"done", 0, false, "", {}});
        });

    auto request = runtime.chat("occupy inference thread");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto timed_out = runtime.add_system_message("ephemeral context", 20ms);
    ASSERT_FALSE(timed_out.has_value());
    EXPECT_EQ(timed_out.error().code, ErrorCode::RequestTimeout);

    release->set_value();
    ASSERT_TRUE(request.await_result(1s).has_value());
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

TEST(AgentRuntimeTest, RegisterToolWaitsUntilCurrentRequestCompletesBeforeBecomingVisible) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();

    auto config = make_agent_config();
    config.max_tool_retries = 0;
    AgentRuntime runtime(make_model_config(), config, GenerationOptions{}, std::move(backend));

    auto existing = zoo::tools::detail::make_tool_definition("existing", "Existing tool",
                                                             std::vector<std::string>{"value"},
                                                             [](int value) { return value; });
    ASSERT_TRUE(existing.has_value()) << existing.error().to_string();
    ASSERT_TRUE(runtime.register_tool(std::move(*existing)).has_value());

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](TokenCallback, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(tool_call_generation("late", {{"value", 7}}));
        });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"late result", 0, false, "", {}});
    });

    auto handle = runtime.chat("call late");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto register_future = std::async(std::launch::async, [&runtime]() {
        auto definition = zoo::tools::detail::make_tool_definition(
            "late", "Late tool", std::vector<std::string>{"value"},
            [](int value) { return value * 2; });
        if (!definition) {
            return Expected<void>(std::unexpected(definition.error()));
        }
        return runtime.register_tool(std::move(*definition));
    });

    EXPECT_EQ(register_future.wait_for(50ms), std::future_status::timeout);

    release->set_value();

    auto result = handle.await_result();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolRetriesExhausted);

    auto register_result = register_future.get();
    ASSERT_TRUE(register_result.has_value()) << register_result.error().to_string();
    EXPECT_EQ(runtime.tool_count(), 2u);
}

TEST(AgentRuntimeTest, ToolCountCanBeReadWhileRegistrationIsQueued) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto existing = zoo::tools::detail::make_tool_definition("existing", "Existing tool",
                                                             std::vector<std::string>{"value"},
                                                             [](int value) { return value; });
    ASSERT_TRUE(existing.has_value()) << existing.error().to_string();
    ASSERT_TRUE(runtime.register_tool(std::move(*existing)).has_value());
    ASSERT_EQ(runtime.tool_count(), 1u);

    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    backend_ptr->push_generation(
        [entered, release_future](TokenCallback, const CancellationCallback&) {
            entered->set_value();
            release_future.wait();
            return Expected<GenerationResult>(GenerationResult{"done", 0, false, "", {}});
        });

    auto request = runtime.chat("occupy inference thread");
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    std::atomic<bool> keep_reading{true};
    std::atomic<size_t> reads{0};
    std::atomic<size_t> unexpected_counts{0};

    auto register_future = std::async(std::launch::async, [&runtime]() {
        auto definition = zoo::tools::detail::make_tool_definition(
            "late", "Late tool", std::vector<std::string>{"value"},
            [](int value) { return value * 2; });
        if (!definition) {
            return Expected<void>(std::unexpected(definition.error()));
        }
        return runtime.register_tool(std::move(*definition));
    });
    auto reader_future =
        std::async(std::launch::async, [&runtime, &keep_reading, &reads, &unexpected_counts]() {
            while (keep_reading.load(std::memory_order_acquire)) {
                const auto count = runtime.tool_count();
                if (count != 1u && count != 2u) {
                    unexpected_counts.fetch_add(1u, std::memory_order_relaxed);
                }
                reads.fetch_add(1u, std::memory_order_relaxed);
                std::this_thread::yield();
            }
        });
    bool released = false;
    ScopeExit cleanup([&] {
        keep_reading.store(false, std::memory_order_release);
        if (!released) {
            release->set_value();
        }
    });

    EXPECT_EQ(register_future.wait_for(50ms), std::future_status::timeout);
    EXPECT_EQ(runtime.tool_count(), 1u);

    released = true;
    release->set_value();

    auto result = request.await_result(1s);
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    auto register_result = register_future.get();
    ASSERT_TRUE(register_result.has_value()) << register_result.error().to_string();

    keep_reading.store(false, std::memory_order_release);
    reader_future.get();

    EXPECT_GT(reads.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(unexpected_counts.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(runtime.tool_count(), 2u);
}

TEST(AgentRuntimeTest, ToolLoopThroughputExcludesToolExecutionTime) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    // Use a deliberately large tool sleep (500ms) so that the gap between
    // "with exclusion" and "without exclusion" is unambiguous on any runner.
    // Without exclusion: 2 tokens / ~520ms ≈ 3.8 tok/s.
    // With exclusion:    2 tokens / ~20ms  ≈ 100  tok/s (even on slow CI ≥ 20).
    auto definition = zoo::tools::detail::make_tool_definition(
        "slow", "Slow tool", std::vector<std::string>{"value"}, [](int value) {
            std::this_thread::sleep_for(500ms);
            return value * 2;
        });
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();
    ASSERT_TRUE(runtime.register_tool(std::move(*definition)).has_value());

    backend_ptr->push_generation([](TokenCallback on_token, const CancellationCallback&) {
        if (on_token) {
            EXPECT_EQ(on_token("a"), TokenAction::Continue);
            std::this_thread::sleep_for(1ms); // ensure generation_time > 0ms
        }
        return Expected<GenerationResult>(tool_call_generation("slow", {{"value", 5}}));
    });
    backend_ptr->push_generation([](TokenCallback on_token, const CancellationCallback&) {
        if (on_token) {
            EXPECT_EQ(on_token("b"), TokenAction::Continue);
            std::this_thread::sleep_for(1ms);
        }
        return Expected<GenerationResult>(GenerationResult{"done", 0, false, "", {}});
    });

    auto result = runtime.chat("run slow tool").await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "done");
    EXPECT_EQ(result->usage.completion_tokens, 2);
    EXPECT_GT(result->metrics.tokens_per_second, 5.0);
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

TEST(AgentRuntimeTest, ToolHandlerRunsOffInferenceThread) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    std::thread::id inference_thread_id;
    std::thread::id handler_thread_id;

    auto definition = zoo::tools::detail::make_tool_definition(
        "capture_id", "Captures the handler thread id", std::vector<std::string>{"value"},
        [&handler_thread_id](int) -> int {
            handler_thread_id = std::this_thread::get_id();
            return 0;
        });
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();
    ASSERT_TRUE(runtime.register_tool(std::move(*definition)).has_value());

    backend_ptr->push_generation(
        [&inference_thread_id](TokenCallback, const CancellationCallback&) {
            inference_thread_id = std::this_thread::get_id();
            return Expected<GenerationResult>(tool_call_generation("capture_id", {{"value", 1}}));
        });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"done", 0, false, "", {}});
    });

    auto result = runtime.chat("go").await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    EXPECT_NE(inference_thread_id, std::thread::id{});
    EXPECT_NE(handler_thread_id, std::thread::id{});
    EXPECT_NE(handler_thread_id, inference_thread_id);
    EXPECT_NE(handler_thread_id, std::this_thread::get_id());
}

TEST(AgentRuntimeTest, CancelDuringToolHandlerReturnsRequestCancelledAfterHandlerCompletes) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto entered_tool = std::make_shared<std::promise<void>>();
    auto entered_tool_future = entered_tool->get_future();
    auto release_tool = std::make_shared<std::promise<void>>();
    auto release_tool_future = release_tool->get_future().share();

    auto definition = zoo::tools::detail::make_tool_definition(
        "slow", "Blocks until released", std::vector<std::string>{"value"},
        [entered_tool, release_tool_future](int value) {
            entered_tool->set_value();
            release_tool_future.wait();
            return value;
        });
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();
    ASSERT_TRUE(runtime.register_tool(std::move(*definition)).has_value());

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(tool_call_generation("slow", {{"value", 1}}));
    });

    auto handle = runtime.chat("run slow tool");
    ASSERT_EQ(entered_tool_future.wait_for(1s), std::future_status::ready);

    handle.cancel();
    release_tool->set_value();

    auto result = handle.await_result(1s);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::RequestCancelled);
}

TEST(AgentRuntimeTest, ToolHandlerExceptionsBecomeExecutionFailedErrors) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    // Register a raw handler that throws (bypasses the make_tool_definition wrapper).
    // Use a schema with one parameter so validation passes and the handler is reached.
    nlohmann::json schema = {{"type", "object"},
                             {"properties", nlohmann::json{{"x", {{"type", "integer"}}}}},
                             {"required", nlohmann::json::array({"x"})},
                             {"additionalProperties", false}};
    zoo::tools::ToolHandler throwing_handler =
        [](const nlohmann::json&) -> Expected<nlohmann::json> {
        throw std::runtime_error("intentional error");
    };
    auto definition = zoo::tools::detail::make_tool_definition("thrower", "throws on every call",
                                                               schema, std::move(throwing_handler));
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();
    ASSERT_TRUE(runtime.register_tool(std::move(*definition)).has_value());

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(tool_call_generation("thrower", {{"x", 42}}));
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"recovered", 0, false, "", {}});
    });

    GenerationOptions options;
    options.record_tool_trace = true;
    auto result = runtime.chat("go", options).await_result();

    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    ASSERT_TRUE(result->tool_trace.has_value());
    ASSERT_EQ(result->tool_trace->invocations.size(), 1u);
    EXPECT_EQ(result->tool_trace->invocations[0].status, ToolInvocationStatus::ExecutionFailed);
}

TEST(RequestHistoryScopeTest, ReplaceRestoresOriginalHistoryOnExit) {
    FakeBackend backend;
    ASSERT_TRUE(backend.add_message(Message::system("base prompt").view()).has_value());
    ASSERT_TRUE(backend.add_message(Message::user("persistent user").view()).has_value());
    const auto before = backend.get_history();

    const std::vector<Message> scoped_messages = {Message::system("scoped prompt"),
                                                  Message::user("scoped user")};
    {
        auto scope =
            RequestHistoryScope::enter(backend, HistoryMode::Replace, scoped_messages, 64, "chat");
        ASSERT_TRUE(scope.has_value()) << scope.error().to_string();
        ASSERT_TRUE(backend.add_message(Message::assistant("scoped reply").view()).has_value());
        const auto scoped = backend.get_history();
        ASSERT_EQ(scoped.size(), 3u);
        EXPECT_EQ(scoped[0].content, "scoped prompt");
        EXPECT_EQ(scoped[2].content, "scoped reply");
    }

    EXPECT_EQ(backend.get_history(), before);
}

TEST(RequestHistoryScopeTest, AppendTrimsRetainedHistoryOnExit) {
    FakeBackend backend;
    backend.set_system_prompt("retain system");
    ASSERT_TRUE(backend.add_message(Message::user("old user").view()).has_value());
    ASSERT_TRUE(backend.add_message(Message::assistant("old reply").view()).has_value());

    const std::vector<Message> request_messages = {Message::user("new user")};
    {
        auto scope =
            RequestHistoryScope::enter(backend, HistoryMode::Append, request_messages, 2, "chat");
        ASSERT_TRUE(scope.has_value()) << scope.error().to_string();
        ASSERT_TRUE(backend.add_message(Message::assistant("new reply").view()).has_value());
        ASSERT_EQ(backend.get_history().size(), 5u);
    }

    const auto history = backend.get_history();
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history[0].role, Role::System);
    EXPECT_EQ(history[1].content, "new user");
    EXPECT_EQ(history[2].content, "new reply");
}

TEST(AgentRuntimeTest, ToolExecutionCompletesCleanlyBeforeRuntimeDestruction) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    auto runtime = std::make_unique<AgentRuntime>(make_model_config(), make_agent_config(),
                                                  GenerationOptions{}, std::move(backend));

    auto definition = zoo::tools::detail::make_tool_definition(
        "noop", "does nothing", std::vector<std::string>{"value"}, [](int) -> int { return 0; });
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();
    ASSERT_TRUE(runtime->register_tool(std::move(*definition)).has_value());

    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(tool_call_generation("noop", {{"value", 1}}));
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"done", 0, false, "", {}});
    });

    auto result = runtime->chat("go").await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    // Destructor must not hang — inference thread and ToolExecutor worker both exit cleanly.
    auto destroy_future = std::async(std::launch::async, [&] { runtime.reset(); });
    EXPECT_EQ(destroy_future.wait_for(3s), std::future_status::ready);
}

TEST(ScopeExitTest, MoveConstructionTransfersSingleExecution) {
    int calls = 0;

    {
        ScopeExit original([&] { ++calls; });
        ScopeExit moved(std::move(original));
    }

    EXPECT_EQ(calls, 1);
}

TEST(ScopeExitTest, MoveAssignmentTransfersSingleExecution) {
    int calls = 0;

    {
        ScopeExit original([&] { ++calls; });
        ScopeExit moved([] {});
        moved = std::move(original);
    }

    EXPECT_EQ(calls, 1);
}

} // namespace
