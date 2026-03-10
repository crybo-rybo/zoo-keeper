/**
 * @file test_agent_runtime.cpp
 * @brief Unit tests for the Agent runtime using a fake backend.
 */

#include "zoo/internal/agent/test_factory.hpp"
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

using zoo::Agent;
using zoo::CancellationCallback;
using zoo::Config;
using zoo::Error;
using zoo::ErrorCode;
using zoo::Expected;
using zoo::Message;
using zoo::TokenCallback;
using zoo::internal::agent::AgentBackend;
using zoo::internal::agent::GenerationResult;

class FakeBackend final : public AgentBackend {
  public:
    using GenerationAction =
        std::function<Expected<GenerationResult>(std::optional<TokenCallback>,
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

    Expected<void> add_message(const Message& message) override {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.push_back(message);
        operations_.push_back("add:" + std::string(zoo::role_to_string(message.role)) + ":" +
                              message.content);
        return {};
    }

    Expected<GenerationResult>
    generate_from_history(std::optional<TokenCallback> on_token,
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

    void clear_tool_grammar() override {
        std::lock_guard<std::mutex> lock(mutex_);
        last_tool_grammar_.clear();
        operations_.push_back("clear_tool_grammar");
    }

  private:
    mutable std::mutex mutex_;
    mutable std::vector<std::string> operations_;
    std::deque<GenerationAction> generations_;
    std::vector<Message> history_;
    std::string last_tool_grammar_;
    bool tool_grammar_supported_ = true;
};

Config make_config() {
    Config config;
    config.request_queue_capacity = 4;
    config.max_tool_iterations = 5;
    config.max_tool_retries = 2;
    return config;
}

std::unique_ptr<Agent> make_agent(Config config, std::unique_ptr<FakeBackend> backend) {
    return zoo::internal::agent::make_test_agent(config, std::move(backend));
}

GenerationResult tool_call_generation(const std::string& tool_name, const nlohmann::json& arguments,
                                      std::string id = "call-1") {
    nlohmann::json payload = {{"id", std::move(id)},
                              {"name", tool_name},
                              {"arguments", arguments}};
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
    auto agent = make_agent(config, std::move(backend));

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
    backend_ptr->push_generation(
        [](std::optional<TokenCallback>, const CancellationCallback&) {
            return Expected<GenerationResult>(GenerationResult{"second reply", 0, false});
        });

    auto first = agent->chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto second = agent->chat(Message::user("second"));
    auto third = agent->chat(Message::user("third"));

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
    auto agent = make_agent(make_config(), std::move(backend));

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

    auto first = agent->chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto second = agent->chat(Message::user("second"));
    agent->cancel(second.id);

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
    auto agent = make_agent(make_config(), std::move(backend));

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

    auto handle = agent->chat(Message::user("cancel me"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    agent->cancel(handle.id);

    auto result = handle.future.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::RequestCancelled);
}

TEST(AgentRuntimeTest, StopDrainsQueuedRequestsWithAgentNotRunning) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.request_queue_capacity = 3;
    auto agent = make_agent(config, std::move(backend));

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

    auto first = agent->chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);

    auto second = agent->chat(Message::user("second"));
    auto third = agent->chat(Message::user("third"));

    auto stop_future = std::async(std::launch::async, [&agent] { agent->stop(); });
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
    auto agent = make_agent(config, std::move(backend));

    ASSERT_TRUE(agent->register_tool("double_value", "Doubles a number", {"value"},
                                     [](int value) { return value * 2; })
                    .has_value());
    EXPECT_FALSE(backend_ptr->last_tool_grammar().empty());

    backend_ptr->push_generation(
        [](std::optional<TokenCallback>, const CancellationCallback&) {
            return Expected<GenerationResult>(
                tool_call_generation("double_value", nlohmann::json{{"value", "wrong"}}));
        });
    backend_ptr->push_generation(
        [](std::optional<TokenCallback>, const CancellationCallback&) {
            return Expected<GenerationResult>(
                tool_call_generation("double_value", nlohmann::json{{"value", "wrong"}}));
        });

    auto handle = agent->chat(Message::user("use the tool"));
    auto result = handle.future.get();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolRetriesExhausted);
}

TEST(AgentRuntimeTest, ToolLoopLimitExhaustionReturnsToolLoopLimitReached) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.max_tool_iterations = 2;
    auto agent = make_agent(config, std::move(backend));

    ASSERT_TRUE(agent->register_tool("double_value", "Doubles a number", {"value"},
                                     [](int value) { return value * 2; })
                    .has_value());

    backend_ptr->push_generation(
        [](std::optional<TokenCallback>, const CancellationCallback&) {
            return Expected<GenerationResult>(
                tool_call_generation("double_value", nlohmann::json{{"value", 1}}, "call-1"));
        });
    backend_ptr->push_generation(
        [](std::optional<TokenCallback>, const CancellationCallback&) {
            return Expected<GenerationResult>(
                tool_call_generation("double_value", nlohmann::json{{"value", 2}}, "call-2"));
        });

    auto handle = agent->chat(Message::user("use the tool twice"));
    auto result = handle.future.get();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolLoopLimitReached);
}

TEST(AgentRuntimeTest, SetSystemPromptSerializesBeforeQueuedRequests) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    Config config = make_config();
    config.request_queue_capacity = 2;
    auto agent = make_agent(config, std::move(backend));

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
    backend_ptr->push_generation(
        [](std::optional<TokenCallback>, const CancellationCallback&) {
            return Expected<GenerationResult>(GenerationResult{"second reply", 0, false});
        });

    auto first = agent->chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    auto second = agent->chat(Message::user("second"));

    auto prompt_future = std::async(std::launch::async, [&agent] {
        agent->set_system_prompt("updated");
    });

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
    auto agent = make_agent(config, std::move(backend));

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
    backend_ptr->push_generation(
        [](std::optional<TokenCallback>, const CancellationCallback&) {
            return Expected<GenerationResult>(GenerationResult{"second reply", 0, false});
        });

    auto first = agent->chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    auto second = agent->chat(Message::user("second"));

    auto history_future = std::async(std::launch::async, [&agent] { return agent->get_history(); });
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
    auto agent = make_agent(config, std::move(backend));

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
    backend_ptr->push_generation(
        [](std::optional<TokenCallback>, const CancellationCallback&) {
            return Expected<GenerationResult>(GenerationResult{"second reply", 0, false});
        });

    auto first = agent->chat(Message::user("first"));
    ASSERT_EQ(entered_future.wait_for(1s), std::future_status::ready);
    auto second = agent->chat(Message::user("second"));

    auto clear_future = std::async(std::launch::async, [&agent] {
        agent->clear_history();
    });
    EXPECT_EQ(clear_future.wait_for(50ms), std::future_status::timeout);

    release->set_value();

    auto first_result = first.future.get();
    ASSERT_TRUE(first_result.has_value());
    clear_future.get();
    auto second_result = second.future.get();
    ASSERT_TRUE(second_result.has_value());

    auto history = agent->get_history();
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].content, "second");
    EXPECT_EQ(history[1].content, "second reply");

    const auto operations = backend_ptr->operations();
    EXPECT_LT(index_of(operations, "clear_history"), index_of(operations, "add:user:second"));
}

TEST(AgentRuntimeTest, BuildToolSystemPromptUsesPublishedGrammarSnapshot) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    auto agent = make_agent(make_config(), std::move(backend));

    backend_ptr->set_tool_grammar_supported(true);
    ASSERT_TRUE(agent->register_tool("adder", "Adds a number", {"value"},
                                     [](int value) { return value + 1; })
                    .has_value());

    std::string prompt = agent->build_tool_system_prompt("base");
    EXPECT_NE(prompt.find("<tool_call>"), std::string::npos);
    EXPECT_EQ(prompt.find("respond with a JSON object"), std::string::npos);
}

TEST(AgentRuntimeTest, BuildToolSystemPromptFallsBackToJsonWhenGrammarRefreshFails) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    auto agent = make_agent(make_config(), std::move(backend));

    backend_ptr->set_tool_grammar_supported(false);
    ASSERT_TRUE(agent->register_tool("adder", "Adds a number", {"value"},
                                     [](int value) { return value + 1; })
                    .has_value());

    std::string prompt = agent->build_tool_system_prompt("base");
    EXPECT_NE(prompt.find("respond with a JSON object"), std::string::npos);
    EXPECT_EQ(prompt.find("<tool_call>"), std::string::npos);
}

} // namespace
