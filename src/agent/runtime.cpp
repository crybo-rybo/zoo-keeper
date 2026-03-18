/**
 * @file runtime.cpp
 * @brief Internal worker-owned runtime: lifecycle, request submission, and dispatch.
 *
 * The agentic tool loop lives in runtime_tool_loop.cpp.
 * Structured output extraction lives in runtime_extraction.cpp.
 */

#include "zoo/internal/agent/runtime.hpp"

#include "zoo/core/model.hpp"
#include "zoo/internal/log.hpp"
#include "zoo/tools/registry.hpp"
#include <cassert>
#include <chrono>
#include <functional>
#include <thread>
#include <variant>

namespace zoo::internal::agent {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

AgentRuntime::AgentRuntime(const Config& cfg, std::unique_ptr<AgentBackend> backend)
    : config_(cfg), backend_(std::move(backend)), request_mailbox_(cfg.request_queue_capacity) {
    inference_thread_ = std::thread([this]() { inference_loop(); });
}

AgentRuntime::~AgentRuntime() {
    stop();
}

void AgentRuntime::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    running_.store(false, std::memory_order_release);
    request_mailbox_.shutdown();
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
}

bool AgentRuntime::is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Request submission (calling thread)
// ---------------------------------------------------------------------------

RequestHandle AgentRuntime::chat(Message message,
                                 std::optional<std::function<void(std::string_view)>> callback) {
    auto prepared = request_tracker_.prepare(std::move(message), std::move(callback));
    RequestHandle handle{prepared.request.id, std::move(prepared.future)};

    if (!running_.load(std::memory_order_acquire)) {
        request_tracker_.fail(handle.id, Error{ErrorCode::AgentNotRunning, "Agent is not running"});
        return handle;
    }

    if (!request_mailbox_.push_request(std::move(prepared.request))) {
        request_tracker_.fail(handle.id, Error{ErrorCode::QueueFull,
                                               "Request queue is full or agent is shutting down"});
        return handle;
    }

    return handle;
}

RequestHandle
AgentRuntime::complete(std::vector<Message> messages,
                       std::optional<std::function<void(std::string_view)>> callback) {
    auto prepared =
        request_tracker_.prepare(std::move(messages), HistoryMode::Replace, std::move(callback));
    RequestHandle handle{prepared.request.id, std::move(prepared.future)};

    if (prepared.request.messages.empty()) {
        request_tracker_.fail(handle.id, Error{ErrorCode::InvalidMessageSequence,
                                               "Request must include at least one message"});
        return handle;
    }

    if (!running_.load(std::memory_order_acquire)) {
        request_tracker_.fail(handle.id, Error{ErrorCode::AgentNotRunning, "Agent is not running"});
        return handle;
    }

    if (!request_mailbox_.push_request(std::move(prepared.request))) {
        request_tracker_.fail(handle.id, Error{ErrorCode::QueueFull,
                                               "Request queue is full or agent is shutting down"});
        return handle;
    }

    return handle;
}

RequestHandle AgentRuntime::extract(const nlohmann::json& output_schema, Message message,
                                    std::optional<std::function<void(std::string_view)>> callback) {
    // Validate schema upfront on the calling thread (fail fast)
    auto params = tools::detail::normalize_schema(output_schema);
    if (!params) {
        auto prepared = request_tracker_.prepare(std::move(message), std::move(callback));
        RequestHandle handle{prepared.request.id, std::move(prepared.future)};
        request_tracker_.fail(handle.id,
                              Error{ErrorCode::InvalidOutputSchema, params.error().message});
        return handle;
    }

    auto prepared = request_tracker_.prepare(std::move(message), nlohmann::json(output_schema),
                                             std::move(callback));
    RequestHandle handle{prepared.request.id, std::move(prepared.future)};

    if (!running_.load(std::memory_order_acquire)) {
        request_tracker_.fail(handle.id, Error{ErrorCode::AgentNotRunning, "Agent is not running"});
        return handle;
    }

    if (!request_mailbox_.push_request(std::move(prepared.request))) {
        request_tracker_.fail(handle.id, Error{ErrorCode::QueueFull,
                                               "Request queue is full or agent is shutting down"});
        return handle;
    }

    return handle;
}

RequestHandle AgentRuntime::extract(const nlohmann::json& output_schema,
                                    std::vector<Message> messages,
                                    std::optional<std::function<void(std::string_view)>> callback) {
    // Validate schema upfront on the calling thread (fail fast)
    auto params = tools::detail::normalize_schema(output_schema);
    if (!params) {
        auto prepared = request_tracker_.prepare(std::move(messages), HistoryMode::Replace,
                                                 std::move(callback));
        RequestHandle handle{prepared.request.id, std::move(prepared.future)};
        request_tracker_.fail(handle.id,
                              Error{ErrorCode::InvalidOutputSchema, params.error().message});
        return handle;
    }

    auto prepared = request_tracker_.prepare(std::move(messages), HistoryMode::Replace,
                                             nlohmann::json(output_schema), std::move(callback));
    RequestHandle handle{prepared.request.id, std::move(prepared.future)};

    if (prepared.request.messages.empty()) {
        request_tracker_.fail(handle.id, Error{ErrorCode::InvalidMessageSequence,
                                               "Request must include at least one message"});
        return handle;
    }

    if (!running_.load(std::memory_order_acquire)) {
        request_tracker_.fail(handle.id, Error{ErrorCode::AgentNotRunning, "Agent is not running"});
        return handle;
    }

    if (!request_mailbox_.push_request(std::move(prepared.request))) {
        request_tracker_.fail(handle.id, Error{ErrorCode::QueueFull,
                                               "Request queue is full or agent is shutting down"});
        return handle;
    }

    return handle;
}

// ---------------------------------------------------------------------------
// Command submission (calling thread)
// ---------------------------------------------------------------------------

void AgentRuntime::cancel(RequestId id) {
    request_tracker_.cancel(id);
}

void AgentRuntime::set_system_prompt(const std::string& prompt) {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(SetSystemPromptCmd{prompt, std::move(done)})) {
        return;
    }
    future.get();
}

std::vector<Message> AgentRuntime::get_history() const {
    if (!running_.load(std::memory_order_acquire)) {
        return {};
    }

    auto done = std::make_shared<std::promise<std::vector<Message>>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(GetHistoryCmd{std::move(done)})) {
        return {};
    }
    return future.get();
}

void AgentRuntime::clear_history() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(ClearHistoryCmd{std::move(done)})) {
        return;
    }
    future.get();
}

// ---------------------------------------------------------------------------
// Tool registration (calling thread)
// ---------------------------------------------------------------------------

Expected<void> AgentRuntime::register_tool(tools::ToolDefinition definition) {
    assert(!inference_thread_.joinable() ||
           std::this_thread::get_id() != inference_thread_.get_id());

    if (auto result = tool_registry_.register_tool(std::move(definition)); !result) {
        return std::unexpected(result.error());
    }

    update_tool_calling();
    return {};
}

size_t AgentRuntime::tool_count() const noexcept {
    return tool_registry_.size();
}

// ---------------------------------------------------------------------------
// Inference thread
// ---------------------------------------------------------------------------

void AgentRuntime::inference_loop() {
    try {
        while (running_.load(std::memory_order_acquire)) {
            auto item_opt = request_mailbox_.pop();
            if (!item_opt) {
                break;
            }

            std::visit(overloaded{
                           [this](Request& request) { handle_request(request); },
                           [this](Command& cmd) { handle_command(cmd); },
                       },
                       *item_opt);
        }

        fail_pending(
            Error{ErrorCode::AgentNotRunning, "Agent stopped before request could be processed"});
    } catch (const std::exception& e) {
        ZOO_LOG("error", "fatal exception escaped inference thread: %s", e.what());
        fail_pending(Error{ErrorCode::InferenceFailed,
                           std::string("Inference thread terminated unexpectedly: ") + e.what()});
    } catch (...) {
        ZOO_LOG("error", "fatal unknown exception escaped inference thread");
        fail_pending(Error{ErrorCode::InferenceFailed, "Inference thread terminated unexpectedly"});
    }
}

void AgentRuntime::handle_request(Request& request) {
    auto promise = request.promise;

    if (request.cancelled && request.cancelled->load(std::memory_order_acquire)) {
        ZOO_LOG("info", "request %lu cancelled before processing",
                static_cast<unsigned long>(request.id));
        request_tracker_.fail(request.id, Error{ErrorCode::RequestCancelled, "Request cancelled"});
        return;
    }

    Expected<Response> result;
    try {
        result = process_request(request);
    } catch (const std::exception& e) {
        ZOO_LOG("error", "unhandled exception in inference: %s", e.what());
        result = std::unexpected(
            Error{ErrorCode::InferenceFailed, std::string("Unhandled exception: ") + e.what()});
    } catch (...) {
        ZOO_LOG("error", "unknown exception in inference thread");
        result = std::unexpected(
            Error{ErrorCode::InferenceFailed, "Unknown exception in inference thread"});
    }

    request_tracker_.cleanup(request.id);

    if (promise) {
        promise->set_value(std::move(result));
    }
}

void AgentRuntime::handle_command(Command& cmd) {
    std::visit(overloaded{
                   [this](SetSystemPromptCmd& c) {
                       backend_->set_system_prompt(c.prompt);
                       c.done->set_value();
                   },
                   [this](GetHistoryCmd& c) { c.done->set_value(backend_->get_history()); },
                   [this](ClearHistoryCmd& c) {
                       backend_->clear_history();
                       c.done->set_value();
                   },
                   [this](RefreshToolCallingCmd& c) {
                       bool active = false;
                       if (c.tools.empty()) {
                           backend_->clear_tool_grammar();
                       } else if (backend_->set_tool_calling(c.tools)) {
                           active = true;
                           ZOO_LOG("info", "tool calling configured (%zu tools, format=%s)",
                                   c.tools.size(), backend_->tool_calling_format_name());
                       } else {
                           backend_->clear_tool_grammar();
                           ZOO_LOG("warn", "tool calling setup failed, falling back to "
                                           "unconstrained generation");
                       }
                       tool_grammar_active_.store(active, std::memory_order_release);
                       c.done->set_value(active);
                   },
               },
               cmd);
}

// ---------------------------------------------------------------------------
// Shutdown helpers
// ---------------------------------------------------------------------------

void AgentRuntime::fail_pending(const Error& error) {
    running_.store(false, std::memory_order_release);
    request_mailbox_.shutdown();

    while (auto remaining = request_mailbox_.pop()) {
        std::visit(overloaded{
                       [&](Request& request) { request_tracker_.fail(request.id, error); },
                       [](Command& cmd) { resolve_command_on_shutdown(cmd); },
                   },
                   *remaining);
    }

    request_tracker_.fail_all(error);
}

void AgentRuntime::resolve_command_on_shutdown(Command& cmd) {
    std::visit(overloaded{
                   [](SetSystemPromptCmd& c) { c.done->set_value(); },
                   [](GetHistoryCmd& c) { c.done->set_value(std::vector<Message>{}); },
                   [](ClearHistoryCmd& c) { c.done->set_value(); },
                   [](RefreshToolCallingCmd& c) { c.done->set_value(false); },
               },
               cmd);
}

void AgentRuntime::update_tool_calling() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    // Convert ToolMetadata → CoreToolInfo for the backend.
    auto metadata = tool_registry_.get_all_tool_metadata();
    std::vector<CoreToolInfo> tools;
    tools.reserve(metadata.size());
    for (const auto& tm : metadata) {
        tools.push_back(CoreToolInfo{
            tm.name,
            tm.description,
            tm.parameters_schema.dump(),
        });
    }

    auto done = std::make_shared<std::promise<bool>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(RefreshToolCallingCmd{std::move(tools), std::move(done)})) {
        return;
    }
    future.get();
}

} // namespace zoo::internal::agent
