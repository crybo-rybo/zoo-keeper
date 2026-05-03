/**
 * @file runtime_commands.cpp
 * @brief Synchronous command lane for the internal agent runtime.
 */

#include "agent/runtime.hpp"

#include "log.hpp"
#include <cassert>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace zoo::internal::agent {

namespace {

Error command_timeout_error(std::string_view command_name) {
    return Error{ErrorCode::RequestTimeout,
                 "Timed out waiting for command to complete: " + std::string(command_name)};
}

template <typename CmdT> auto make_string_cmd(std::string s) {
    return [s = std::move(s)](auto done) mutable -> Command {
        return CmdT{std::move(s), std::move(done)};
    };
}

} // namespace

template <typename Result, typename Maker>
Expected<Result> AgentRuntime::send_sync_command(Maker&& make_cmd,
                                                 std::optional<std::chrono::nanoseconds> timeout,
                                                 std::string_view name) {
    if (!running_.load(std::memory_order_acquire)) {
        return std::unexpected(Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    }

    auto done = std::make_shared<std::promise<Expected<Result>>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(std::forward<Maker>(make_cmd)(std::move(done)))) {
        return std::unexpected(Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    }
    if (timeout && future.wait_for(*timeout) != std::future_status::ready) {
        return std::unexpected(command_timeout_error(name));
    }
    return future.get();
}

Expected<void>
AgentRuntime::set_system_prompt_impl(std::string prompt,
                                     std::optional<std::chrono::nanoseconds> timeout) {
    return send_sync_command<void>(make_string_cmd<SetSystemPromptCmd>(std::move(prompt)), timeout,
                                   "set_system_prompt");
}

void AgentRuntime::set_system_prompt(std::string_view prompt) {
    (void)set_system_prompt_impl(std::string(prompt), std::nullopt);
}

Expected<void> AgentRuntime::set_system_prompt(std::string_view prompt,
                                               std::chrono::nanoseconds timeout) {
    return set_system_prompt_impl(std::string(prompt), timeout);
}

Expected<void>
AgentRuntime::add_system_message_impl(std::string message,
                                      std::optional<std::chrono::nanoseconds> timeout) {
    return send_sync_command<void>(make_string_cmd<AddSystemMessageCmd>(std::move(message)),
                                   timeout, "add_system_message");
}

Expected<void> AgentRuntime::add_system_message(std::string_view message) {
    return add_system_message_impl(std::string(message), std::nullopt);
}

Expected<void> AgentRuntime::add_system_message(std::string_view message,
                                                std::chrono::nanoseconds timeout) {
    return add_system_message_impl(std::string(message), timeout);
}

Expected<HistorySnapshot>
AgentRuntime::get_history_impl(std::optional<std::chrono::nanoseconds> timeout) const {
    return const_cast<AgentRuntime*>(this)->send_sync_command<HistorySnapshot>(
        [](auto done) -> Command { return GetHistoryCmd{std::move(done)}; }, timeout,
        "get_history");
}

HistorySnapshot AgentRuntime::get_history() const {
    return get_history_impl(std::nullopt).value_or(HistorySnapshot{});
}

Expected<HistorySnapshot> AgentRuntime::get_history(std::chrono::nanoseconds timeout) const {
    return get_history_impl(timeout);
}

Expected<void> AgentRuntime::clear_history_impl(std::optional<std::chrono::nanoseconds> timeout) {
    return send_sync_command<void>(
        [](auto done) -> Command { return ClearHistoryCmd{std::move(done)}; }, timeout,
        "clear_history");
}

void AgentRuntime::clear_history() {
    (void)clear_history_impl(std::nullopt);
}

Expected<void> AgentRuntime::clear_history(std::chrono::nanoseconds timeout) {
    return clear_history_impl(timeout);
}

Expected<void> AgentRuntime::register_tool_impl(tools::ToolDefinition definition,
                                                std::optional<std::chrono::nanoseconds> timeout) {
    assert(!inference_thread_.joinable() ||
           std::this_thread::get_id() != inference_thread_.get_id());
    return send_sync_command<void>(
        [d = std::move(definition)](auto done) mutable -> Command {
            return RegisterToolCmd{std::move(d), std::move(done)};
        },
        timeout, "register_tool");
}

Expected<void> AgentRuntime::register_tool(tools::ToolDefinition definition) {
    return register_tool_impl(std::move(definition), std::nullopt);
}

Expected<void> AgentRuntime::register_tool(tools::ToolDefinition definition,
                                           std::chrono::nanoseconds timeout) {
    return register_tool_impl(std::move(definition), timeout);
}

Expected<void> AgentRuntime::register_tools_impl(std::vector<tools::ToolDefinition> definitions,
                                                 std::optional<std::chrono::nanoseconds> timeout) {
    assert(!inference_thread_.joinable() ||
           std::this_thread::get_id() != inference_thread_.get_id());
    if (definitions.empty()) {
        return {};
    }
    return send_sync_command<void>(
        [d = std::move(definitions)](auto done) mutable -> Command {
            return RegisterToolsCmd{std::move(d), std::move(done)};
        },
        timeout, "register_tools");
}

Expected<void> AgentRuntime::register_tools(std::vector<tools::ToolDefinition> definitions) {
    return register_tools_impl(std::move(definitions), std::nullopt);
}

Expected<void> AgentRuntime::register_tools(std::vector<tools::ToolDefinition> definitions,
                                            std::chrono::nanoseconds timeout) {
    return register_tools_impl(std::move(definitions), timeout);
}

size_t AgentRuntime::tool_count() const noexcept {
    return tool_registry_.size();
}

void AgentRuntime::handle_command(Command& cmd) {
    std::visit(
        overloaded{
            [this](SetSystemPromptCmd& c) {
                backend_->set_system_prompt(c.prompt);
                c.done->set_value(Expected<void>{});
            },
            [this](GetHistoryCmd& c) {
                c.done->set_value(Expected<HistorySnapshot>{backend_->get_history()});
            },
            [this](ClearHistoryCmd& c) {
                backend_->clear_history();
                c.done->set_value(Expected<void>{});
            },
            [this](AddSystemMessageCmd& c) {
                c.done->set_value(backend_->add_message(MessageView{Role::System, c.message}));
            },
            [this](RegisterToolCmd& c) {
                if (auto result = tool_registry_.register_tool(std::move(c.definition)); !result) {
                    c.done->set_value(std::unexpected(result.error()));
                    return;
                }

                refresh_tool_calling_state();
                c.done->set_value({});
            },
            [this](RegisterToolsCmd& c) {
                if (auto result = tool_registry_.register_tools(std::move(c.definitions));
                    !result) {
                    c.done->set_value(std::unexpected(result.error()));
                    return;
                }

                refresh_tool_calling_state();
                c.done->set_value({});
            },
        },
        cmd);
}

void AgentRuntime::resolve_command_on_shutdown(Command& cmd) {
    auto shutdown_error = []() {
        return std::unexpected(Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    };
    std::visit(overloaded{
                   [&](SetSystemPromptCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](GetHistoryCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](ClearHistoryCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](AddSystemMessageCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](RegisterToolCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](RegisterToolsCmd& c) { c.done->set_value(shutdown_error()); },
               },
               cmd);
}

bool AgentRuntime::refresh_tool_calling_state() {
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

    bool active = false;
    if (tools.empty()) {
        backend_->clear_tool_grammar();
    } else if (backend_->set_tool_calling(tools)) {
        active = true;
        ZOO_LOG("info", "tool calling configured (%zu tools, format=%s)", tools.size(),
                backend_->tool_calling_format_name());
    } else {
        backend_->clear_tool_grammar();
        ZOO_LOG("warn", "tool calling setup failed, falling back to unconstrained generation");
    }
    tool_grammar_active_.store(active, std::memory_order_release);
    return active;
}

} // namespace zoo::internal::agent
