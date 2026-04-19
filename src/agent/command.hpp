/**
 * @file command.hpp
 * @brief Control commands routed through the agent runtime mailbox.
 *
 * Each command carries a typed promise so the calling thread can block until
 * the inference thread has executed the operation.
 */

#pragma once

#include <future>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <zoo/core/types.hpp>
#include <zoo/tools/types.hpp>

namespace zoo::internal::agent {

/// Replaces the system prompt on the underlying model.
struct SetSystemPromptCmd {
    std::string prompt;
    std::shared_ptr<std::promise<void>> done;
};

/// Snapshots the current conversation history.
struct GetHistoryCmd {
    std::shared_ptr<std::promise<HistorySnapshot>> done;
};

/// Clears the conversation history and KV cache.
struct ClearHistoryCmd {
    std::shared_ptr<std::promise<void>> done;
};

/// Appends a system-role message to the conversation without replacing the initial system prompt.
struct AddSystemMessageCmd {
    std::string message;
    std::shared_ptr<std::promise<Expected<void>>> done;
};

/// Registers a single tool on the inference thread.
struct RegisterToolCmd {
    tools::ToolDefinition definition;
    std::shared_ptr<std::promise<Expected<void>>> done;
};

/// Registers a batch of tools on the inference thread.
struct RegisterToolsCmd {
    std::vector<tools::ToolDefinition> definitions;
    std::shared_ptr<std::promise<Expected<void>>> done;
};

/// Discriminated union of all control commands the runtime accepts.
using Command = std::variant<SetSystemPromptCmd, GetHistoryCmd, ClearHistoryCmd,
                             AddSystemMessageCmd, RegisterToolCmd, RegisterToolsCmd>;

/// Helper for exhaustive std::visit with overloaded lambdas.
template <class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};

} // namespace zoo::internal::agent
