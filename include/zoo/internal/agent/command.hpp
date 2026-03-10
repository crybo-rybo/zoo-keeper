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
    std::shared_ptr<std::promise<std::vector<Message>>> done;
};

/// Clears the conversation history and KV cache.
struct ClearHistoryCmd {
    std::shared_ptr<std::promise<void>> done;
};

/// Rebuilds grammar constraints from updated tool metadata.
struct RefreshToolGrammarCmd {
    std::vector<tools::ToolMetadata> metadata;
    std::shared_ptr<std::promise<bool>> done;
};

/// Discriminated union of all control commands the runtime accepts.
using Command =
    std::variant<SetSystemPromptCmd, GetHistoryCmd, ClearHistoryCmd, RefreshToolGrammarCmd>;

/// Helper for exhaustive std::visit with overloaded lambdas.
template <class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};

} // namespace zoo::internal::agent
