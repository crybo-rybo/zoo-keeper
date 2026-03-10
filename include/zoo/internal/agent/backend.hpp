/**
 * @file backend.hpp
 * @brief Internal interface for model operations used by the agent runtime.
 *
 * This seam exists so agent orchestration can be unit-tested with a fake
 * backend that returns scripted generations.  It is intentionally internal
 * to the agent layer and must not be exposed in the public API.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>
#include <zoo/core/types.hpp>

namespace zoo::internal::agent {

/**
 * @brief Result of a single generation pass started from existing history.
 *
 * Mirrors `core::Model::GenerationResult` without depending on model.hpp
 * so that test fakes avoid pulling in llama.cpp forward declarations.
 */
struct GenerationResult {
    std::string text;      ///< Raw generated text for the pass.
    int prompt_tokens = 0; ///< Number of prompt tokens rendered for the pass.
    bool tool_call_detected =
        false; ///< Whether sentinel-based grammar mode emitted a tool call.
};

/**
 * @brief Minimal model surface consumed by the agent runtime.
 *
 * Production code wraps `core::Model`; unit tests substitute a fake that
 * returns scripted generations and records history mutations.
 */
class AgentBackend {
  public:
    virtual ~AgentBackend() = default;

    virtual Expected<void> add_message(const Message& message) = 0;
    virtual Expected<GenerationResult>
    generate_from_history(std::optional<TokenCallback> on_token,
                          CancellationCallback should_cancel) = 0;
    virtual void finalize_response() = 0;

    virtual void set_system_prompt(const std::string& prompt) = 0;
    virtual std::vector<Message> get_history() const = 0;
    virtual void clear_history() = 0;

    virtual bool set_tool_grammar(const std::string& grammar_str) = 0;
    virtual void clear_tool_grammar() = 0;
};

} // namespace zoo::internal::agent
