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
    std::string text;                ///< Raw generated text for the pass.
    int prompt_tokens = 0;           ///< Number of prompt tokens rendered for the pass.
    bool tool_call_detected = false; ///< Whether tool calling detected a tool call in the output.
};

/**
 * @brief Parsed tool response from model output.
 *
 * Mirrors `core::Model::ParsedResponse` for the backend interface.
 */
struct ParsedToolResponse {
    std::string content;
    std::vector<ToolCallInfo> tool_calls;
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

    /**
     * @brief Replaces the retained message history without flushing the KV cache.
     */
    virtual void replace_messages(std::vector<Message> messages) = 0;

    /**
     * @brief Configures template-driven tool calling.
     *
     * @param tools Tool descriptions for the model's chat template.
     * @return `true` when tool calling was set up successfully.
     */
    virtual bool set_tool_calling(const std::vector<CoreToolInfo>& tools) = 0;

    /**
     * @brief Enables immediate grammar-constrained generation for schema output.
     *
     * @param grammar_str GBNF grammar string rooted at `root`.
     * @return `true` when the sampler chain was rebuilt successfully.
     */
    virtual bool set_schema_grammar(const std::string& grammar_str) = 0;

    /// Disables any active grammar/tool calling and restores the default sampler chain.
    virtual void clear_tool_grammar() = 0;

    /**
     * @brief Parses a generated text into structured content and tool calls.
     */
    virtual ParsedToolResponse parse_tool_response(const std::string& text) const = 0;

    /// Returns the name of the detected tool calling format.
    virtual const char* tool_calling_format_name() const noexcept = 0;
};

} // namespace zoo::internal::agent
