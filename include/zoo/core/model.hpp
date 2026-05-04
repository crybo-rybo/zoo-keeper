/**
 * @file model.hpp
 * @brief Public wrapper around llama.cpp model loading, prompting, and generation.
 */

#pragma once

#include "types.hpp"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace zoo::core {

struct ModelTestAccess;

/**
 * @brief Direct llama.cpp wrapper for model lifecycle, history, and generation.
 *
 * `Model` owns the backend model state and incremental chat-template state. It
 * can be used standalone, but it is intentionally not thread-safe; callers
 * that need concurrency should use `zoo::Agent`.
 */
class Model {
  public:
    struct Impl;

    /**
     * @brief Loads and initializes a model from the supplied configuration.
     *
     * @param model_config Runtime model configuration to validate and apply.
     * @param default_generation Default generation policy used when a call does
     *        not override it explicitly.
     * @return A fully initialized model, or an error if validation or backend
     *         setup fails.
     */
    static Expected<std::unique_ptr<Model>>
    load(const ModelConfig& model_config,
         const GenerationOptions& default_generation = GenerationOptions{});

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    /**
     * @brief Generates an assistant response for a new user message.
     */
    Expected<TextResponse> generate(std::string_view user_message,
                                    GenerationOverride generation = {}, TokenCallback on_token = {},
                                    CancellationCallback should_cancel = {});

    /**
     * @brief Generates an assistant response for a structured inbound message.
     */
    Expected<TextResponse> generate(MessageView message, GenerationOverride generation = {},
                                    TokenCallback on_token = {},
                                    CancellationCallback should_cancel = {});

    /**
     * @brief Result of a low-level generation pass started from existing history.
     */
    struct GenerationResult {
        std::string text;      ///< Raw generated text for the pass.
        int prompt_tokens = 0; ///< Number of prompt tokens rendered for the pass.
        bool tool_call_detected =
            false;                  ///< Whether tool calling detected a tool call in the output.
        std::string parsed_content; ///< Visible content after stripping tool syntax.
        std::vector<OwnedToolCall> tool_calls; ///< Structured tool calls extracted from the output.
    };

    /**
     * @brief Generates from the current history without appending a new user message.
     *
     * @note This method does not commit the assistant turn to history.
     */
    Expected<GenerationResult> generate_from_history(GenerationOverride generation = {},
                                                     TokenCallback on_token = {},
                                                     CancellationCallback should_cancel = {});

    /**
     * @brief Advances the incremental chat-template checkpoint to the current history.
     */
    void finalize_response();

    /**
     * @brief Sets or replaces the leading system prompt in the tracked history.
     */
    void set_system_prompt(std::string_view prompt);

    /**
     * @brief Appends a message to retained history after validating role sequencing.
     */
    Expected<void> add_message(MessageView message);

    /// Returns an owning snapshot of the current conversation history.
    [[nodiscard]] HistorySnapshot get_history() const;

    /// Clears conversation history, token estimates, and cached KV state.
    void clear_history();

    void trim_history(size_t max_non_system_messages);

    /**
     * @brief Replaces the retained message history without flushing the KV cache.
     */
    void replace_history(HistorySnapshot snapshot);

    /**
     * @brief Atomically swaps retained history with a provided snapshot.
     */
    [[nodiscard]] HistorySnapshot swap_history(HistorySnapshot snapshot);

    /**
     * @brief Configures template-driven tool calling from registered tool metadata.
     */
    bool set_tool_calling(const std::vector<CoreToolInfo>& tools);

    /**
     * @brief Enables grammar-constrained schema output for future generations.
     */
    bool set_schema_grammar(const std::string& grammar_str);

    /// Disables any active grammar/tool calling and restores the default sampler chain.
    void clear_tool_grammar() noexcept;

    [[nodiscard]] bool has_tool_calling() const noexcept;
    [[nodiscard]] bool has_schema_grammar() const noexcept;

    /**
     * @brief Parses a generated text into structured content and tool calls.
     */
    struct ParsedResponse {
        std::string content;
        std::vector<OwnedToolCall> tool_calls;
    };
    [[nodiscard]] ParsedResponse parse_tool_response(std::string_view text) const;

    [[nodiscard]] const char* tool_calling_format_name() const noexcept;
    [[nodiscard]] int context_size() const noexcept;
    [[nodiscard]] int estimated_tokens() const noexcept;
    [[nodiscard]] bool is_context_exceeded() const noexcept;
    [[nodiscard]] const ModelConfig& model_config() const noexcept;
    [[nodiscard]] const GenerationOptions& default_generation_options() const noexcept;

  private:
    friend struct ModelTestAccess;

    explicit Model(ModelConfig model_config, GenerationOptions default_generation);

    std::unique_ptr<Impl> impl_;
};

} // namespace zoo::core
