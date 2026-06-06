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
 * is intentionally not thread-safe; callers that need concurrency should own
 * separate model instances or synchronize externally.
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
     * @brief Generates against an explicit conversation without mutating retained history.
     *
     * Installs @p messages as the working history, runs generation, then restores
     * the previous retained history unconditionally — even when generation fails.
     * Use this for one-shot completions that must not affect the ongoing session.
     */
    Expected<TextResponse> complete(ConversationView messages, GenerationOverride generation = {},
                                    TokenCallback on_token = {},
                                    CancellationCallback should_cancel = {});

    /**
     * @brief Generates schema-constrained JSON for a new user message.
     */
    Expected<ExtractionResponse> extract(const nlohmann::json& output_schema,
                                         std::string_view user_message,
                                         GenerationOverride generation = {},
                                         TokenCallback on_token = {},
                                         CancellationCallback should_cancel = {});

    /**
     * @brief Generates schema-constrained JSON for a structured inbound message.
     */
    Expected<ExtractionResponse> extract(const nlohmann::json& output_schema, MessageView message,
                                         GenerationOverride generation = {},
                                         TokenCallback on_token = {},
                                         CancellationCallback should_cancel = {});

    /**
     * @brief Generates schema-constrained JSON against explicit request-scoped history.
     */
    Expected<ExtractionResponse> extract(const nlohmann::json& output_schema,
                                         ConversationView messages,
                                         GenerationOverride generation = {},
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
     * @brief Generates from the current history and commits the assistant turn.
     *
     * When native tool calling is active, returned tool calls are also stored on
     * the appended assistant message so callers can add matching tool results
     * before a follow-up generation pass.
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

    /**
     * @brief Trims old non-system messages so the session stays within a length budget.
     *
     * Keeps up to @p max_non_system_messages of the most recent non-system messages.
     * Erasure always stops at a user-message boundary so the retained prefix forms a
     * valid exchange. A leading system message, if present, is never removed.
     *
     * @param max_non_system_messages Maximum number of non-system messages to retain.
     */
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
     * @brief Configures template-driven tool calling from model-facing tool metadata.
     *
     * Asks llama.cpp's `common_chat_templates` to prepare the model's native tool-call
     * format (29+ formats recognized). Passing an empty @p tools vector disables tool
     * calling and restores the default sampler chain.
     *
     * @param tools Tool specs exposed to the model. Must not contain handlers or executors;
     *        the registry is schema-only and execution is caller-owned.
     * @return `true` if native tool calling was enabled, `false` if the model's chat
     *         template does not support a recognized native format.
     */
    bool set_tool_calling(const std::vector<ToolSpec>& tools);

    /**
     * @brief Enables grammar-constrained schema output for future generations.
     *
     * Installs a GBNF grammar string into the sampler chain, constraining the next
     * generation pass to tokens that match the grammar. Used internally by `extract()`;
     * callers that build grammars manually can invoke this directly.
     *
     * @param grammar_str GBNF grammar string to install.
     * @return `true` if the grammar sampler was installed successfully, `false` otherwise.
     */
    bool set_schema_grammar(const std::string& grammar_str);

    /// Disables any active grammar/tool calling and restores the default sampler chain.
    void clear_tool_grammar();

    /// Returns `true` when native tool calling is currently active.
    [[nodiscard]] bool has_tool_calling() const noexcept;
    /// Returns `true` when a schema grammar sampler is currently installed.
    [[nodiscard]] bool has_schema_grammar() const noexcept;

    /**
     * @brief Parses generated text into structured content and tool calls.
     *
     * Uses the model's native parser to split raw assistant output into visible
     * content and structured tool call records. Returns the raw text unchanged
     * when no tool calling format is active.
     *
     * @param text Raw assistant-generated text to parse.
     * @return Parsed visible content and any extracted tool calls.
     */
    struct ParsedResponse {
        std::string content;                   ///< Visible content after stripping tool syntax.
        std::vector<OwnedToolCall> tool_calls; ///< Structured tool calls extracted from the text.
    };
    [[nodiscard]] ParsedResponse parse_tool_response(std::string_view text) const;

    /// Returns the llama.cpp chat format name currently active, or `"none"`.
    [[nodiscard]] const char* tool_calling_format_name() const noexcept;
    /// Returns the configured context window size in tokens.
    [[nodiscard]] int context_size() const noexcept;
    /// Returns the running token estimate for the retained history.
    [[nodiscard]] int estimated_tokens() const noexcept;
    /// Returns `true` when the estimated token count exceeds the context window.
    [[nodiscard]] bool is_context_exceeded() const noexcept;
    /// Returns the `ModelConfig` used to load this session.
    [[nodiscard]] const ModelConfig& model_config() const noexcept;
    /// Returns the default generation options applied when no override is supplied.
    [[nodiscard]] const GenerationOptions& default_generation_options() const noexcept;

  private:
    friend struct ModelTestAccess;

    explicit Model(ModelConfig model_config, GenerationOptions default_generation);

    std::unique_ptr<Impl> impl_;
};

} // namespace zoo::core
