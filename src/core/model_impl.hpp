/**
 * @file model_impl.hpp
 * @brief Private `zoo::core::Model` implementation state.
 */

#pragma once

#include "core/stream_filter.hpp"
#include "zoo/core/model.hpp"

#include <chat.h>
#include <common.h>
#include <llama.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace zoo::core {

struct LlamaModelDeleter {
    void operator()(llama_model* model) const noexcept;
};
struct LlamaContextDeleter {
    void operator()(llama_context* context) const noexcept;
};
struct LlamaSamplerDeleter {
    void operator()(llama_sampler* sampler) const noexcept;
};
struct ChatTemplatesDeleter {
    void operator()(common_chat_templates* tmpls) const noexcept;
};

using LlamaModelHandle = std::unique_ptr<llama_model, LlamaModelDeleter>;
using LlamaContextHandle = std::unique_ptr<llama_context, LlamaContextDeleter>;
using LlamaSamplerHandle = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;
using ChatTemplatesHandle = std::unique_ptr<common_chat_templates, ChatTemplatesDeleter>;

struct Model::Impl {
    struct ToolCallingState {
        std::vector<common_chat_tool> tools;
        common_chat_parser_params parser_params;
        std::string grammar;
        bool grammar_lazy = false;
        std::vector<common_grammar_trigger> grammar_triggers;
        ToolCallTriggerMatcher trigger_matcher;
        std::vector<std::string> preserved_tokens;
        std::vector<std::string> additional_stops;
    };

    struct PromptState {
        int committed_prompt_len = 0;
        bool dirty = true;
    };

    struct SamplerPolicy {
        enum class Mode {
            Plain,
            NativeToolCall,
            Schema,
        };

        Mode mode = Mode::Plain;
        std::string grammar;

        [[nodiscard]] bool is_native_tool_call() const noexcept {
            return mode == Mode::NativeToolCall;
        }

        [[nodiscard]] bool is_schema() const noexcept {
            return mode == Mode::Schema;
        }

        static SamplerPolicy plain() {
            return {};
        }

        static SamplerPolicy native_tool_call(std::string grammar) {
            return {Mode::NativeToolCall, std::move(grammar)};
        }

        static SamplerPolicy schema(std::string grammar) {
            return {Mode::Schema, std::move(grammar)};
        }

        Expected<void> ensure_sampler_for_pass(Model::Impl& impl) const;
    };

    // Loaded model state: set once during initialize() and immutable thereafter.
    // Member declaration order matters for destruction: chat_templates is
    // destroyed before llama_model (templates were initialized from the model).
    struct LoadedModel {
        ModelConfig model_config;
        GenerationOptions default_generation_options;

        LlamaModelHandle llama_model;
        ChatTemplatesHandle chat_templates;
        const llama_vocab* vocab = nullptr;
        int context_size = 0;

        LoadedModel(ModelConfig cfg, GenerationOptions defaults)
            : model_config(std::move(cfg)), default_generation_options(std::move(defaults)) {}
    };

    // Per-conversation state: mutates during chat/extraction.
    // Member declaration order matters for destruction: sampler is destroyed
    // before ctx (sampler chain may reference vocab through grammar samplers
    // and is built from the ctx).
    struct Session {
        LlamaContextHandle ctx;
        LlamaSamplerHandle sampler;

        SamplerPolicy sampler_policy = SamplerPolicy::plain();
        std::unique_ptr<ToolCallingState> tool_state;

        PromptState prompt_state;

        std::vector<Message> messages;
        int estimated_tokens = 0;
        std::vector<int> token_buffer;
        SamplingParams active_sampling;

        explicit Session(SamplingParams initial_sampling)
            : active_sampling(std::move(initial_sampling)) {}
    };

    explicit Impl(ModelConfig model_config, GenerationOptions default_generation)
        : loaded_(std::move(model_config), std::move(default_generation)),
          session_(loaded_.default_generation_options.sampling) {}

    // Destruction order: session_ first (samplers/ctx), then loaded_ (chat
    // templates, then llama_model). This invariant must hold — see comments
    // on each struct above.
    LoadedModel loaded_;
    Session session_;

    static constexpr int kTemplateOverheadPerMessage = 8;
};

// The conversion constructor copies metadata fields (format, generation_prompt, …)
// from `params`, but the PEG parser itself must be re-loaded from `params.parser`'s
// serialized form before it is usable.
inline common_chat_parser_params make_tool_parser_params(const common_chat_params& params) {
    common_chat_parser_params parser_params(params);
    parser_params.parse_tool_calls = true;
    if (!params.parser.empty()) {
        parser_params.parser.load(params.parser);
    }
    return parser_params;
}

void initialize_model_backend();
[[nodiscard]] Expected<void> initialize_model(Model::Impl& impl);
[[nodiscard]] Expected<std::vector<int>> tokenize(Model::Impl& impl, std::string_view text);
[[nodiscard]] Expected<std::string>
run_inference(Model::Impl& impl, const std::vector<int>& prompt_tokens, int max_tokens,
              const std::vector<std::string>& stop_sequences, TokenCallback on_token = {},
              CancellationCallback should_cancel = {});
[[nodiscard]] Expected<std::string> render_prompt_delta(Model::Impl& impl);
void clear_kv_cache(Model::Impl& impl);
void note_history_append(Model::Impl& impl) noexcept;
void note_history_rewrite(Model::Impl& impl) noexcept;
void note_history_reset(Model::Impl& impl) noexcept;
[[nodiscard]] LlamaSamplerHandle create_sampler_chain(Model::Impl& impl);
bool rebuild_sampler_with_tool_grammar(Model::Impl& impl);
bool rebuild_sampler_with_schema_grammar(Model::Impl& impl);
[[nodiscard]] Expected<void> ensure_grammar_sampler_for_pass(Model::Impl& impl);
[[nodiscard]] std::vector<std::string> merge_stop_sequences(const Model::Impl& impl,
                                                            std::vector<std::string> base);
[[nodiscard]] int estimate_tokens(const Model::Impl& impl, std::string_view text);
[[nodiscard]] int estimate_message_tokens(const Model::Impl& impl, const Message& message);
void trim_history_to_fit(Model::Impl& impl);
void rollback_last_message(Model::Impl& impl) noexcept;
[[nodiscard]] GenerationOptions resolve_generation_options(const Model::Impl& impl,
                                                           GenerationOverride generation);

} // namespace zoo::core
