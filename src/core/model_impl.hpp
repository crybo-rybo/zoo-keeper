/**
 * @file model_impl.hpp
 * @brief Private `zoo::core::Model` implementation state.
 */

#pragma once

#include "core/stream_filter.hpp"
#include "zoo/core/model.hpp"

#include <chat.h>
#include <common.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace zoo::core {

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
        std::string rendered_prompt;
        bool dirty = true;
    };

    enum class GrammarMode {
        None,
        NativeToolCall,
        Schema,
    };

    // Loaded model state: set once during initialize() and immutable thereafter.
    // Member declaration order matters for destruction: chat_templates is
    // destroyed before llama_model (templates were initialized from the model).
    struct LoadedModel {
        ModelConfig model_config;
        GenerationOptions default_generation_options;

        Model::LlamaModelHandle llama_model;
        Model::ChatTemplatesHandle chat_templates;
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
        Model::LlamaContextHandle ctx;
        Model::LlamaSamplerHandle sampler;

        std::string tool_grammar_str;
        GrammarMode grammar_mode = GrammarMode::None;
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

} // namespace zoo::core
