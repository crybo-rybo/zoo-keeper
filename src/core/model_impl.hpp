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
        common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
        std::string grammar;
        bool grammar_lazy = false;
        std::vector<common_grammar_trigger> grammar_triggers;
        ToolCallTriggerMatcher trigger_matcher;
        std::vector<std::string> preserved_tokens;
        std::vector<std::string> additional_stops;
        bool thinking_forced_open = false;
        common_peg_arena parser;
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

    explicit Impl(ModelConfig model_config, GenerationOptions default_generation)
        : model_config_(std::move(model_config)),
          default_generation_options_(std::move(default_generation)),
          active_sampling_(default_generation_options_.sampling) {}

    ModelConfig model_config_;
    GenerationOptions default_generation_options_;

    LlamaModelHandle llama_model_;
    LlamaContextHandle ctx_;
    LlamaSamplerHandle sampler_;
    const llama_vocab* vocab_ = nullptr;

    int context_size_ = 0;
    ChatTemplatesHandle chat_templates_;

    std::string tool_grammar_str_;
    GrammarMode grammar_mode_ = GrammarMode::None;
    std::unique_ptr<ToolCallingState> tool_state_;

    PromptState prompt_state_;

    std::vector<Message> messages_;
    int estimated_tokens_ = 0;
    std::vector<int> token_buffer_;
    SamplingParams active_sampling_;
    static constexpr int kTemplateOverheadPerMessage = 8;
};

} // namespace zoo::core
