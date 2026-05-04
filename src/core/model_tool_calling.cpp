/**
 * @file model_tool_calling.cpp
 * @brief Template-driven tool calling: prepare, parse, and grammar management.
 */

#include "core/model_impl.hpp"
#include "log.hpp"
#include "zoo/core/model.hpp"

#include <chat.h>
#include <common.h>
#include <llama.h>

namespace zoo::core {

// ---------------------------------------------------------------------------
// set_tool_calling
// ---------------------------------------------------------------------------

bool Model::set_tool_calling(const std::vector<CoreToolInfo>& tools) {
    if (tools.empty()) {
        clear_tool_grammar();
        return true;
    }

    // Convert CoreToolInfo → common_chat_tool
    std::vector<common_chat_tool> chat_tools;
    chat_tools.reserve(tools.size());
    for (const auto& t : tools) {
        chat_tools.push_back({t.name, t.description, t.parameters_json});
    }

    // Build a minimal message set to probe the template for grammar/triggers.
    // We need at least one user message for the template to produce output.
    common_chat_templates_inputs inputs;
    inputs.messages = {{
        .role = "user",
        .content = "hello",
    }};
    inputs.tools = chat_tools;
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    common_chat_params params;
    try {
        params = common_chat_templates_apply(impl_->loaded_.chat_templates.get(), inputs);
    } catch (const std::exception&) {
        ZOO_LOG("info", "native tool calling not available for this model");
        return false;
    }

    // Content-only templates cannot emit structured native tool calls.
    if (params.format == COMMON_CHAT_FORMAT_CONTENT_ONLY) {
        ZOO_LOG("info", "native tool calling not available for this model (format: '%s')",
                common_chat_format_name(params.format));
        return false;
    }

    common_chat_parser_params parser_params;
    try {
        parser_params = make_tool_parser_params(params);
    } catch (const std::exception&) {
        return false;
    }

    auto state = std::make_unique<Impl::ToolCallingState>();
    state->tools = std::move(chat_tools);
    state->parser_params = std::move(parser_params);
    state->grammar = std::move(params.grammar);
    state->grammar_lazy = params.grammar_lazy;
    state->grammar_triggers = std::move(params.grammar_triggers);
    state->trigger_matcher = ToolCallTriggerMatcher(state->grammar_triggers);
    state->preserved_tokens = std::move(params.preserved_tokens);
    state->additional_stops = std::move(params.additional_stops);

    impl_->session_.tool_state = std::move(state);
    impl_->session_.sampler_policy =
        Impl::SamplerPolicy::native_tool_call(impl_->session_.tool_state->grammar);

    if (!impl_->session_.sampler_policy.grammar.empty()) {
        if (!rebuild_sampler_with_tool_grammar(*impl_)) {
            impl_->session_.tool_state.reset();
            impl_->session_.sampler_policy = Impl::SamplerPolicy::plain();
            return false;
        }
    } else {
        impl_->session_.sampler = create_sampler_chain(*impl_);
        if (!impl_->session_.sampler) {
            impl_->session_.tool_state.reset();
            impl_->session_.sampler_policy = Impl::SamplerPolicy::plain();
            return false;
        }
    }

    ZOO_LOG("info", "native tool calling enabled: format '%s' (%zu tools registered)",
            common_chat_format_name(impl_->session_.tool_state->parser_params.format),
            impl_->session_.tool_state->tools.size());
    return true;
}

// ---------------------------------------------------------------------------
// parse_tool_response
// ---------------------------------------------------------------------------

Model::ParsedResponse Model::parse_tool_response(std::string_view text) const {
    ParsedResponse result;

    if (!impl_->session_.tool_state) {
        result.content = std::string(text);
        return result;
    }

    common_chat_msg parsed;
    try {
        parsed =
            common_chat_parse(std::string(text), false, impl_->session_.tool_state->parser_params);
    } catch (const std::exception&) {
        result.content = std::string(text);
        return result;
    }

    result.content = std::move(parsed.content);
    result.tool_calls.reserve(parsed.tool_calls.size());
    for (auto& tc : parsed.tool_calls) {
        result.tool_calls.push_back(ToolCallInfo{
            std::move(tc.id),
            std::move(tc.name),
            std::move(tc.arguments),
        });
    }

    return result;
}

// ---------------------------------------------------------------------------
// clear_tool_grammar
// ---------------------------------------------------------------------------

void Model::clear_tool_grammar() noexcept {
    if (impl_->session_.sampler_policy.mode == Impl::SamplerPolicy::Mode::Plain) {
        return;
    }

    impl_->session_.tool_state.reset();
    impl_->session_.sampler_policy = Impl::SamplerPolicy::plain();
    impl_->session_.sampler = create_sampler_chain(*impl_);
}

// ---------------------------------------------------------------------------
// tool_calling_format_name
// ---------------------------------------------------------------------------

const char* Model::tool_calling_format_name() const noexcept {
    if (!impl_->session_.tool_state) {
        return "none";
    }
    return common_chat_format_name(impl_->session_.tool_state->parser_params.format);
}

} // namespace zoo::core
