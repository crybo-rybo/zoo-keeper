/**
 * @file model_inference.cpp
 * @brief Inference and response assembly for `zoo::core::Model`.
 */

#include "core/model_impl.hpp"
#include "zoo/core/model.hpp"

#include "core/batch.hpp"
#include "core/stream_filter.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <llama.h>
#include <span>
#include <string_view>

namespace zoo::core {

namespace {

struct DecodedToken {
    llama_token token = 0;
    std::string piece;
    bool end_of_generation = false;
};

struct InferenceCtx {
    llama_context* ctx;
    llama_sampler* sampler;
    const llama_vocab* vocab;
    int context_size;
};

struct InferencePhase {
    InferenceCtx phase_ctx;
    const CancellationCallback& should_cancel;

    [[nodiscard]] Expected<int> prefill(const std::vector<int>& prompt_tokens) const {
        const int n_batch = static_cast<int>(llama_n_batch(phase_ctx.ctx));
        const int base_pos = llama_memory_seq_pos_max(llama_get_memory(phase_ctx.ctx), 0) + 1;

        auto chunks = compute_prefill_chunks(static_cast<int>(prompt_tokens.size()), n_batch);
        for (const auto& chunk : chunks) {
            if (should_cancel && should_cancel()) {
                return std::unexpected(
                    Error{ErrorCode::RequestCancelled, "Request cancelled during prompt prefill"});
            }

            int n_ctx_used = base_pos + chunk.offset + chunk.count;
            if (n_ctx_used > phase_ctx.context_size) {
                return std::unexpected(
                    Error{ErrorCode::ContextWindowExceeded, "Prompt tokens exceed context size"});
            }

            LlamaBatchHandle batch(chunk.count, 0, 1);
            auto& raw_batch = batch.get();
            for (int i = 0; i < chunk.count; ++i) {
                raw_batch.token[i] = static_cast<llama_token>(prompt_tokens[chunk.offset + i]);
                raw_batch.pos[i] = static_cast<llama_pos>(base_pos + chunk.offset + i);
                raw_batch.n_seq_id[i] = 1;
                raw_batch.seq_id[i][0] = 0;
                raw_batch.logits[i] = (chunk.emit_logits && i == chunk.count - 1);
            }
            raw_batch.n_tokens = chunk.count;

            int rc = llama_decode(phase_ctx.ctx, raw_batch);
            if (rc != 0) {
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "Failed to decode prefill batch"});
            }
        }

        return base_pos + static_cast<int>(prompt_tokens.size());
    }

    [[nodiscard]] Expected<DecodedToken> decode() const {
        if (should_cancel && should_cancel()) {
            return std::unexpected(
                Error{ErrorCode::RequestCancelled, "Request cancelled during generation"});
        }

        const llama_token token = llama_sampler_sample(phase_ctx.sampler, phase_ctx.ctx, -1);
        if (llama_vocab_is_eog(phase_ctx.vocab, token)) {
            return DecodedToken{token, {}, true};
        }

        char buffer[256];
        const int bytes =
            llama_token_to_piece(phase_ctx.vocab, token, buffer, sizeof(buffer), 0, true);
        if (bytes < 0) {
            return std::unexpected(Error{ErrorCode::Unknown, "Failed to convert token"});
        }

        return DecodedToken{token, std::string(buffer, static_cast<size_t>(bytes)), false};
    }

    [[nodiscard]] Expected<void> finalize(llama_batch& batch, llama_token token,
                                          int& current_pos) const {
        batch.token[0] = token;
        batch.pos[0] = static_cast<llama_pos>(current_pos);
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;

        int rc = llama_decode(phase_ctx.ctx, batch);
        if (rc != 0) {
            return std::unexpected(Error{ErrorCode::InferenceFailed, "Failed to decode token"});
        }
        ++current_pos;
        return {};
    }
};

Expected<TokenAction> invoke_token_callback(const TokenCallback& callback, std::string_view token) {
    try {
        return callback(token);
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::InferenceFailed, "Token callback threw an exception", e.what()});
    } catch (...) {
        return std::unexpected(
            Error{ErrorCode::InferenceFailed, "Token callback threw an unknown exception"});
    }
}

} // namespace

Expected<std::string> run_inference(Model::Impl& impl, const std::vector<int>& prompt_tokens,
                                    int max_tokens, const std::vector<std::string>& stop_sequences,
                                    TokenCallback on_token, CancellationCallback should_cancel) {
    std::string generated_text;
    const int effective_max = (max_tokens > 0) ? max_tokens : impl.loaded_.context_size;
    generated_text.reserve(std::min(static_cast<size_t>(effective_max) * 8, size_t{65536}));
    int token_count = 0;
    bool stopped_by_callback = false;
    StopSequenceMatcher stop_matcher{std::span<const std::string>(stop_sequences)};
    StreamFilter stream_filter;
    if (on_token && impl.session_.tool_state &&
        impl.session_.sampler_policy.is_native_tool_call()) {
        const auto& trigger_matcher = impl.session_.tool_state->trigger_matcher;
        stream_filter = StreamFilter(std::span<const std::string>(trigger_matcher.word_triggers()),
                                     &trigger_matcher);
    }

    InferencePhase phase{InferenceCtx{impl.session_.ctx.get(), impl.session_.sampler.get(),
                                      impl.loaded_.vocab, impl.loaded_.context_size},
                         should_cancel};
    auto current_pos_result = phase.prefill(prompt_tokens);
    if (!current_pos_result) {
        return std::unexpected(current_pos_result.error());
    }

    int current_pos = *current_pos_result;
    LlamaBatchHandle ar_batch(1, 0, 1);

    while (true) {
        auto decoded = phase.decode();
        if (!decoded) {
            return std::unexpected(decoded.error());
        }
        if (decoded->end_of_generation) {
            break;
        }

        generated_text.append(decoded->piece);
        ++token_count;

        if (!stop_sequences.empty()) {
            const size_t match_len = stop_matcher.match_suffix(generated_text);
            if (match_len > 0) {
                generated_text.resize(generated_text.size() - match_len);
                break;
            }
        }

        if (on_token && !stream_filter.suppressing()) {
            std::string visible_chunk = stream_filter.consume(decoded->piece, generated_text);
            if (!visible_chunk.empty()) {
                auto action = invoke_token_callback(on_token, visible_chunk);
                if (!action) {
                    return std::unexpected(action.error());
                }
                if (*action == TokenAction::Stop) {
                    stopped_by_callback = true;
                    break;
                }
            }
        }

        if (token_count >= effective_max || current_pos >= impl.loaded_.context_size) {
            break;
        }

        if (auto result = phase.finalize(ar_batch.get(), decoded->token, current_pos); !result) {
            return std::unexpected(result.error());
        }
    }

    if (on_token && !stream_filter.suppressing() && !stopped_by_callback) {
        std::string trailing = stream_filter.finalize();
        if (!trailing.empty()) {
            auto action = invoke_token_callback(on_token, trailing);
            if (!action) {
                return std::unexpected(action.error());
            }
        }
    }

    return generated_text;
}

Expected<TextResponse> Model::generate(std::string_view user_message, GenerationOverride generation,
                                       TokenCallback on_token, CancellationCallback should_cancel) {
    return generate(MessageView{Role::User, user_message}, generation, on_token, should_cancel);
}

Expected<TextResponse> Model::generate(MessageView message, GenerationOverride generation,
                                       TokenCallback on_token, CancellationCallback should_cancel) {
    auto start_time = std::chrono::steady_clock::now();
    auto effective_options = resolve_generation_options(*impl_, generation);
    if (auto validation = effective_options.validate(); !validation) {
        return std::unexpected(validation.error());
    }

    auto add_result = add_message(message);
    if (!add_result) {
        return std::unexpected(add_result.error());
    }

    std::chrono::steady_clock::time_point first_token_time;
    bool first_token_received = false;
    int completion_tokens = 0;

    impl_->session_.active_sampling = effective_options.sampling;

    auto wrapped_callback = [&](std::string_view token) -> TokenAction {
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        if (on_token) {
            return on_token(token);
        }
        return TokenAction::Continue;
    };

    auto prompt_result = render_prompt_delta(*impl_);
    if (!prompt_result) {
        rollback_last_message(*impl_);
        return std::unexpected(prompt_result.error());
    }

    if (auto rebuild = ensure_grammar_sampler_for_pass(*impl_); !rebuild) {
        rollback_last_message(*impl_);
        return std::unexpected(rebuild.error());
    }

    auto tokens_result = tokenize(*impl_, *prompt_result);
    if (!tokens_result) {
        rollback_last_message(*impl_);
        return std::unexpected(tokens_result.error());
    }

    const int prompt_tokens = static_cast<int>(tokens_result->size());

    auto all_stops = merge_stop_sequences(*impl_, effective_options.stop_sequences);

    auto generate_result = run_inference(*impl_, *tokens_result, effective_options.max_tokens,
                                         all_stops, TokenCallback(wrapped_callback), should_cancel);

    if (!generate_result) {
        rollback_last_message(*impl_);
        return std::unexpected(generate_result.error());
    }

    std::string generated_text = std::move(*generate_result);

    // When native tool calling is active, parse the output to extract
    // structured tool calls for proper history round-tripping.
    if (impl_->session_.sampler_policy.is_native_tool_call() && impl_->session_.tool_state) {
        auto parsed = parse_tool_response(generated_text);
        if (!parsed.tool_calls.empty()) {
            impl_->session_.messages.push_back(Message::assistant_with_tool_calls(
                std::move(parsed.content), std::move(parsed.tool_calls)));
        } else {
            impl_->session_.messages.push_back(Message::assistant(std::move(parsed.content)));
        }
    } else {
        impl_->session_.messages.push_back(Message::assistant(std::move(generated_text)));
    }

    impl_->session_.estimated_tokens +=
        estimate_message_tokens(*impl_, impl_->session_.messages.back());
    note_history_append(*impl_);
    finalize_response();

    auto end_time = std::chrono::steady_clock::now();

    TextResponse response;
    response.text = impl_->session_.messages.back().content;
    response.usage.prompt_tokens = prompt_tokens;
    response.usage.completion_tokens = completion_tokens;
    response.usage.total_tokens = prompt_tokens + completion_tokens;

    response.metrics.latency_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (first_token_received) {
        response.metrics.time_to_first_token_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        auto generation_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - first_token_time);
        if (generation_time.count() > 0) {
            response.metrics.tokens_per_second =
                (completion_tokens * 1000.0) / generation_time.count();
        }
    }

    return response;
}

Expected<Model::GenerationResult> Model::generate_from_history(GenerationOverride generation,
                                                               TokenCallback on_token,
                                                               CancellationCallback should_cancel) {
    auto effective_options = resolve_generation_options(*impl_, generation);
    if (auto validation = effective_options.validate(); !validation) {
        return std::unexpected(validation.error());
    }

    impl_->session_.active_sampling = effective_options.sampling;

    auto prompt_result = render_prompt_delta(*impl_);
    if (!prompt_result) {
        return std::unexpected(prompt_result.error());
    }

    if (auto rebuild = ensure_grammar_sampler_for_pass(*impl_); !rebuild) {
        return std::unexpected(rebuild.error());
    }

    auto tokens_result = tokenize(*impl_, *prompt_result);
    if (!tokens_result) {
        return std::unexpected(tokens_result.error());
    }

    const int prompt_tokens = static_cast<int>(tokens_result->size());

    auto all_stops = merge_stop_sequences(*impl_, effective_options.stop_sequences);

    auto text_result = run_inference(*impl_, *tokens_result, effective_options.max_tokens,
                                     all_stops, on_token, should_cancel);

    if (!text_result) {
        return std::unexpected(text_result.error());
    }

    // Tool call detection: if tool calling is active, parse the output and
    // return the structured result so callers avoid a redundant re-parse.
    bool tool_detected = false;
    std::string parsed_content;
    std::vector<ToolCallInfo> parsed_tool_calls;
    if (impl_->session_.sampler_policy.is_native_tool_call()) {
        auto parsed = parse_tool_response(*text_result);
        tool_detected = !parsed.tool_calls.empty();
        parsed_content = std::move(parsed.content);
        parsed_tool_calls = std::move(parsed.tool_calls);
    }

    return GenerationResult{std::move(*text_result), prompt_tokens, tool_detected,
                            std::move(parsed_content), std::move(parsed_tool_calls)};
}

Expected<void> ensure_grammar_sampler_for_pass(Model::Impl& impl) {
    return impl.session_.sampler_policy.ensure_sampler_for_pass(impl);
}

std::vector<std::string> merge_stop_sequences(const Model::Impl& impl,
                                              std::vector<std::string> base) {
    if (impl.session_.sampler_policy.is_native_tool_call() && impl.session_.tool_state) {
        const auto& extras = impl.session_.tool_state->additional_stops;
        base.insert(base.end(), extras.begin(), extras.end());
    }
    return base;
}

GenerationOptions resolve_generation_options(const Model::Impl& impl,
                                             GenerationOverride generation) {
    if (!generation.options()) {
        return impl.loaded_.default_generation_options;
    }
    return *generation.options();
}

} // namespace zoo::core
