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

template <typename Impl> struct InferencePhase {
    Impl& impl;
    const CancellationCallback& should_cancel;

    [[nodiscard]] Expected<int> prefill(const std::vector<int>& prompt_tokens) const {
        const int n_batch = static_cast<int>(llama_n_batch(impl.ctx_.get()));
        const int base_pos = llama_memory_seq_pos_max(llama_get_memory(impl.ctx_.get()), 0) + 1;

        auto chunks = compute_prefill_chunks(static_cast<int>(prompt_tokens.size()), n_batch);
        for (const auto& chunk : chunks) {
            if (should_cancel && should_cancel()) {
                return std::unexpected(
                    Error{ErrorCode::RequestCancelled, "Request cancelled during prompt prefill"});
            }

            int n_ctx_used = base_pos + chunk.offset + chunk.count;
            if (n_ctx_used > impl.context_size_) {
                return std::unexpected(
                    Error{ErrorCode::ContextWindowExceeded, "Prompt tokens exceed context size"});
            }

            llama_batch batch = llama_batch_init(chunk.count, 0, 1);
            for (int i = 0; i < chunk.count; ++i) {
                batch.token[i] = static_cast<llama_token>(prompt_tokens[chunk.offset + i]);
                batch.pos[i] = static_cast<llama_pos>(base_pos + chunk.offset + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (chunk.emit_logits && i == chunk.count - 1);
            }
            batch.n_tokens = chunk.count;

            int rc = llama_decode(impl.ctx_.get(), batch);
            llama_batch_free(batch);
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

        const llama_token token = llama_sampler_sample(impl.sampler_.get(), impl.ctx_.get(), -1);
        if (llama_vocab_is_eog(impl.vocab_, token)) {
            return DecodedToken{token, {}, true};
        }

        char buffer[256];
        const int bytes = llama_token_to_piece(impl.vocab_, token, buffer, sizeof(buffer), 0, true);
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

        int rc = llama_decode(impl.ctx_.get(), batch);
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

Expected<std::string> Model::run_inference(const std::vector<int>& prompt_tokens, int max_tokens,
                                           const std::vector<std::string>& stop_sequences,
                                           TokenCallback on_token,
                                           CancellationCallback should_cancel) {
    std::string generated_text;
    const int effective_max = (max_tokens > 0) ? max_tokens : impl_->context_size_;
    generated_text.reserve(std::min(static_cast<size_t>(effective_max) * 8, size_t{65536}));
    int token_count = 0;
    bool stopped_by_callback = false;
    StopSequenceMatcher stop_matcher{std::span<const std::string>(stop_sequences)};
    StreamFilter stream_filter;
    if (on_token && impl_->tool_state_ &&
        impl_->grammar_mode_ == Impl::GrammarMode::NativeToolCall) {
        const auto& trigger_matcher = impl_->tool_state_->trigger_matcher;
        stream_filter = StreamFilter(std::span<const std::string>(trigger_matcher.word_triggers()),
                                     &trigger_matcher);
    }

    InferencePhase phase{*impl_, should_cancel};
    auto current_pos_result = phase.prefill(prompt_tokens);
    if (!current_pos_result) {
        return std::unexpected(current_pos_result.error());
    }

    int current_pos = *current_pos_result;
    llama_batch ar_batch = llama_batch_init(1, 0, 1);

    while (true) {
        auto decoded = phase.decode();
        if (!decoded) {
            llama_batch_free(ar_batch);
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
                    llama_batch_free(ar_batch);
                    return std::unexpected(action.error());
                }
                if (*action == TokenAction::Stop) {
                    stopped_by_callback = true;
                    break;
                }
            }
        }

        if (token_count >= effective_max || current_pos >= impl_->context_size_) {
            break;
        }

        if (auto result = phase.finalize(ar_batch, decoded->token, current_pos); !result) {
            llama_batch_free(ar_batch);
            return std::unexpected(result.error());
        }
    }

    llama_batch_free(ar_batch);

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

Expected<TextResponse> Model::generate(std::string_view user_message,
                                       const GenerationOptions& options, TokenCallback on_token,
                                       CancellationCallback should_cancel) {
    return generate(MessageView{Role::User, user_message}, options, on_token, should_cancel);
}

Expected<TextResponse> Model::generate(MessageView message, const GenerationOptions& options,
                                       TokenCallback on_token, CancellationCallback should_cancel) {
    auto start_time = std::chrono::steady_clock::now();
    auto effective_options = resolve_generation_options(options);
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

    impl_->active_sampling_ = effective_options.sampling;

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

    auto prompt_result = render_prompt_delta();
    if (!prompt_result) {
        rollback_last_message();
        return std::unexpected(prompt_result.error());
    }

    if (impl_->grammar_mode_ == Impl::GrammarMode::NativeToolCall) {
        if (impl_->tool_grammar_str_.empty()) {
            impl_->sampler_ = create_sampler_chain();
            if (!impl_->sampler_) {
                rollback_last_message();
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "Failed to rebuild sampler chain"});
            }
        } else if (!rebuild_sampler_with_tool_grammar()) {
            rollback_last_message();
            return std::unexpected(
                Error{ErrorCode::InferenceFailed, "Failed to rebuild tool grammar sampler"});
        }
    } else if (impl_->grammar_mode_ == Impl::GrammarMode::Schema &&
               !rebuild_sampler_with_schema_grammar()) {
        rollback_last_message();
        return std::unexpected(
            Error{ErrorCode::InferenceFailed, "Failed to rebuild schema grammar sampler"});
    }

    auto tokens_result = tokenize(*prompt_result);
    if (!tokens_result) {
        rollback_last_message();
        return std::unexpected(tokens_result.error());
    }

    const int prompt_tokens = static_cast<int>(tokens_result->size());

    // Merge config stop sequences with tool-calling additional stops.
    auto all_stops = effective_options.stop_sequences;
    if (impl_->tool_state_) {
        for (const auto& s : impl_->tool_state_->additional_stops) {
            all_stops.push_back(s);
        }
    }

    auto generate_result = run_inference(*tokens_result, effective_options.max_tokens, all_stops,
                                         TokenCallback(wrapped_callback), should_cancel);

    if (!generate_result) {
        rollback_last_message();
        return std::unexpected(generate_result.error());
    }

    std::string generated_text = std::move(*generate_result);

    // When native tool calling is active, parse the output to extract
    // structured tool calls for proper history round-tripping.
    if (impl_->grammar_mode_ == Impl::GrammarMode::NativeToolCall && impl_->tool_state_) {
        auto parsed = parse_tool_response(generated_text);
        if (!parsed.tool_calls.empty()) {
            impl_->messages_.push_back(Message::assistant_with_tool_calls(
                std::move(parsed.content), std::move(parsed.tool_calls)));
        } else {
            impl_->messages_.push_back(Message::assistant(std::move(parsed.content)));
        }
    } else {
        impl_->messages_.push_back(Message::assistant(std::move(generated_text)));
    }

    impl_->estimated_tokens_ += estimate_message_tokens(impl_->messages_.back());
    note_history_append();
    finalize_response();

    auto end_time = std::chrono::steady_clock::now();

    TextResponse response;
    response.text = impl_->messages_.back().content;
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

Expected<Model::GenerationResult> Model::generate_from_history(const GenerationOptions& options,
                                                               TokenCallback on_token,
                                                               CancellationCallback should_cancel) {
    auto effective_options = resolve_generation_options(options);
    if (auto validation = effective_options.validate(); !validation) {
        return std::unexpected(validation.error());
    }

    impl_->active_sampling_ = effective_options.sampling;

    auto prompt_result = render_prompt_delta();
    if (!prompt_result) {
        return std::unexpected(prompt_result.error());
    }

    if (impl_->grammar_mode_ == Impl::GrammarMode::NativeToolCall) {
        if (impl_->tool_grammar_str_.empty()) {
            impl_->sampler_ = create_sampler_chain();
            if (!impl_->sampler_) {
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "Failed to rebuild sampler chain"});
            }
        } else if (!rebuild_sampler_with_tool_grammar()) {
            return std::unexpected(
                Error{ErrorCode::InferenceFailed, "Failed to rebuild tool grammar sampler"});
        }
    } else if (impl_->grammar_mode_ == Impl::GrammarMode::Schema &&
               !rebuild_sampler_with_schema_grammar()) {
        return std::unexpected(
            Error{ErrorCode::InferenceFailed, "Failed to rebuild schema grammar sampler"});
    }

    auto tokens_result = tokenize(*prompt_result);
    if (!tokens_result) {
        return std::unexpected(tokens_result.error());
    }

    const int prompt_tokens = static_cast<int>(tokens_result->size());

    // Merge config stop sequences with tool-calling additional stops.
    auto all_stops = effective_options.stop_sequences;
    if (impl_->tool_state_) {
        for (const auto& s : impl_->tool_state_->additional_stops) {
            all_stops.push_back(s);
        }
    }

    auto text_result = run_inference(*tokens_result, effective_options.max_tokens, all_stops,
                                     on_token, should_cancel);

    if (!text_result) {
        return std::unexpected(text_result.error());
    }

    // Tool call detection: if tool calling is active, parse the output and
    // return the structured result so callers avoid a redundant re-parse.
    bool tool_detected = false;
    std::string parsed_content;
    std::vector<ToolCallInfo> parsed_tool_calls;
    if (impl_->grammar_mode_ == Impl::GrammarMode::NativeToolCall) {
        auto parsed = parse_tool_response(*text_result);
        tool_detected = !parsed.tool_calls.empty();
        parsed_content = std::move(parsed.content);
        parsed_tool_calls = std::move(parsed.tool_calls);
    }

    return GenerationResult{std::move(*text_result), prompt_tokens, tool_detected,
                            std::move(parsed_content), std::move(parsed_tool_calls)};
}

GenerationOptions Model::resolve_generation_options(const GenerationOptions& overrides) const {
    if (overrides.is_default()) {
        return impl_->default_generation_options_;
    }
    return overrides;
}

} // namespace zoo::core
