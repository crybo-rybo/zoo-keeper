/**
 * @file model_inference.cpp
 * @brief Inference and response assembly for `zoo::core::Model`.
 */

#include "zoo/core/model.hpp"
#include "zoo/core/model_tool_calling_state.hpp"

#include "zoo/internal/core/batch.hpp"
#include "zoo/internal/core/stream_filter.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <llama.h>
#include <string_view>

namespace zoo::core {

namespace {

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

size_t find_stop_sequence(const std::string& generated_text,
                          const std::vector<std::string>& stop_sequences) {
    for (const auto& stop_sequence : stop_sequences) {
        if (stop_sequence.empty()) {
            continue;
        }
        if (generated_text.size() >= stop_sequence.size() &&
            generated_text.compare(generated_text.size() - stop_sequence.size(),
                                   stop_sequence.size(), stop_sequence) == 0) {
            return stop_sequence.size();
        }
    }
    return 0;
}

} // namespace

Expected<std::string> Model::run_inference(const std::vector<int>& prompt_tokens, int max_tokens,
                                           const std::vector<std::string>& stop_sequences,
                                           const std::optional<TokenCallback>& on_token,
                                           const CancellationCallback& should_cancel) {
    std::string generated_text;
    const int effective_max = (max_tokens > 0) ? max_tokens : context_size_;
    generated_text.reserve(std::min(static_cast<size_t>(effective_max) * 8, size_t{65536}));
    int token_count = 0;
    bool stopped_by_callback = false;
    bool tool_stream_suppressed = false;
    std::optional<ToolCallWordTriggerFilter> word_trigger_filter;
    if (on_token && tool_state_ && grammar_mode_ == GrammarMode::NativeToolCall) {
        auto word_triggers = extract_word_triggers(tool_state_->grammar_triggers);
        if (!word_triggers.empty()) {
            word_trigger_filter.emplace(std::move(word_triggers));
        }
    }

    const int n_batch = static_cast<int>(llama_n_batch(ctx_.get()));
    const int base_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_.get()), 0) + 1;

    auto chunks = compute_prefill_chunks(static_cast<int>(prompt_tokens.size()), n_batch);

    for (const auto& chunk : chunks) {
        if (should_cancel && should_cancel()) {
            return std::unexpected(
                Error{ErrorCode::RequestCancelled, "Request cancelled during prompt prefill"});
        }

        int n_ctx_used = base_pos + chunk.offset + chunk.count;
        if (n_ctx_used > context_size_) {
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

        int rc = llama_decode(ctx_.get(), batch);
        llama_batch_free(batch);
        if (rc != 0) {
            return std::unexpected(
                Error{ErrorCode::InferenceFailed, "Failed to decode prefill batch"});
        }
    }

    int current_pos = base_pos + static_cast<int>(prompt_tokens.size());
    llama_token new_token;
    llama_batch ar_batch = llama_batch_init(1, 0, 1);

    while (true) {
        if (should_cancel && should_cancel()) {
            return std::unexpected(
                Error{ErrorCode::RequestCancelled, "Request cancelled during generation"});
        }

        new_token = llama_sampler_sample(sampler_.get(), ctx_.get(), -1);
        if (llama_vocab_is_eog(vocab_, new_token)) {
            break;
        }

        char buff[256];
        const int n = llama_token_to_piece(vocab_, new_token, buff, sizeof(buff), 0, true);
        if (n < 0) {
            return std::unexpected(Error{ErrorCode::Unknown, "Failed to convert token"});
        }

        generated_text.append(buff, static_cast<size_t>(n));
        ++token_count;

        if (!stop_sequences.empty()) {
            const size_t match_len = find_stop_sequence(generated_text, stop_sequences);
            if (match_len > 0) {
                generated_text.resize(generated_text.size() - match_len);
                break;
            }
        }

        if (on_token) {
            if (!tool_stream_suppressed) {
                std::string visible_chunk;
                if (word_trigger_filter) {
                    visible_chunk = word_trigger_filter->consume(
                        std::string_view(buff, static_cast<size_t>(n)));
                    tool_stream_suppressed = word_trigger_filter->suppressing();
                } else {
                    visible_chunk.assign(buff, static_cast<size_t>(n));
                }

                // For regex-style triggers we can only detect once the full
                // accumulated text matches, so suppress the current fragment
                // before forwarding anything visible from it.
                if (!tool_stream_suppressed && tool_state_ &&
                    grammar_mode_ == GrammarMode::NativeToolCall &&
                    is_tool_trigger_detected(generated_text, tool_state_->grammar_triggers)) {
                    tool_stream_suppressed = true;
                    visible_chunk.clear();
                }

                if (!visible_chunk.empty()) {
                    auto action = invoke_token_callback(*on_token, visible_chunk);
                    if (!action) {
                        return std::unexpected(action.error());
                    }
                    if (*action == TokenAction::Stop) {
                        stopped_by_callback = true;
                        break;
                    }
                }
            }
        }

        if (token_count >= effective_max || current_pos >= context_size_) {
            break;
        }

        ar_batch.token[0] = new_token;
        ar_batch.pos[0] = static_cast<llama_pos>(current_pos);
        ar_batch.n_seq_id[0] = 1;
        ar_batch.seq_id[0][0] = 0;
        ar_batch.logits[0] = true;
        ar_batch.n_tokens = 1;

        int rc = llama_decode(ctx_.get(), ar_batch);
        if (rc != 0) {
            llama_batch_free(ar_batch);
            return std::unexpected(Error{ErrorCode::InferenceFailed, "Failed to decode token"});
        }
        ++current_pos;
    }

    llama_batch_free(ar_batch);

    if (on_token && word_trigger_filter && !tool_stream_suppressed && !stopped_by_callback) {
        std::string trailing = word_trigger_filter->finalize();
        if (!trailing.empty()) {
            auto action = invoke_token_callback(*on_token, trailing);
            if (!action) {
                return std::unexpected(action.error());
            }
        }
    }

    return generated_text;
}

Expected<Response> Model::generate(const std::string& user_message,
                                   std::optional<TokenCallback> on_token,
                                   CancellationCallback should_cancel) {
    auto start_time = std::chrono::steady_clock::now();

    auto add_result = add_message(Message::user(user_message));
    if (!add_result) {
        return std::unexpected(add_result.error());
    }

    std::chrono::steady_clock::time_point first_token_time;
    bool first_token_received = false;
    int completion_tokens = 0;

    const std::optional<TokenCallback> effective_callback =
        on_token ? std::move(on_token) : config_.on_token;

    auto wrapped_callback = [&](std::string_view token) -> TokenAction {
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        if (effective_callback) {
            return (*effective_callback)(token);
        }
        return TokenAction::Continue;
    };

    auto prompt_result = render_prompt_delta();
    if (!prompt_result) {
        rollback_last_message();
        return std::unexpected(prompt_result.error());
    }

    if (grammar_mode_ == GrammarMode::NativeToolCall) {
        if (tool_grammar_str_.empty()) {
            sampler_ = create_sampler_chain();
            if (!sampler_) {
                rollback_last_message();
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "Failed to rebuild sampler chain"});
            }
        } else if (!rebuild_sampler_with_tool_grammar()) {
            rollback_last_message();
            return std::unexpected(
                Error{ErrorCode::InferenceFailed, "Failed to rebuild tool grammar sampler"});
        }
    } else if (grammar_mode_ == GrammarMode::Schema && !rebuild_sampler_with_schema_grammar()) {
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
    auto all_stops = config_.stop_sequences;
    if (tool_state_) {
        for (const auto& s : tool_state_->additional_stops) {
            all_stops.push_back(s);
        }
    }

    auto generate_result =
        run_inference(*tokens_result, config_.max_tokens, all_stops,
                      std::optional<TokenCallback>(wrapped_callback), should_cancel);

    if (!generate_result) {
        rollback_last_message();
        return std::unexpected(generate_result.error());
    }

    std::string generated_text = std::move(*generate_result);

    // When native tool calling is active, parse the output to extract
    // structured tool calls for proper history round-tripping.
    if (grammar_mode_ == GrammarMode::NativeToolCall && tool_state_) {
        auto parsed = parse_tool_response(generated_text);
        if (!parsed.tool_calls.empty()) {
            messages_.push_back(Message::assistant_with_tool_calls(std::move(parsed.content),
                                                                   std::move(parsed.tool_calls)));
        } else {
            messages_.push_back(Message::assistant(std::move(parsed.content)));
        }
    } else {
        messages_.push_back(Message::assistant(std::move(generated_text)));
    }

    estimated_tokens_ += estimate_message_tokens(messages_.back());
    note_history_append();
    finalize_response();

    auto end_time = std::chrono::steady_clock::now();

    Response response;
    response.text = messages_.back().content;
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

Expected<Model::GenerationResult>
Model::generate_from_history(std::optional<TokenCallback> on_token,
                             CancellationCallback should_cancel) {
    auto prompt_result = render_prompt_delta();
    if (!prompt_result) {
        return std::unexpected(prompt_result.error());
    }

    if (grammar_mode_ == GrammarMode::NativeToolCall) {
        if (tool_grammar_str_.empty()) {
            sampler_ = create_sampler_chain();
            if (!sampler_) {
                return std::unexpected(
                    Error{ErrorCode::InferenceFailed, "Failed to rebuild sampler chain"});
            }
        } else if (!rebuild_sampler_with_tool_grammar()) {
            return std::unexpected(
                Error{ErrorCode::InferenceFailed, "Failed to rebuild tool grammar sampler"});
        }
    } else if (grammar_mode_ == GrammarMode::Schema && !rebuild_sampler_with_schema_grammar()) {
        return std::unexpected(
            Error{ErrorCode::InferenceFailed, "Failed to rebuild schema grammar sampler"});
    }

    auto tokens_result = tokenize(*prompt_result);
    if (!tokens_result) {
        return std::unexpected(tokens_result.error());
    }

    const int prompt_tokens = static_cast<int>(tokens_result->size());

    // Merge config stop sequences with tool-calling additional stops.
    auto all_stops = config_.stop_sequences;
    if (tool_state_) {
        for (const auto& s : tool_state_->additional_stops) {
            all_stops.push_back(s);
        }
    }

    auto text_result =
        run_inference(*tokens_result, config_.max_tokens, all_stops, on_token, should_cancel);

    if (!text_result) {
        return std::unexpected(text_result.error());
    }

    // Tool call detection: if tool calling is active, parse the output and
    // return the structured result so callers avoid a redundant re-parse.
    bool tool_detected = false;
    std::string parsed_content;
    std::vector<ToolCallInfo> parsed_tool_calls;
    if (grammar_mode_ == GrammarMode::NativeToolCall) {
        auto parsed = parse_tool_response(*text_result);
        tool_detected = !parsed.tool_calls.empty();
        parsed_content = std::move(parsed.content);
        parsed_tool_calls = std::move(parsed.tool_calls);
    }

    return GenerationResult{std::move(*text_result), prompt_tokens, tool_detected,
                            std::move(parsed_content), std::move(parsed_tool_calls)};
}

} // namespace zoo::core
