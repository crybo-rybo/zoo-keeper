/**
 * @file inference_phase.hpp
 * @brief Private llama.cpp prefill/decode helpers for model inference.
 */

#pragma once

#include "core/batch.hpp"
#include "core/model_impl.hpp"

#include <llama.h>
#include <string>
#include <vector>

namespace zoo::core {

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

} // namespace zoo::core
