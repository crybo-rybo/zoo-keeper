/**
 * @file batch.hpp
 * @brief Helpers for chunking prompt prefill work into llama.cpp decode batches.
 */

#pragma once

#include <algorithm>
#include <llama.h>
#include <utility>
#include <vector>

namespace zoo::core {

/**
 * @brief Describes one contiguous token range to decode as a batch.
 */
struct BatchChunk {
    int offset;       ///< Zero-based start index into the prompt token array.
    int count;        ///< Number of tokens included in the chunk.
    bool emit_logits; ///< Whether the final token in the chunk should request logits.

    /// Compares two chunk plans for equality.
    bool operator==(const BatchChunk& other) const = default;
};

/**
 * @brief Splits a prompt into decode batches that fit within `n_batch`.
 *
 * Only the last token of the final chunk emits logits; earlier chunks are used
 * purely for KV-cache prefill.
 *
 * @param total_tokens Total number of prompt tokens to prefill.
 * @param n_batch Maximum number of tokens the backend may decode per batch.
 * @return A sequence of chunk descriptors, or an empty vector when either input
 *         is non-positive.
 */
[[nodiscard]] inline std::vector<BatchChunk> compute_prefill_chunks(int total_tokens, int n_batch) {
    if (total_tokens <= 0 || n_batch <= 0)
        return {};

    std::vector<BatchChunk> chunks;
    int offset = 0;
    while (offset < total_tokens) {
        int count = std::min(n_batch, total_tokens - offset);
        bool is_last = (offset + count == total_tokens);
        chunks.push_back({offset, count, is_last});
        offset += count;
    }
    return chunks;
}

class LlamaBatchHandle {
  public:
    LlamaBatchHandle(int n_tokens, int embd, int n_seq_max)
        : batch_(llama_batch_init(n_tokens, embd, n_seq_max)), owns_(true) {}

    LlamaBatchHandle(const LlamaBatchHandle&) = delete;
    LlamaBatchHandle& operator=(const LlamaBatchHandle&) = delete;

    LlamaBatchHandle(LlamaBatchHandle&& other) noexcept
        : batch_(std::exchange(other.batch_, llama_batch{})),
          owns_(std::exchange(other.owns_, false)) {}

    LlamaBatchHandle& operator=(LlamaBatchHandle&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        batch_ = std::exchange(other.batch_, llama_batch{});
        owns_ = std::exchange(other.owns_, false);
        return *this;
    }

    ~LlamaBatchHandle() {
        reset();
    }

    [[nodiscard]] llama_batch& get() noexcept {
        return batch_;
    }

  private:
    void reset() noexcept {
        if (owns_) {
            llama_batch_free(batch_);
            owns_ = false;
            batch_ = llama_batch{};
        }
    }

    llama_batch batch_{};
    bool owns_ = false;
};

} // namespace zoo::core
