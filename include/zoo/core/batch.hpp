#pragma once

#include <vector>
#include <algorithm>

namespace zoo::core {

/// Describes one chunk of tokens to be decoded as a batch.
struct BatchChunk {
    int offset;      // Start index into the token array
    int count;       // Number of tokens in this chunk
    bool emit_logits; // Whether the last token should emit logits

    bool operator==(const BatchChunk& other) const = default;
};

/// Compute a chunked prefill plan for a prompt of `total_tokens` tokens,
/// given a maximum batch size of `n_batch`. Only the final token of the
/// final chunk has emit_logits=true; all others are false.
///
/// Preconditions: total_tokens > 0, n_batch > 0
/// Returns empty vector if preconditions are violated.
[[nodiscard]] inline std::vector<BatchChunk> compute_prefill_chunks(
    int total_tokens, int n_batch
) {
    if (total_tokens <= 0 || n_batch <= 0) return {};

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

} // namespace zoo::core
