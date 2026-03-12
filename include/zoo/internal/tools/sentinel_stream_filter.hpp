/**
 * @file sentinel_stream_filter.hpp
 * @brief Streaming filter that suppresses grammar-mode tool-call sentinels.
 */

#pragma once

#include <string>
#include <string_view>

namespace zoo::tools {

/**
 * @brief Filters streamed text until the `<tool_call>` sentinel is fully matched.
 *
 * Grammar-constrained tool calling wraps tool JSON in `<tool_call>...</tool_call>`.
 * This filter buffers a possible opening sentinel across token boundaries, emits
 * any mismatched prefix as visible text, and suppresses all output once the
 * opening sentinel is fully matched.
 */
class SentinelStreamFilter {
  public:
    /**
     * @brief Consumes one streamed token fragment.
     *
     * @param token Raw token text from the model.
     * @return Visible text that should be forwarded to the user for this token.
     */
    std::string consume(std::string_view token) {
        std::string visible;
        if (suppressing_) {
            return visible;
        }

        for (char c : token) {
            if (suppressing_) {
                break;
            }

            std::string candidate = prefix_buf_;
            candidate.push_back(c);

            if (candidate == kOpenTag) {
                suppressing_ = true;
                prefix_buf_.clear();
                continue;
            }

            if (kOpenTag.starts_with(std::string_view(candidate))) {
                prefix_buf_ = std::move(candidate);
                continue;
            }

            if (!prefix_buf_.empty()) {
                visible += prefix_buf_;
                prefix_buf_.clear();
            }

            if (c == '<') {
                prefix_buf_ = "<";
            } else {
                visible.push_back(c);
            }
        }

        return visible;
    }

    /**
     * @brief Flushes any incomplete sentinel prefix at end-of-stream.
     *
     * @return Trailing visible text that was buffered while checking for a sentinel.
     */
    std::string finalize() {
        if (suppressing_) {
            return {};
        }
        std::string trailing = std::move(prefix_buf_);
        prefix_buf_.clear();
        return trailing;
    }

    /// Returns whether the opening sentinel has been fully matched.
    bool suppressing() const noexcept {
        return suppressing_;
    }

  private:
    static constexpr std::string_view kOpenTag = "<tool_call>";

    bool suppressing_ = false;
    std::string prefix_buf_;
};

} // namespace zoo::tools
