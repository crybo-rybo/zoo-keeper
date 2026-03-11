/**
 * @file interceptor.hpp
 * @brief Streaming interceptor that hides tool-call JSON from end users.
 */

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <zoo/core/types.hpp>
#include <zoo/tools/parser.hpp>
#include <zoo/tools/types.hpp>

namespace zoo::tools {

/**
 * @brief Intercepts streamed tokens and extracts brace-delimited tool calls.
 *
 * In heuristic mode the interceptor forwards visible text to the user callback,
 * buffers candidate JSON objects, and stops generation once a valid tool call
 * has been fully observed.
 */
class ToolCallInterceptor {
  public:
    /**
     * @brief Final state produced after a streaming pass completes.
     */
    struct Result {
        std::optional<ToolCall> tool_call; ///< Parsed tool call when one was detected.
        std::string visible_text;          ///< User-visible text with tool-call JSON removed.
        std::string full_text;             ///< Full generated text including any buffered JSON.
    };

    /**
     * @brief Creates an interceptor for one streaming pass.
     *
     * @param user_callback Optional callback that receives only visible text.
     */
    explicit ToolCallInterceptor(
        std::optional<std::function<void(std::string_view)>> user_callback = std::nullopt)
        : user_callback_(std::move(user_callback)) {}

    /// Moving is disabled because callbacks returned by `make_callback()` capture `this`.
    ToolCallInterceptor(ToolCallInterceptor&&) = delete;
    /// Moving is disabled because callbacks returned by `make_callback()` capture `this`.
    ToolCallInterceptor& operator=(ToolCallInterceptor&&) = delete;
    /// Copying is disabled because the interceptor owns mutable streaming state.
    ToolCallInterceptor(const ToolCallInterceptor&) = delete;
    /// Copying is disabled because the interceptor owns mutable streaming state.
    ToolCallInterceptor& operator=(const ToolCallInterceptor&) = delete;

    /**
     * @brief Returns a token callback that feeds the interceptor state machine.
     *
     * @return Callback suitable for `TokenCallback`.
     */
    TokenCallback make_callback() {
        return [this](std::string_view token) -> TokenAction {
            full_text_.append(token.data(), token.size());
            return process_token(token);
        };
    }

    /**
     * @brief Finalizes interception after generation stops.
     *
     * @return Parsed tool call, user-visible text, and full generated text for the pass.
     */
    Result finalize() {
        Result result;
        result.full_text = std::move(full_text_);

        // Tool call was detected mid-stream (callback returned Stop)
        if (pending_tool_call_.has_value()) {
            result.tool_call = std::move(pending_tool_call_);
            result.visible_text = std::move(visible_text_);
            return result;
        }

        // Partial sentinel prefix buffered but never completed — treat as visible text
        if (state_ == State::SentinelPrefix && !sentinel_buf_.empty()) {
            visible_text_ += sentinel_buf_;
            sentinel_buf_.clear();
        }

        if (state_ == State::Buffering) {
            // Generation ended while buffering — check if buffer is a complete tool call
            auto parse_result = ToolCallParser::parse(buffer_);
            if (parse_result.tool_call.has_value()) {
                result.tool_call = std::move(parse_result.tool_call);
                result.visible_text = std::move(visible_text_);
                return result;
            }
            // Not a tool call — include buffer in visible_text for the Response.
            // Not emitted via user_callback_ since streaming is already complete.
            visible_text_ += buffer_;
        }

        result.visible_text = std::move(visible_text_);
        return result;
    }

  private:
    enum class State { Normal, SentinelPrefix, TagMatched, Buffering };

    static constexpr std::string_view kOpenTag = "<tool_call>";

    /// Dispatches token handling based on current state.
    TokenAction process_token(std::string_view token) {
        switch (state_) {
        case State::Normal:         return process_normal(token);
        case State::SentinelPrefix: return process_sentinel_prefix(token);
        case State::TagMatched:     return process_tag_matched(token);
        case State::Buffering:      return process_buffering(token);
        }
        return TokenAction::Continue;
    }

    /// Streams normal text until a potential JSON object or sentinel tag begins.
    TokenAction process_normal(std::string_view token) {
        for (size_t i = 0; i < token.size(); ++i) {
            if (token[i] == '{') {
                // Flush everything before the '{' to the user
                if (i > 0) {
                    auto prefix = token.substr(0, i);
                    visible_text_.append(prefix.data(), prefix.size());
                    emit(prefix);
                }
                state_ = State::Buffering;
                brace_depth_ = 0;
                buffer_.clear();
                return process_buffering(token.substr(i));
            }
            if (token[i] == '<') {
                // Flush everything before '<' to the user
                if (i > 0) {
                    auto prefix = token.substr(0, i);
                    visible_text_.append(prefix.data(), prefix.size());
                    emit(prefix);
                }
                state_ = State::SentinelPrefix;
                sentinel_buf_ = "<";
                return process_sentinel_prefix(token.substr(i + 1));
            }
        }

        // No trigger found — pass entire token through
        visible_text_.append(token.data(), token.size());
        emit(token);
        return TokenAction::Continue;
    }

    /// Accumulates characters to determine whether a '<tool_call>' tag is forming.
    /// Suppresses the accumulated prefix if it matches; flushes it as visible text otherwise.
    TokenAction process_sentinel_prefix(std::string_view token) {
        for (size_t i = 0; i < token.size(); ++i) {
            sentinel_buf_ += token[i];
            if (sentinel_buf_ == kOpenTag) {
                sentinel_buf_.clear();
                state_ = State::TagMatched;
                return process_tag_matched(token.substr(i + 1));
            }
            if (!kOpenTag.starts_with(std::string_view(sentinel_buf_))) {
                // Not a match — emit what was buffered and resume normal processing
                visible_text_ += sentinel_buf_;
                emit(sentinel_buf_);
                sentinel_buf_.clear();
                state_ = State::Normal;
                return process_normal(token.substr(i + 1));
            }
        }
        return TokenAction::Continue;
    }

    /// Silently discards whitespace between '<tool_call>' and the opening '{'.
    TokenAction process_tag_matched(std::string_view token) {
        for (size_t i = 0; i < token.size(); ++i) {
            if (token[i] == '{') {
                state_ = State::Buffering;
                brace_depth_ = 0;
                buffer_.clear();
                in_string_ = false;
                escape_next_ = false;
                return process_buffering(token.substr(i));
            }
            // Suppress whitespace (and anything else) between the tag and the JSON object
        }
        return TokenAction::Continue;
    }

    /// Buffers JSON text until it either parses as a tool call or falls back to visible text.
    TokenAction process_buffering(std::string_view token) {
        for (size_t i = 0; i < token.size(); ++i) {
            char c = token[i];
            buffer_ += c;

            if (in_string_) {
                if (escape_next_) {
                    escape_next_ = false;
                } else if (c == '\\') {
                    escape_next_ = true;
                } else if (c == '"') {
                    in_string_ = false;
                }
                continue;
            }

            if (c == '"') {
                in_string_ = true;
                continue;
            }

            if (c == '{') {
                ++brace_depth_;
            } else if (c == '}') {
                --brace_depth_;
                if (brace_depth_ == 0) {
                    // Complete JSON object — check if it's a tool call
                    auto parse_result = ToolCallParser::parse(buffer_);
                    if (parse_result.tool_call.has_value()) {
                        // Tool call detected — stop generation
                        // Any text before the tool call in the buffer is visible
                        if (!parse_result.text_before.empty()) {
                            visible_text_ += parse_result.text_before;
                            emit(parse_result.text_before);
                        }
                        pending_tool_call_ = std::move(parse_result.tool_call);
                        return TokenAction::Stop;
                    }

                    // Not a tool call — flush buffer as normal text
                    visible_text_ += buffer_;
                    emit(buffer_);
                    buffer_.clear();
                    in_string_ = false;
                    escape_next_ = false;
                    state_ = State::Normal;

                    // Process any remaining characters in this token
                    if (i + 1 < token.size()) {
                        return process_normal(token.substr(i + 1));
                    }
                    return TokenAction::Continue;
                }
            }
        }

        return TokenAction::Continue;
    }

    /// Forwards visible text to the optional user callback.
    void emit(std::string_view text) {
        if (user_callback_) {
            (*user_callback_)(text);
        }
    }

    std::optional<std::function<void(std::string_view)>> user_callback_;

    State state_ = State::Normal;
    std::string buffer_;
    std::string sentinel_buf_;
    std::string visible_text_;
    std::string full_text_;
    int brace_depth_ = 0;
    bool in_string_ = false;
    bool escape_next_ = false;
    std::optional<ToolCall> pending_tool_call_;
};

} // namespace zoo::tools
