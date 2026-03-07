#pragma once

#include "types.hpp"
#include "parser.hpp"
#include <zoo/core/types.hpp>
#include <string>
#include <string_view>
#include <optional>
#include <functional>

namespace zoo::tools {

class ToolCallInterceptor {
public:
    struct Result {
        std::optional<ToolCall> tool_call;
        std::string visible_text;
        std::string full_text;
    };

    explicit ToolCallInterceptor(
        std::optional<std::function<void(std::string_view)>> user_callback = std::nullopt
    )
        : user_callback_(std::move(user_callback))
    {}

    // make_callback() captures 'this', so moves would dangle the callback
    ToolCallInterceptor(ToolCallInterceptor&&) = delete;
    ToolCallInterceptor& operator=(ToolCallInterceptor&&) = delete;
    ToolCallInterceptor(const ToolCallInterceptor&) = delete;
    ToolCallInterceptor& operator=(const ToolCallInterceptor&) = delete;

    TokenCallback make_callback() {
        return [this](std::string_view token) -> TokenAction {
            full_text_.append(token.data(), token.size());
            return process_token(token);
        };
    }

    Result finalize() {
        Result result;
        result.full_text = std::move(full_text_);

        // Tool call was detected mid-stream (callback returned Stop)
        if (pending_tool_call_.has_value()) {
            result.tool_call = std::move(pending_tool_call_);
            result.visible_text = std::move(visible_text_);
            return result;
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
    enum class State { Normal, Buffering };

    TokenAction process_token(std::string_view token) {
        if (state_ == State::Normal) {
            return process_normal(token);
        }
        return process_buffering(token);
    }

    TokenAction process_normal(std::string_view token) {
        // Scan each character for the start of a potential JSON object
        for (size_t i = 0; i < token.size(); ++i) {
            if (token[i] == '{') {
                // Flush everything before the '{' to the user
                if (i > 0) {
                    auto prefix = token.substr(0, i);
                    visible_text_.append(prefix.data(), prefix.size());
                    emit(prefix);
                }

                // Start buffering from '{'
                state_ = State::Buffering;
                brace_depth_ = 0;
                buffer_.clear();
                auto remainder = token.substr(i);
                return process_buffering(remainder);
            }
        }

        // No '{' found — pass entire token through
        visible_text_.append(token.data(), token.size());
        emit(token);
        return TokenAction::Continue;
    }

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

    void emit(std::string_view text) {
        if (user_callback_) {
            (*user_callback_)(text);
        }
    }

    std::optional<std::function<void(std::string_view)>> user_callback_;

    State state_ = State::Normal;
    std::string buffer_;
    std::string visible_text_;
    std::string full_text_;
    int brace_depth_ = 0;
    bool in_string_ = false;
    bool escape_next_ = false;
    std::optional<ToolCall> pending_tool_call_;
};

} // namespace zoo::tools
