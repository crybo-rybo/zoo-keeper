/**
 * @file log.hpp
 * @brief Internal logging macros used by the zoo-keeper runtime.
 */

#pragma once

#include "zoo/log.hpp"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace zoo::internal {

[[nodiscard]] inline LogLevel log_level_from_name(const char* level) noexcept {
    if (level == nullptr) {
        return LogLevel::Info;
    }
    if (std::string_view(level) == "debug") {
        return LogLevel::Debug;
    }
    if (std::string_view(level) == "warn" || std::string_view(level) == "warning") {
        return LogLevel::Warning;
    }
    if (std::string_view(level) == "error") {
        return LogLevel::Error;
    }
    return LogLevel::Info;
}

[[nodiscard]] inline std::string format_log_message(const char* fmt, std::va_list args) {
    if (fmt == nullptr) {
        return {};
    }

    std::array<char, 512> stack_buffer{};
    va_list args_copy;
    va_copy(args_copy, args);
    const int needed = std::vsnprintf(stack_buffer.data(), stack_buffer.size(), fmt, args_copy);
    va_end(args_copy);

    if (needed < 0) {
        return fmt;
    }
    if (static_cast<size_t>(needed) < stack_buffer.size()) {
        return stack_buffer.data();
    }

    std::vector<char> heap_buffer(static_cast<size_t>(needed) + 1);
    std::vsnprintf(heap_buffer.data(), heap_buffer.size(), fmt, args);
    return heap_buffer.data();
}

inline void log_emitf(const char* level, const char* fmt, ...) noexcept {
    std::string message;
    va_list args;
    va_start(args, fmt);
    try {
        message = format_log_message(fmt, args);
    } catch (...) {
        message = fmt == nullptr ? "" : fmt;
    }
    va_end(args);

    if (detail::has_log_callback()) {
        detail::dispatch_log(log_level_from_name(level), message.c_str());
        return;
    }

#ifdef ZOO_LOGGING_ENABLED
    std::fprintf(stderr, "[zoo:%s] %s\n", level == nullptr ? "info" : level, message.c_str());
#endif
}

} // namespace zoo::internal

#ifdef ZOO_LOGGING_ENABLED
#define ZOO_LOG(level, fmt, ...) ::zoo::internal::log_emitf(level, fmt __VA_OPT__(, ) __VA_ARGS__)
#else
#define ZOO_LOG(level, fmt, ...)                                                                   \
    do {                                                                                           \
        if (::zoo::detail::has_log_callback()) {                                                   \
            ::zoo::internal::log_emitf(level, fmt __VA_OPT__(, ) __VA_ARGS__);                     \
        }                                                                                          \
    } while (false)
#endif
