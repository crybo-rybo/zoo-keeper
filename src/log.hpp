/**
 * @file log.hpp
 * @brief Internal logging macros used by the zoo-keeper runtime.
 *
 * When a consumer-supplied callback is registered via `zoo::set_log_callback()`,
 * messages are routed there instead of stderr.
 */

#pragma once

#include "zoo/log.hpp"

#include <cstdio>

namespace zoo::internal {
LogCallback get_log_callback() noexcept;
void* get_log_user_data() noexcept;

inline LogLevel log_level_from_string(const char* level) noexcept {
    switch (level[0]) {
    case 'e':
        return LogLevel::Error;
    case 'w':
        return LogLevel::Warn;
    case 'd':
        return LogLevel::Debug;
    default:
        return LogLevel::Info;
    }
}
} // namespace zoo::internal

#ifdef ZOO_LOGGING_ENABLED
#define ZOO_LOG(level_str, fmt, ...)                                                               \
    do {                                                                                           \
        auto zoo_log_cb_ = ::zoo::internal::get_log_callback();                                    \
        if (zoo_log_cb_) {                                                                         \
            char zoo_log_buf_[512];                                                                \
            std::snprintf(zoo_log_buf_, sizeof(zoo_log_buf_), fmt __VA_OPT__(, ) __VA_ARGS__);     \
            zoo_log_cb_(::zoo::internal::log_level_from_string(level_str), zoo_log_buf_,           \
                        ::zoo::internal::get_log_user_data());                                     \
        } else {                                                                                   \
            std::fprintf(stderr, "[zoo:%s] " fmt "\n", level_str __VA_OPT__(, ) __VA_ARGS__);      \
        }                                                                                          \
    } while (0)
#else
#define ZOO_LOG(level, fmt, ...) ((void)0)
#endif
