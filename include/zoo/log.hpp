/**
 * @file log.hpp
 * @brief Consumer-configurable diagnostics emitted by zoo-keeper internals.
 */

#pragma once

#include <mutex>

namespace zoo {

/**
 * @brief Severity attached to a zoo-keeper diagnostic message.
 */
enum class LogLevel {
    Debug,   ///< Verbose diagnostic detail.
    Info,    ///< Informational runtime event.
    Warning, ///< Recoverable condition worth surfacing.
    Error    ///< Operation or worker failure.
};

[[nodiscard]] inline const char* to_string(LogLevel level) noexcept {
    switch (level) {
    case LogLevel::Debug:
        return "debug";
    case LogLevel::Info:
        return "info";
    case LogLevel::Warning:
        return "warn";
    case LogLevel::Error:
        return "error";
    }
    return "unknown";
}

/**
 * @brief Callback invoked for zoo-keeper log messages.
 *
 * The `message` pointer is valid only for the duration of the callback.
 */
using LogCallback = void (*)(LogLevel level, const char* message, void* user_data);

namespace detail {

struct LogState {
    std::mutex mutex;
    LogCallback callback = nullptr;
    void* user_data = nullptr;
};

[[nodiscard]] inline LogState& log_state() {
    static LogState state;
    return state;
}

[[nodiscard]] inline bool has_log_callback() noexcept {
    try {
        auto& state = log_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.callback != nullptr;
    } catch (...) {
        return false;
    }
}

inline void dispatch_log(LogLevel level, const char* message) noexcept {
    LogCallback callback = nullptr;
    void* user_data = nullptr;

    try {
        auto& state = log_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        callback = state.callback;
        user_data = state.user_data;
    } catch (...) {
        return;
    }

    if (callback == nullptr) {
        return;
    }

    try {
        callback(level, message == nullptr ? "" : message, user_data);
    } catch (...) {
        // Logging callbacks must not unwind through zoo-keeper internals.
    }
}

} // namespace detail

/**
 * @brief Routes zoo-keeper diagnostics to a consumer callback.
 *
 * Passing `nullptr` suppresses callback logging and restores the default behavior
 * selected at build time.
 */
inline void set_log_callback(LogCallback callback, void* user_data = nullptr) {
    auto& state = detail::log_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.callback = callback;
    state.user_data = callback == nullptr ? nullptr : user_data;
}

/**
 * @brief Clears the configured zoo-keeper log callback.
 */
inline void reset_log_callback() {
    set_log_callback(nullptr, nullptr);
}

} // namespace zoo
