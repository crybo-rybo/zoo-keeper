/**
 * @file log.hpp
 * @brief Consumer-configurable logging callback for the zoo-keeper runtime.
 *
 * By default, zoo-keeper logs to stderr via `fprintf` when `ZOO_LOGGING_ENABLED`
 * is defined. Call `set_log_callback()` to redirect log output into your own
 * logging infrastructure instead.
 */

#pragma once

#include <cstdint>

namespace zoo {

/**
 * @brief Severity levels emitted by zoo-keeper's internal logging.
 */
enum class LogLevel : uint8_t {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
};

/**
 * @brief Signature for a user-supplied log sink.
 *
 * @param level    Severity of the message.
 * @param message  Null-terminated log message (no trailing newline).
 * @param user_data  Opaque pointer passed to `set_log_callback()`.
 */
using LogCallback = void (*)(LogLevel level, const char* message, void* user_data);

/**
 * @brief Registers a callback that receives all zoo-keeper log output.
 *
 * Pass `nullptr` to restore the default stderr behavior.
 * Thread-safe: the callback pointer is stored atomically.
 *
 * @param callback  Function to invoke for each log message, or `nullptr`.
 * @param user_data Opaque pointer forwarded to every callback invocation.
 */
void set_log_callback(LogCallback callback, void* user_data = nullptr);

} // namespace zoo
