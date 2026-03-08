/**
 * @file log.hpp
 * @brief Internal logging macros used by the zoo-keeper runtime.
 */

#pragma once

#include <cstdio>

/**
 * @brief Emits a formatted log message when `ZOO_LOGGING_ENABLED` is defined.
 *
 * When logging is disabled the macro compiles to a no-op.
 */
#ifdef ZOO_LOGGING_ENABLED
#define ZOO_LOG(level, fmt, ...) \
    std::fprintf(stderr, "[zoo:%s] " fmt "\n", level __VA_OPT__(,) __VA_ARGS__)
#else
#define ZOO_LOG(level, fmt, ...) ((void)0)
#endif
