#pragma once

#include <cstdio>

#ifdef ZOO_LOGGING_ENABLED
#define ZOO_LOG(level, fmt, ...) \
    std::fprintf(stderr, "[zoo:%s] " fmt "\n", level __VA_OPT__(,) __VA_ARGS__)
#else
#define ZOO_LOG(level, fmt, ...) ((void)0)
#endif
