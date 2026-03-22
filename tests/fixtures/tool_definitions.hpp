/**
 * @file tool_definitions.hpp
 * @brief Reusable tool implementations for registry and validation tests.
 */

#pragma once

#include <string>

namespace zoo {
namespace testing {
namespace tools {

/// Returns the sum of two integers.
inline int add(int a, int b) {
    return a + b;
}

/// Returns the product of two floating-point values.
inline double multiply(double a, double b) {
    return a * b;
}

/// Returns a greeting for the supplied name.
inline std::string greet(std::string name) {
    return "Hello, " + name + "!";
}

/// Returns a fixed timestamp for deterministic tests.
inline std::string get_time() {
    return "2024-01-01T00:00:00Z";
}

} // namespace tools
} // namespace testing
} // namespace zoo
