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

/// Reports whether the supplied integer is positive.
inline bool is_positive(int n) {
    return n > 0;
}

/// Concatenates two strings.
inline std::string concat(std::string a, std::string b) {
    return a + b;
}

/// Returns the area of a circle for the given radius.
inline double circle_area(double radius) {
    return 3.14159265358979323846 * radius * radius;
}

/// Returns the arithmetic negation of the supplied value.
inline int negate(int value) {
    return -value;
}

/// Returns a fixed timestamp for deterministic tests.
inline std::string get_time() {
    return "2024-01-01T00:00:00Z";
}

} // namespace tools
} // namespace testing
} // namespace zoo
