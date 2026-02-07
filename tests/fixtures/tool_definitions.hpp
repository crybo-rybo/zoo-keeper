#pragma once

#include <string>

namespace zoo {
namespace testing {
namespace tools {

// Simple tool functions for testing

inline int add(int a, int b) {
    return a + b;
}

inline double multiply(double a, double b) {
    return a * b;
}

inline std::string greet(std::string name) {
    return "Hello, " + name + "!";
}

inline bool is_positive(int n) {
    return n > 0;
}

inline std::string concat(std::string a, std::string b) {
    return a + b;
}

inline double circle_area(double radius) {
    return 3.14159265358979323846 * radius * radius;
}

inline int negate(int value) {
    return -value;
}

inline std::string get_time() {
    return "2024-01-01T00:00:00Z";
}

} // namespace tools
} // namespace testing
} // namespace zoo
