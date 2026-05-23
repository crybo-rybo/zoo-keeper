/**
 * @file sampling_helpers.hpp
 * @brief Internal helpers for sampler trigger pattern construction.
 */

#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace zoo::core::detail {

/// Returns true when a trigger-pattern character must be escaped for regex use.
inline bool regex_escape_needs_backslash(char ch) {
    switch (ch) {
    case '-':
    case '[':
    case ']':
    case '{':
    case '}':
    case '(':
    case ')':
    case '*':
    case '+':
    case '?':
    case '.':
    case ',':
    case '\\':
    case '^':
    case '$':
    case '|':
    case '#':
        return true;
    default:
        return std::isspace(static_cast<unsigned char>(ch)) != 0;
    }
}

/// Escapes regex metacharacters in sampler trigger patterns without `std::regex`.
inline std::string regex_escape(std::string_view str) {
    std::string escaped;
    escaped.reserve(str.size() * 2);
    for (const char ch : str) {
        if (regex_escape_needs_backslash(ch)) {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

} // namespace zoo::core::detail
