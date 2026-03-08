/**
 * @file sample_responses.hpp
 * @brief Reusable model-output fixtures for parser and interceptor tests.
 */

#pragma once

#include <string>

namespace zoo {
namespace testing {
namespace responses {

/// Model output containing explanatory text followed by a valid tool call.
inline const std::string TOOL_CALL_ADD =
    R"(I'll add those numbers for you.
{"name": "add", "arguments": {"a": 3, "b": 4}})";

/// Model output containing only plain assistant text.
inline const std::string PLAIN_TEXT =
    "The capital of France is Paris.";

/// Model output with a valid tool call followed by trailing text.
inline const std::string TOOL_CALL_WITH_TRAILING =
    R"(Let me calculate that.
{"name": "multiply", "arguments": {"a": 5.0, "b": 3.0}}
Done!)";

/// Model output containing only a tool call.
inline const std::string TOOL_CALL_ONLY =
    R"({"name": "greet", "arguments": {"name": "Alice"}})";

/// Model output with a nested JSON object that should not be treated as a tool call.
inline const std::string NESTED_JSON_NOT_TOOL =
    R"(Here is the data: {"key": "value", "count": 42})";

/// Model output containing malformed JSON.
inline const std::string INVALID_JSON =
    R"({"name": "add", "arguments": {"a": 3, "b":)";

/// Model output containing a tool call with an explicit id field.
inline const std::string TOOL_CALL_WITH_ID =
    R"({"name": "search", "id": "call_123", "arguments": {"query": "test"}})";

/// First step of a multi-turn tool interaction.
inline const std::string TOOL_CALL_STEP1 =
    R"({"name": "get_time", "arguments": {}})";

/// Final answer emitted after a tool result has been injected.
inline const std::string FINAL_ANSWER_STEP2 =
    "The current time is 2024-01-01T00:00:00Z.";

/// Tool call that uses incorrect argument names.
inline const std::string TOOL_CALL_WRONG_ARGS =
    R"({"name": "add", "arguments": {"x": 3, "y": 4}})";

/// Tool call that uses incorrect argument types.
inline const std::string TOOL_CALL_WRONG_TYPES =
    R"({"name": "add", "arguments": {"a": "three", "b": "four"}})";

} // namespace responses
} // namespace testing
} // namespace zoo
