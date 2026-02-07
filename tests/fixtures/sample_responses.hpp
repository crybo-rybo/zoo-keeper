#pragma once

#include <string>

namespace zoo {
namespace testing {
namespace responses {

// Model output containing a tool call
inline const std::string TOOL_CALL_ADD =
    R"(I'll add those numbers for you.
{"name": "add", "arguments": {"a": 3, "b": 4}})";

// Model output with no tool call
inline const std::string PLAIN_TEXT =
    "The capital of France is Paris.";

// Model output with a tool call and text after
inline const std::string TOOL_CALL_WITH_TRAILING =
    R"(Let me calculate that.
{"name": "multiply", "arguments": {"a": 5.0, "b": 3.0}}
Done!)";

// Model output with just a tool call
inline const std::string TOOL_CALL_ONLY =
    R"({"name": "greet", "arguments": {"name": "Alice"}})";

// Model output with nested JSON (not a tool call)
inline const std::string NESTED_JSON_NOT_TOOL =
    R"(Here is the data: {"key": "value", "count": 42})";

// Model output with invalid JSON
inline const std::string INVALID_JSON =
    R"({"name": "add", "arguments": {"a": 3, "b":)";

// Model output with tool call with id
inline const std::string TOOL_CALL_WITH_ID =
    R"({"name": "search", "id": "call_123", "arguments": {"query": "test"}})";

// Multi-step: first response calls a tool, second gives final answer
inline const std::string TOOL_CALL_STEP1 =
    R"({"name": "get_time", "arguments": {}})";

inline const std::string FINAL_ANSWER_STEP2 =
    "The current time is 2024-01-01T00:00:00Z.";

// Tool call with wrong argument names
inline const std::string TOOL_CALL_WRONG_ARGS =
    R"({"name": "add", "arguments": {"x": 3, "y": 4}})";

// Tool call with wrong argument types
inline const std::string TOOL_CALL_WRONG_TYPES =
    R"({"name": "add", "arguments": {"a": "three", "b": "four"}})";

} // namespace responses
} // namespace testing
} // namespace zoo
