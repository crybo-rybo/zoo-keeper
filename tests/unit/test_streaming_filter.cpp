/**
 * @file test_streaming_filter.cpp
 * @brief Unit tests for tool-call streaming trigger detection.
 */

#include "zoo/internal/core/stream_filter.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include "../../extern/llama.cpp/common/common.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace {

TEST(StreamingFilterTest, NoTriggersNeverDetects) {
    std::vector<common_grammar_trigger> triggers;
    EXPECT_FALSE(zoo::core::is_tool_trigger_detected("any text", triggers));
}

TEST(StreamingFilterTest, WordTriggerDetectedInText) {
    std::vector<common_grammar_trigger> triggers;
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
    EXPECT_TRUE(zoo::core::is_tool_trigger_detected("Hello [TOOL_CALLS] data", triggers));
}

TEST(StreamingFilterTest, WordTriggerNotYetPresent) {
    std::vector<common_grammar_trigger> triggers;
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
    EXPECT_FALSE(zoo::core::is_tool_trigger_detected("Hello world", triggers));
}

TEST(StreamingFilterTest, MultipleTriggers) {
    std::vector<common_grammar_trigger> triggers;
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_use>"});
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
    EXPECT_TRUE(zoo::core::is_tool_trigger_detected("text [TOOL_CALLS] more", triggers));
}

TEST(StreamingFilterTest, EmptyTriggerValueSkipped) {
    std::vector<common_grammar_trigger> triggers;
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, ""});
    EXPECT_FALSE(zoo::core::is_tool_trigger_detected("any text", triggers));
}

TEST(StreamingFilterTest, PatternTriggerMatchesRegex) {
    std::vector<common_grammar_trigger> triggers;
    // Regex pattern similar to what DeepSeek/Qwen use.
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, R"(\s*<\|tool_call_start\|>\s*\[)"});
    EXPECT_TRUE(zoo::core::is_tool_trigger_detected(" <|tool_call_start|> [", triggers));
    EXPECT_FALSE(zoo::core::is_tool_trigger_detected("Hello world", triggers));
}

TEST(StreamingFilterTest, PatternFullTriggerMatchesEntireText) {
    std::vector<common_grammar_trigger> triggers;
    // PATTERN_FULL must match the entire accumulated text.
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, R"(.*<\|tool_call_start\|>.*)"});
    EXPECT_TRUE(
        zoo::core::is_tool_trigger_detected("some text <|tool_call_start|> rest", triggers));
    EXPECT_FALSE(zoo::core::is_tool_trigger_detected("no trigger here", triggers));
}

TEST(StreamingFilterTest, TokenTriggerIsSkipped) {
    std::vector<common_grammar_trigger> triggers;
    // TOKEN triggers operate at token-id level, not text. Should be skipped.
    triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN, ""});
    EXPECT_FALSE(zoo::core::is_tool_trigger_detected("any text at all", triggers));
}

} // namespace
