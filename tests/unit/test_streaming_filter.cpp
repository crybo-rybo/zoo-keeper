/**
 * @file test_streaming_filter.cpp
 * @brief Unit tests for tool-call streaming trigger detection.
 */

#include "core/stream_filter.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <common.h>
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

TEST(StreamingFilterTest, SplitWordTriggerPrefixIsBufferedUntilDecision) {
    zoo::core::ToolCallWordTriggerFilter filter({"[TOOL_CALLS]"});

    EXPECT_EQ(filter.consume("[TOO"), "");
    EXPECT_FALSE(filter.suppressing());

    EXPECT_EQ(filter.consume("L_CALLS]"), "");
    EXPECT_TRUE(filter.suppressing());
}

TEST(StreamingFilterTest, FalsePositiveWordPrefixFlushesAsVisibleText) {
    zoo::core::ToolCallWordTriggerFilter filter({"<function="});

    EXPECT_EQ(filter.consume("<fun"), "");
    EXPECT_FALSE(filter.suppressing());

    EXPECT_EQ(filter.consume("ny value"), "<funny value");
    EXPECT_FALSE(filter.suppressing());
}

TEST(StreamingFilterTest, VisibleTextBeforeTriggerStillStreams) {
    zoo::core::ToolCallWordTriggerFilter filter({"<function="});

    EXPECT_EQ(filter.consume("hello <fun"), "hello ");
    EXPECT_FALSE(filter.suppressing());

    EXPECT_EQ(filter.consume("ction="), "");
    EXPECT_TRUE(filter.suppressing());
}

TEST(StreamingFilterTest, FinalizeFlushesIncompleteBufferedPrefix) {
    zoo::core::ToolCallWordTriggerFilter filter({"[TOOL_CALLS]"});

    EXPECT_EQ(filter.consume("[TOO"), "");
    EXPECT_EQ(filter.finalize(), "[TOO");
    EXPECT_FALSE(filter.suppressing());
}

} // namespace
