/**
 * @file test_sentinel_stream_filter.cpp
 * @brief Unit tests for grammar-mode sentinel stream suppression.
 */

#include "zoo/internal/tools/sentinel_stream_filter.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using zoo::tools::SentinelStreamFilter;

namespace {

std::string simulate_tokens(const std::vector<std::string>& tokens) {
    SentinelStreamFilter filter;
    std::string visible;
    for (const auto& token : tokens) {
        visible += filter.consume(token);
    }
    visible += filter.finalize();
    return visible;
}

} // namespace

TEST(SentinelStreamFilterTest, SuppressesSentinelWrappedToolCall) {
    std::vector<std::string> tokens = {
        "<tool_call>", "{\"name\": \"add\", \"arguments\": {\"a\": 3, \"b\": 4}}", "</tool_call>"};

    EXPECT_TRUE(simulate_tokens(tokens).empty());
}

TEST(SentinelStreamFilterTest, PreservesVisibleTextBeforeSentinel) {
    std::vector<std::string> tokens = {
        "I'll use a tool: ", "<tool_call>",
        "{\"name\": \"multiply\", \"arguments\": {\"a\": 6.0, \"b\": 7.0}}", "</tool_call>"};

    EXPECT_EQ(simulate_tokens(tokens), "I'll use a tool: ");
}

TEST(SentinelStreamFilterTest, PreservesVisiblePrefixInSameToken) {
    std::vector<std::string> tokens = {
        "Prefix <tool_call>{\"name\": \"ping\", \"arguments\": {}}</tool_call>"};

    EXPECT_EQ(simulate_tokens(tokens), "Prefix ");
}

TEST(SentinelStreamFilterTest, SentinelSplitAcrossTokensIsSuppressed) {
    std::string sentinel = "<tool_call>";
    std::vector<std::string> tokens;
    for (char c : sentinel) {
        tokens.emplace_back(1, c);
    }
    tokens.push_back("{\"name\": \"ping\", \"arguments\": {}}");

    EXPECT_TRUE(simulate_tokens(tokens).empty());
}

TEST(SentinelStreamFilterTest, OverlappingLessThanStillMatchesSentinel) {
    std::vector<std::string> tokens = {"<<tool_call>", "{\"name\": \"ping\", \"arguments\": {}}",
                                       "</tool_call>"};

    EXPECT_EQ(simulate_tokens(tokens), "<");
}

TEST(SentinelStreamFilterTest, NonSentinelAngleBracketPassesThrough) {
    std::vector<std::string> tokens = {"2 < 3 is true."};

    EXPECT_EQ(simulate_tokens(tokens), "2 < 3 is true.");
}

TEST(SentinelStreamFilterTest, PartialSentinelAtEosIsFlushed) {
    std::vector<std::string> tokens = {"Hello ", "<tool_ca"};

    EXPECT_EQ(simulate_tokens(tokens), "Hello <tool_ca");
}

TEST(SentinelStreamFilterTest, WhitespaceAfterSentinelIsSuppressed) {
    std::vector<std::string> tokens = {
        "<tool_call>", " \n", "{\"name\": \"get_time\", \"arguments\": {}}", "\n</tool_call>"};

    EXPECT_TRUE(simulate_tokens(tokens).empty());
}
