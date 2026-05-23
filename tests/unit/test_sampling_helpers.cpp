/**
 * @file test_sampling_helpers.cpp
 * @brief Unit tests for internal sampler helper utilities.
 */

#include "core/sampling_helpers.hpp"

#include <gtest/gtest.h>

TEST(SamplingHelpersTest, RegexEscapeLeavesPlainTextUnchanged) {
    EXPECT_EQ(zoo::core::detail::regex_escape("hello"), "hello");
}

TEST(SamplingHelpersTest, RegexEscapeEscapesSpecialCharacters) {
    EXPECT_EQ(zoo::core::detail::regex_escape("a.b+c?"), R"(a\.b\+c\?)");
    EXPECT_EQ(zoo::core::detail::regex_escape("[TOOL_CALLS]"), R"(\[TOOL_CALLS\])");
}

TEST(SamplingHelpersTest, RegexEscapeEscapesWhitespace) {
    EXPECT_EQ(zoo::core::detail::regex_escape("a b"), R"(a\ b)");
}

TEST(SamplingHelpersTest, RegexEscapeHandlesEmptyInput) {
    EXPECT_EQ(zoo::core::detail::regex_escape(""), "");
}
