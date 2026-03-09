/**
 * @file test_error_recovery.cpp
 * @brief Unit tests for tool argument validation and retry tracking.
 */

#include "fixtures/tool_definitions.hpp"
#include "zoo/tools/validation.hpp"
#include <gtest/gtest.h>

using namespace zoo::testing::tools;

/// Shared fixture that pre-registers common tools for validation tests.
class ErrorRecoveryTest : public ::testing::Test {
  protected:
    zoo::tools::ToolRegistry registry;
    zoo::tools::ErrorRecovery recovery{2};

    void SetUp() override {
        (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);
        (void)registry.register_tool("greet", "Greet someone", {"name"}, greet);
        (void)registry.register_tool("multiply", "Multiply doubles", {"a", "b"}, multiply);
    }
};

TEST_F(ErrorRecoveryTest, ValidArgsPass) {
    zoo::tools::ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", 3}, {"b", 4}};
    EXPECT_TRUE(recovery.validate_args(tc, registry).empty());
}

TEST_F(ErrorRecoveryTest, ValidStringArgs) {
    zoo::tools::ToolCall tc;
    tc.name = "greet";
    tc.arguments = {{"name", "Alice"}};
    EXPECT_TRUE(recovery.validate_args(tc, registry).empty());
}

TEST_F(ErrorRecoveryTest, MissingRequiredArg) {
    zoo::tools::ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", 3}};
    auto error = recovery.validate_args(tc, registry);
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("Missing"), std::string::npos);
}

TEST_F(ErrorRecoveryTest, WrongArgType) {
    zoo::tools::ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", "three"}, {"b", 4}};
    auto error = recovery.validate_args(tc, registry);
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("wrong type"), std::string::npos);
}

TEST_F(ErrorRecoveryTest, UnknownTool) {
    zoo::tools::ToolCall tc;
    tc.name = "nonexistent";
    tc.arguments = {};
    auto error = recovery.validate_args(tc, registry);
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("not found"), std::string::npos);
}

TEST_F(ErrorRecoveryTest, RetryTracking) {
    EXPECT_TRUE(recovery.can_retry("add"));
    EXPECT_EQ(recovery.get_retry_count("add"), 0);

    recovery.record_retry("add");
    EXPECT_EQ(recovery.get_retry_count("add"), 1);
    EXPECT_TRUE(recovery.can_retry("add"));

    recovery.record_retry("add");
    EXPECT_EQ(recovery.get_retry_count("add"), 2);
    EXPECT_FALSE(recovery.can_retry("add"));
}

TEST_F(ErrorRecoveryTest, Reset) {
    recovery.record_retry("add");
    recovery.record_retry("add");
    EXPECT_FALSE(recovery.can_retry("add"));
    recovery.reset();
    EXPECT_TRUE(recovery.can_retry("add"));
    EXPECT_EQ(recovery.get_retry_count("add"), 0);
}

TEST_F(ErrorRecoveryTest, MaxRetries) {
    EXPECT_EQ(recovery.max_retries(), 2);
}
