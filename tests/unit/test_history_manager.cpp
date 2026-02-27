#include <gtest/gtest.h>
#include "zoo/engine/history_manager.hpp"

using namespace zoo;
using namespace zoo::engine;

class HistoryManagerTest : public ::testing::Test {
protected:
    HistoryManagerTest() : manager(8192) {}

    void SetUp() override {
        manager.clear();
    }

    HistoryManager manager;
};

// ============================================================================
// Basic Operations
// ============================================================================

TEST_F(HistoryManagerTest, InitialState) {
    EXPECT_EQ(manager.get_messages().size(), 0);
    EXPECT_EQ(manager.get_estimated_tokens(), 0);
    EXPECT_EQ(manager.get_context_size(), 8192);
    EXPECT_FALSE(manager.is_context_exceeded());
}

TEST_F(HistoryManagerTest, AddMessage) {
    auto msg = Message::user("Hello, world!");
    auto result = manager.add_message(msg);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(manager.get_messages().size(), 1);
    EXPECT_EQ(manager.get_messages()[0].content, "Hello, world!");
    EXPECT_GT(manager.get_estimated_tokens(), 0);
}

TEST_F(HistoryManagerTest, AddMultipleMessages) {
    auto msg1 = Message::user("Hello");
    auto msg2 = Message::assistant("Hi there!");
    auto msg3 = Message::user("How are you?");

    EXPECT_TRUE(manager.add_message(msg1).has_value());
    EXPECT_TRUE(manager.add_message(msg2).has_value());
    EXPECT_TRUE(manager.add_message(msg3).has_value());

    EXPECT_EQ(manager.get_messages().size(), 3);
}

TEST_F(HistoryManagerTest, Clear) {
    (void)manager.add_message(Message::user("Test 1"));
    (void)manager.add_message(Message::assistant("Test 2"));

    EXPECT_GT(manager.get_messages().size(), 0);
    EXPECT_GT(manager.get_estimated_tokens(), 0);

    manager.clear();

    EXPECT_EQ(manager.get_messages().size(), 0);
    EXPECT_EQ(manager.get_estimated_tokens(), 0);
}

// ============================================================================
// System Prompt Tests
// ============================================================================

TEST_F(HistoryManagerTest, SetSystemPrompt) {
    manager.set_system_prompt("You are a helpful assistant.");

    EXPECT_EQ(manager.get_messages().size(), 1);
    EXPECT_EQ(manager.get_messages()[0].role, Role::System);
    EXPECT_EQ(manager.get_messages()[0].content, "You are a helpful assistant.");
    EXPECT_GT(manager.get_estimated_tokens(), 0);
}

TEST_F(HistoryManagerTest, ReplaceSystemPrompt) {
    manager.set_system_prompt("First prompt.");
    int first_tokens = manager.get_estimated_tokens();

    manager.set_system_prompt("Second prompt, much longer for testing purposes.");

    EXPECT_EQ(manager.get_messages().size(), 1);
    EXPECT_EQ(manager.get_messages()[0].content, "Second prompt, much longer for testing purposes.");
    EXPECT_GT(manager.get_estimated_tokens(), first_tokens);
}

TEST_F(HistoryManagerTest, SetSystemPromptAfterMessages) {
    (void)manager.add_message(Message::user("Hello"));

    manager.set_system_prompt("System prompt added after user message.");

    // System prompt should be at the beginning
    ASSERT_EQ(manager.get_messages().size(), 2);
    EXPECT_EQ(manager.get_messages()[0].role, Role::System);
    EXPECT_EQ(manager.get_messages()[1].role, Role::User);
}

// ============================================================================
// Role Validation Tests
// ============================================================================

TEST_F(HistoryManagerTest, FirstMessageCannotBeTool) {
    auto tool_msg = Message::tool("Result", "call_123");
    auto result = manager.add_message(tool_msg);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

TEST_F(HistoryManagerTest, FirstMessageCanBeUser) {
    auto msg = Message::user("Hello");
    auto result = manager.add_message(msg);

    EXPECT_TRUE(result.has_value());
}

TEST_F(HistoryManagerTest, FirstMessageCanBeSystem) {
    auto msg = Message::system("System prompt");
    auto result = manager.add_message(msg);

    EXPECT_TRUE(result.has_value());
}

TEST_F(HistoryManagerTest, FirstMessageCanBeAssistant) {
    auto msg = Message::assistant("Hello");
    auto result = manager.add_message(msg);

    EXPECT_TRUE(result.has_value());
}

TEST_F(HistoryManagerTest, SystemMessageOnlyAtStart) {
    (void)manager.add_message(Message::user("Hello"));

    auto sys_msg = Message::system("Late system message");
    auto result = manager.add_message(sys_msg);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

TEST_F(HistoryManagerTest, NoConsecutiveSameRoles) {
    (void)manager.add_message(Message::user("First user message"));

    auto result = manager.add_message(Message::user("Second user message"));

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

TEST_F(HistoryManagerTest, AlternatingRolesAllowed) {
    EXPECT_TRUE(manager.add_message(Message::user("User 1")).has_value());
    EXPECT_TRUE(manager.add_message(Message::assistant("Assistant 1")).has_value());
    EXPECT_TRUE(manager.add_message(Message::user("User 2")).has_value());
    EXPECT_TRUE(manager.add_message(Message::assistant("Assistant 2")).has_value());

    EXPECT_EQ(manager.get_messages().size(), 4);
}

TEST_F(HistoryManagerTest, ConsecutiveToolMessagesAllowed) {
    (void)manager.add_message(Message::user("User message"));
    (void)manager.add_message(Message::assistant("Calling tools..."));

    // Multiple tool responses are allowed
    EXPECT_TRUE(manager.add_message(Message::tool("Result 1", "call_1")).has_value());
    EXPECT_TRUE(manager.add_message(Message::tool("Result 2", "call_2")).has_value());

    EXPECT_EQ(manager.get_messages().size(), 4);
}

// ============================================================================
// Token Estimation Tests
// ============================================================================

TEST_F(HistoryManagerTest, TokenEstimation) {
    // Rough heuristic: 4 chars ≈ 1 token, plus template_overhead_per_message (default: 8)
    auto msg = Message::user("1234");  // ~1 content token + 8 overhead = ~9 total
    (void)manager.add_message(msg);

    // Should be at least 1 (content) + overhead (>=0)
    EXPECT_GE(manager.get_estimated_tokens(), 1);
    // With default overhead of 8, upper bound is content_tokens + 8 overhead
    // "1234" -> 1 content token + 8 overhead = 9, allow some slack for heuristic variance
    EXPECT_LE(manager.get_estimated_tokens(), 20);
}

TEST_F(HistoryManagerTest, TokenEstimationAccumulates) {
    (void)manager.add_message(Message::user("1234"));  // ~1 token
    int after_first = manager.get_estimated_tokens();

    (void)manager.add_message(Message::assistant("12345678"));  // ~2 tokens
    int after_second = manager.get_estimated_tokens();

    EXPECT_GT(after_second, after_first);
}

TEST_F(HistoryManagerTest, TokenEstimationAfterClear) {
    (void)manager.add_message(Message::user("Some message that takes tokens"));
    EXPECT_GT(manager.get_estimated_tokens(), 0);

    manager.clear();
    EXPECT_EQ(manager.get_estimated_tokens(), 0);
}

TEST_F(HistoryManagerTest, MinimumOneToken) {
    // Even a single character should count as at least 1 token
    (void)manager.add_message(Message::user("x"));
    EXPECT_GE(manager.get_estimated_tokens(), 1);
}

// ============================================================================
// Context Window Tests
// ============================================================================

TEST_F(HistoryManagerTest, ContextNotExceededInitially) {
    EXPECT_FALSE(manager.is_context_exceeded());
}

TEST_F(HistoryManagerTest, ContextNotExceededSmallMessages) {
    for (int i = 0; i < 10; ++i) {
        if (i % 2 == 0) {
            (void)manager.add_message(Message::user("Short message"));
        } else {
            (void)manager.add_message(Message::assistant("Short reply"));
        }
    }

    EXPECT_FALSE(manager.is_context_exceeded());
}

TEST_F(HistoryManagerTest, ContextExceeded) {
    HistoryManager small_manager(100);  // Very small context

    // Add a very long message
    std::string long_content(1000, 'x');  // 1000 chars ≈ 250 tokens
    (void)small_manager.add_message(Message::user(long_content));

    EXPECT_TRUE(small_manager.is_context_exceeded());
}

TEST_F(HistoryManagerTest, ContextExceededMultipleMessages) {
    HistoryManager small_manager(50);  // Small context

    for (int i = 0; i < 20; ++i) {
        if (i % 2 == 0) {
            (void)small_manager.add_message(Message::user("Some longer message"));  // 19 chars = 4 tokens
        } else {
            (void)small_manager.add_message(Message::assistant("Some reply"));  // 10 chars = 2 tokens
        }
    }

    // 10 * 4 + 10 * 2 = 60 tokens > 50
    EXPECT_TRUE(small_manager.is_context_exceeded());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(HistoryManagerTest, EmptyMessage) {
    auto result = manager.add_message(Message::user(""));

    EXPECT_TRUE(result.has_value());
    EXPECT_GE(manager.get_estimated_tokens(), 1);  // Minimum 1 token
}

TEST_F(HistoryManagerTest, VeryLongMessage) {
    std::string very_long(10000, 'a');
    auto result = manager.add_message(Message::user(very_long));

    EXPECT_TRUE(result.has_value());
    EXPECT_GT(manager.get_estimated_tokens(), 2000);  // ~2500 tokens expected
}

TEST_F(HistoryManagerTest, GetMessagesReturnsCopy) {
    (void)manager.add_message(Message::user("Test"));

    auto messages = manager.get_messages();
    EXPECT_EQ(messages.size(), 1);

    // Verify the returned value is an independent copy: mutating it does not
    // affect the manager's internal history.
    messages.clear();
    EXPECT_EQ(manager.get_messages().size(), 1);
}

// ============================================================================
// Complex Scenarios
// ============================================================================

TEST_F(HistoryManagerTest, TypicalConversation) {
    manager.set_system_prompt("You are a helpful AI assistant.");

    EXPECT_TRUE(manager.add_message(Message::user("What is the capital of France?")).has_value());
    EXPECT_TRUE(manager.add_message(Message::assistant("The capital of France is Paris.")).has_value());
    EXPECT_TRUE(manager.add_message(Message::user("What is its population?")).has_value());
    EXPECT_TRUE(manager.add_message(Message::assistant("Paris has approximately 2.2 million residents.")).has_value());

    EXPECT_EQ(manager.get_messages().size(), 5);  // System + 4 messages
    EXPECT_EQ(manager.get_messages()[0].role, Role::System);
    EXPECT_FALSE(manager.is_context_exceeded());
}

TEST_F(HistoryManagerTest, ConversationWithToolCalls) {
    (void)manager.add_message(Message::user("What's the weather?"));
    (void)manager.add_message(Message::assistant("Let me check the weather API."));
    (void)manager.add_message(Message::tool("Temperature: 72F, Sunny", "weather_call_1"));
    (void)manager.add_message(Message::assistant("It's 72 degrees and sunny!"));

    EXPECT_EQ(manager.get_messages().size(), 4);
}

TEST_F(HistoryManagerTest, MultipleToolCallsInSequence) {
    (void)manager.add_message(Message::user("Check weather and time"));
    (void)manager.add_message(Message::assistant("Checking both..."));

    // Multiple tool responses
    EXPECT_TRUE(manager.add_message(Message::tool("72F, Sunny", "weather_call")).has_value());
    EXPECT_TRUE(manager.add_message(Message::tool("3:45 PM", "time_call")).has_value());

    (void)manager.add_message(Message::assistant("It's 72F and sunny, and the time is 3:45 PM."));

    EXPECT_EQ(manager.get_messages().size(), 5);
}

// ============================================================================
// Remove Last Message (Error Recovery)
// ============================================================================

TEST_F(HistoryManagerTest, RemoveLastMessage) {
    (void)manager.add_message(Message::user("Hello"));
    (void)manager.add_message(Message::assistant("Hi there"));

    EXPECT_EQ(manager.get_messages().size(), 2);
    EXPECT_TRUE(manager.remove_last_message());
    EXPECT_EQ(manager.get_messages().size(), 1);
    EXPECT_EQ(manager.get_messages().back().role, Role::User);
}

TEST_F(HistoryManagerTest, RemoveLastMessageUpdatesTokenEstimate) {
    (void)manager.add_message(Message::user("Hello world test message"));
    int tokens_after_one = manager.get_estimated_tokens();

    (void)manager.add_message(Message::assistant("Response text here"));
    int tokens_after_two = manager.get_estimated_tokens();
    EXPECT_GT(tokens_after_two, tokens_after_one);

    manager.remove_last_message();
    EXPECT_EQ(manager.get_estimated_tokens(), tokens_after_one);
}

TEST_F(HistoryManagerTest, RemoveLastMessageFromEmptyHistory) {
    EXPECT_FALSE(manager.remove_last_message());
}

TEST_F(HistoryManagerTest, AddMessageMoveOverload) {
    auto msg = Message::user("Hello, world!");
    std::string content_copy = msg.content;

    auto result = manager.add_message(std::move(msg));

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(manager.get_messages().size(), 1);
    EXPECT_EQ(manager.get_messages()[0].content, content_copy);
    EXPECT_GT(manager.get_estimated_tokens(), 0);
    // After move, original msg.content may be empty (moved-from state)
}

TEST_F(HistoryManagerTest, AddMessageMoveValidation) {
    // Move overload should still validate role sequences
    (void)manager.add_message(Message::user("Hello"));

    auto msg = Message::user("Another user message");
    auto result = manager.add_message(std::move(msg));

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

TEST_F(HistoryManagerTest, RemoveLastMessageAllowsRetry) {
    // Simulate: user message added, generation fails, rollback, retry
    (void)manager.add_message(Message::user("First question"));
    (void)manager.add_message(Message::assistant("Answer"));
    (void)manager.add_message(Message::user("Second question"));

    // Simulate generation failure - rollback user message
    manager.remove_last_message();
    EXPECT_EQ(manager.get_messages().size(), 2);
    EXPECT_EQ(manager.get_messages().back().role, Role::Assistant);

    // Retry should succeed (not rejected as consecutive User)
    auto result = manager.add_message(Message::user("Second question retry"));
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(manager.get_messages().size(), 3);
}

// ============================================================================
// Template Overhead Tests (Issue #40)
// ============================================================================

TEST(HistoryManagerOverheadTest, PerMessageOverheadIsAddedToEstimate) {
    // With overhead=0 vs overhead=8, the difference per message should be 8 tokens
    HistoryManager manager_no_overhead(8192, nullptr, 0);
    HistoryManager manager_with_overhead(8192, nullptr, 8);

    (void)manager_no_overhead.add_message(Message::user("Hello world"));
    (void)manager_with_overhead.add_message(Message::user("Hello world"));

    int tokens_no_overhead = manager_no_overhead.get_estimated_tokens();
    int tokens_with_overhead = manager_with_overhead.get_estimated_tokens();

    // The difference should be exactly the overhead value
    EXPECT_EQ(tokens_with_overhead - tokens_no_overhead, 8);
}

TEST(HistoryManagerOverheadTest, OverheadAccumulatesAcrossMessages) {
    const int overhead = 5;
    HistoryManager manager_no_overhead(8192, nullptr, 0);
    HistoryManager manager_with_overhead(8192, nullptr, overhead);

    (void)manager_no_overhead.add_message(Message::user("Hello"));
    (void)manager_no_overhead.add_message(Message::assistant("Hi"));
    (void)manager_no_overhead.add_message(Message::user("How are you?"));

    (void)manager_with_overhead.add_message(Message::user("Hello"));
    (void)manager_with_overhead.add_message(Message::assistant("Hi"));
    (void)manager_with_overhead.add_message(Message::user("How are you?"));

    int tokens_no_overhead = manager_no_overhead.get_estimated_tokens();
    int tokens_with_overhead = manager_with_overhead.get_estimated_tokens();

    // 3 messages × 5 overhead = 15 additional tokens
    EXPECT_EQ(tokens_with_overhead - tokens_no_overhead, 3 * overhead);
}

TEST(HistoryManagerOverheadTest, RemoveLastMessageSubtractsOverhead) {
    const int overhead = 10;
    HistoryManager mgr(8192, nullptr, overhead);

    (void)mgr.add_message(Message::user("First"));
    int tokens_after_one = mgr.get_estimated_tokens();

    (void)mgr.add_message(Message::assistant("Second"));
    int tokens_after_two = mgr.get_estimated_tokens();
    EXPECT_GT(tokens_after_two, tokens_after_one);

    // Removing the last message should restore the estimate to tokens_after_one
    mgr.remove_last_message();
    EXPECT_EQ(mgr.get_estimated_tokens(), tokens_after_one);
}

TEST(HistoryManagerOverheadTest, SystemPromptOverheadIsAccountedFor) {
    const int overhead = 6;
    HistoryManager manager_no_overhead(8192, nullptr, 0);
    HistoryManager manager_with_overhead(8192, nullptr, overhead);

    manager_no_overhead.set_system_prompt("You are helpful.");
    manager_with_overhead.set_system_prompt("You are helpful.");

    int tokens_no_overhead = manager_no_overhead.get_estimated_tokens();
    int tokens_with_overhead = manager_with_overhead.get_estimated_tokens();

    EXPECT_EQ(tokens_with_overhead - tokens_no_overhead, overhead);
}

TEST(HistoryManagerOverheadTest, SystemPromptReplacementSubtractsOldOverhead) {
    const int overhead = 7;
    HistoryManager mgr(8192, nullptr, overhead);

    mgr.set_system_prompt("Short");
    int tokens_short = mgr.get_estimated_tokens();

    // Replace with a longer prompt — overhead should only be counted once
    mgr.set_system_prompt("Much longer system prompt for testing purposes");
    int tokens_long = mgr.get_estimated_tokens();

    // Tokens should increase (longer content), but overhead should still be 1x
    EXPECT_GT(tokens_long, tokens_short);
}

// ============================================================================
// sync_token_estimate Tests (Issue #40)
// ============================================================================

TEST(HistoryManagerSyncTest, SyncTokenEstimateUpdatesInternalCount) {
    HistoryManager mgr(8192);

    (void)mgr.add_message(Message::user("Hello"));
    int estimated = mgr.get_estimated_tokens();

    // Sync with a known actual value
    const int actual_kv_usage = 42;
    mgr.sync_token_estimate(actual_kv_usage);

    EXPECT_EQ(mgr.get_estimated_tokens(), actual_kv_usage);
    (void)estimated; // suppress unused warning
}

TEST(HistoryManagerSyncTest, SyncTokenEstimateIgnoresZero) {
    HistoryManager mgr(8192);

    (void)mgr.add_message(Message::user("Hello"));
    int estimated = mgr.get_estimated_tokens();
    EXPECT_GT(estimated, 0);

    // Syncing with 0 should be a no-op
    mgr.sync_token_estimate(0);
    EXPECT_EQ(mgr.get_estimated_tokens(), estimated);
}

TEST(HistoryManagerSyncTest, SyncTokenEstimateIgnoresNegative) {
    HistoryManager mgr(8192);

    (void)mgr.add_message(Message::user("Hello"));
    int estimated = mgr.get_estimated_tokens();

    // Syncing with negative should be a no-op
    mgr.sync_token_estimate(-5);
    EXPECT_EQ(mgr.get_estimated_tokens(), estimated);
}

TEST(HistoryManagerSyncTest, SyncCalibratesToActualUsage) {
    // After sync, is_context_exceeded() should reflect the synced value
    HistoryManager mgr(100);

    (void)mgr.add_message(Message::user("Hello"));
    EXPECT_FALSE(mgr.is_context_exceeded());

    // Sync with a value that exceeds context
    mgr.sync_token_estimate(150);
    EXPECT_TRUE(mgr.is_context_exceeded());

    // Sync back below context limit
    mgr.sync_token_estimate(50);
    EXPECT_FALSE(mgr.is_context_exceeded());
}
