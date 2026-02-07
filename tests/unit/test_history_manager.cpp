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
    manager.add_message(Message::user("Test 1"));
    manager.add_message(Message::assistant("Test 2"));

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
    manager.add_message(Message::user("Hello"));

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
    manager.add_message(Message::user("Hello"));

    auto sys_msg = Message::system("Late system message");
    auto result = manager.add_message(sys_msg);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

TEST_F(HistoryManagerTest, NoConsecutiveSameRoles) {
    manager.add_message(Message::user("First user message"));

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
    manager.add_message(Message::user("User message"));
    manager.add_message(Message::assistant("Calling tools..."));

    // Multiple tool responses are allowed
    EXPECT_TRUE(manager.add_message(Message::tool("Result 1", "call_1")).has_value());
    EXPECT_TRUE(manager.add_message(Message::tool("Result 2", "call_2")).has_value());

    EXPECT_EQ(manager.get_messages().size(), 4);
}

// ============================================================================
// Token Estimation Tests
// ============================================================================

TEST_F(HistoryManagerTest, TokenEstimation) {
    // Rough heuristic: 4 chars ≈ 1 token
    auto msg = Message::user("1234");  // Should be ~1 token
    manager.add_message(msg);

    // Actual estimation may vary slightly, but should be close
    EXPECT_GE(manager.get_estimated_tokens(), 1);
    EXPECT_LE(manager.get_estimated_tokens(), 2);
}

TEST_F(HistoryManagerTest, TokenEstimationAccumulates) {
    manager.add_message(Message::user("1234"));  // ~1 token
    int after_first = manager.get_estimated_tokens();

    manager.add_message(Message::assistant("12345678"));  // ~2 tokens
    int after_second = manager.get_estimated_tokens();

    EXPECT_GT(after_second, after_first);
}

TEST_F(HistoryManagerTest, TokenEstimationAfterClear) {
    manager.add_message(Message::user("Some message that takes tokens"));
    EXPECT_GT(manager.get_estimated_tokens(), 0);

    manager.clear();
    EXPECT_EQ(manager.get_estimated_tokens(), 0);
}

TEST_F(HistoryManagerTest, MinimumOneToken) {
    // Even a single character should count as at least 1 token
    manager.add_message(Message::user("x"));
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
            manager.add_message(Message::user("Short message"));
        } else {
            manager.add_message(Message::assistant("Short reply"));
        }
    }

    EXPECT_FALSE(manager.is_context_exceeded());
}

TEST_F(HistoryManagerTest, ContextExceeded) {
    HistoryManager small_manager(100);  // Very small context

    // Add a very long message
    std::string long_content(1000, 'x');  // 1000 chars ≈ 250 tokens
    small_manager.add_message(Message::user(long_content));

    EXPECT_TRUE(small_manager.is_context_exceeded());
}

TEST_F(HistoryManagerTest, ContextExceededMultipleMessages) {
    HistoryManager small_manager(50);  // Small context

    for (int i = 0; i < 20; ++i) {
        if (i % 2 == 0) {
            small_manager.add_message(Message::user("Some longer message"));  // 19 chars = 4 tokens
        } else {
            small_manager.add_message(Message::assistant("Some reply"));  // 10 chars = 2 tokens
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

TEST_F(HistoryManagerTest, GetMessagesReturnsConst) {
    manager.add_message(Message::user("Test"));

    const auto& messages = manager.get_messages();
    EXPECT_EQ(messages.size(), 1);

    // Verify it's actually const
    static_assert(std::is_const_v<std::remove_reference_t<decltype(messages)>>,
                  "get_messages() should return const reference");
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
    manager.add_message(Message::user("What's the weather?"));
    manager.add_message(Message::assistant("Let me check the weather API."));
    manager.add_message(Message::tool("Temperature: 72F, Sunny", "weather_call_1"));
    manager.add_message(Message::assistant("It's 72 degrees and sunny!"));

    EXPECT_EQ(manager.get_messages().size(), 4);
}

TEST_F(HistoryManagerTest, MultipleToolCallsInSequence) {
    manager.add_message(Message::user("Check weather and time"));
    manager.add_message(Message::assistant("Checking both..."));

    // Multiple tool responses
    EXPECT_TRUE(manager.add_message(Message::tool("72F, Sunny", "weather_call")).has_value());
    EXPECT_TRUE(manager.add_message(Message::tool("3:45 PM", "time_call")).has_value());

    manager.add_message(Message::assistant("It's 72F and sunny, and the time is 3:45 PM."));

    EXPECT_EQ(manager.get_messages().size(), 5);
}

// ============================================================================
// Remove Last Message (Error Recovery)
// ============================================================================

TEST_F(HistoryManagerTest, RemoveLastMessage) {
    manager.add_message(Message::user("Hello"));
    manager.add_message(Message::assistant("Hi there"));

    EXPECT_EQ(manager.get_messages().size(), 2);
    EXPECT_TRUE(manager.remove_last_message());
    EXPECT_EQ(manager.get_messages().size(), 1);
    EXPECT_EQ(manager.get_messages().back().role, Role::User);
}

TEST_F(HistoryManagerTest, RemoveLastMessageUpdatesTokenEstimate) {
    manager.add_message(Message::user("Hello world test message"));
    int tokens_after_one = manager.get_estimated_tokens();

    manager.add_message(Message::assistant("Response text here"));
    int tokens_after_two = manager.get_estimated_tokens();
    EXPECT_GT(tokens_after_two, tokens_after_one);

    manager.remove_last_message();
    EXPECT_EQ(manager.get_estimated_tokens(), tokens_after_one);
}

TEST_F(HistoryManagerTest, RemoveLastMessageFromEmptyHistory) {
    EXPECT_FALSE(manager.remove_last_message());
}

TEST_F(HistoryManagerTest, RemoveLastMessageAllowsRetry) {
    // Simulate: user message added, generation fails, rollback, retry
    manager.add_message(Message::user("First question"));
    manager.add_message(Message::assistant("Answer"));
    manager.add_message(Message::user("Second question"));

    // Simulate generation failure - rollback user message
    manager.remove_last_message();
    EXPECT_EQ(manager.get_messages().size(), 2);
    EXPECT_EQ(manager.get_messages().back().role, Role::Assistant);

    // Retry should succeed (not rejected as consecutive User)
    auto result = manager.add_message(Message::user("Second question retry"));
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(manager.get_messages().size(), 3);
}
