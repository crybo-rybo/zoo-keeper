#include <gtest/gtest.h>
#include "zoo/engine/template_engine.hpp"
#include "fixtures/template_expectations.hpp"

using namespace zoo;
using namespace zoo::engine;
using namespace zoo::testing::fixtures;

class TemplateEngineTest : public ::testing::Test {
protected:
    std::vector<Message> create_simple_conversation() {
        return {
            Message::system("You are a helpful assistant."),
            Message::user("Hello!"),
            Message::assistant("Hi! How can I help you?")
        };
    }

    std::vector<Message> create_minimal_conversation() {
        return {
            Message::user("What is 2+2?")
        };
    }
};

// ============================================================================
// Llama3 Template Tests
// ============================================================================

TEST_F(TemplateEngineTest, Llama3SimpleConversation) {
    TemplateEngine engine(PromptTemplate::Llama3);
    auto messages = create_simple_conversation();

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    std::string expected = TemplateExpectations::llama3_simple_conversation();
    EXPECT_EQ(*result, expected);
}

TEST_F(TemplateEngineTest, Llama3MinimalConversation) {
    TemplateEngine engine(PromptTemplate::Llama3);
    auto messages = create_minimal_conversation();

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    std::string expected = TemplateExpectations::llama3_minimal_conversation();
    EXPECT_EQ(*result, expected);
}

TEST_F(TemplateEngineTest, Llama3IncludesBOT) {
    TemplateEngine engine(PromptTemplate::Llama3);
    auto messages = create_minimal_conversation();

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // Should start with <|begin_of_text|>
    EXPECT_EQ(result->substr(0, 17), "<|begin_of_text|>");
}

TEST_F(TemplateEngineTest, Llama3HeaderFormat) {
    TemplateEngine engine(PromptTemplate::Llama3);
    std::vector<Message> messages = {Message::user("Test")};

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // Should contain proper header tags
    EXPECT_NE(result->find("<|start_header_id|>user<|end_header_id|>"), std::string::npos);
    EXPECT_NE(result->find("<|eot_id|>"), std::string::npos);
}

TEST_F(TemplateEngineTest, Llama3EndsWithAssistantHeader) {
    TemplateEngine engine(PromptTemplate::Llama3);
    auto messages = create_simple_conversation();

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // When last message is assistant, should NOT add another assistant header
    // But when last is user, SHOULD add assistant header
    std::vector<Message> user_last = {
        Message::user("Hello")
    };

    result = engine.render(user_last);
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("<|start_header_id|>assistant<|end_header_id|>"), std::string::npos);
}

TEST_F(TemplateEngineTest, Llama3MultipleRoles) {
    TemplateEngine engine(PromptTemplate::Llama3);

    std::vector<Message> messages = {
        Message::system("Be helpful."),
        Message::user("Hello"),
        Message::assistant("Hi"),
        Message::user("How are you?"),
        Message::assistant("I'm good!")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // Verify all roles are present
    EXPECT_NE(result->find("system"), std::string::npos);
    EXPECT_NE(result->find("user"), std::string::npos);
    EXPECT_NE(result->find("assistant"), std::string::npos);
}

// ============================================================================
// ChatML Template Tests
// ============================================================================

TEST_F(TemplateEngineTest, ChatMLSimpleConversation) {
    TemplateEngine engine(PromptTemplate::ChatML);
    auto messages = create_simple_conversation();

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    std::string expected = TemplateExpectations::chatml_simple_conversation();
    EXPECT_EQ(*result, expected);
}

TEST_F(TemplateEngineTest, ChatMLMinimalConversation) {
    TemplateEngine engine(PromptTemplate::ChatML);
    auto messages = create_minimal_conversation();

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    std::string expected = TemplateExpectations::chatml_minimal_conversation();
    EXPECT_EQ(*result, expected);
}

TEST_F(TemplateEngineTest, ChatMLFormat) {
    TemplateEngine engine(PromptTemplate::ChatML);
    std::vector<Message> messages = {Message::user("Test")};

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // Should contain proper ChatML tags
    EXPECT_NE(result->find("<|im_start|>user"), std::string::npos);
    EXPECT_NE(result->find("<|im_end|>"), std::string::npos);
}

TEST_F(TemplateEngineTest, ChatMLEndsWithAssistantStart) {
    TemplateEngine engine(PromptTemplate::ChatML);
    std::vector<Message> messages = {Message::user("Hello")};

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // Should end with assistant start tag (ready for response)
    EXPECT_NE(result->find("<|im_start|>assistant"), std::string::npos);
}

TEST_F(TemplateEngineTest, ChatMLMultipleMessages) {
    TemplateEngine engine(PromptTemplate::ChatML);

    std::vector<Message> messages = {
        Message::system("Be concise."),
        Message::user("Hi"),
        Message::assistant("Hello"),
        Message::user("Bye")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // Count occurrences of im_start
    size_t count = 0;
    size_t pos = 0;
    while ((pos = result->find("<|im_start|>", pos)) != std::string::npos) {
        count++;
        pos += 12;
    }

    // 4 messages + 1 for assistant response = 5
    EXPECT_EQ(count, 5);
}

// ============================================================================
// Custom Template Tests
// ============================================================================

TEST_F(TemplateEngineTest, CustomTemplateSimple) {
    std::string custom = "{{role}}: {{content}}\n";
    TemplateEngine engine(PromptTemplate::Custom, custom);

    std::vector<Message> messages = {
        Message::user("Hello"),
        Message::assistant("Hi there")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(*result, "user: Hello\nassistant: Hi there\n");
}

TEST_F(TemplateEngineTest, CustomTemplateMultiplePlaceholders) {
    std::string custom = "[{{role}}] {{content}} [/{{role}}]\n";
    TemplateEngine engine(PromptTemplate::Custom, custom);

    std::vector<Message> messages = {Message::user("Test")};

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(*result, "[user] Test [/user]\n");
}

TEST_F(TemplateEngineTest, CustomTemplateNoPlaceholders) {
    std::string custom = "Static text\n";
    TemplateEngine engine(PromptTemplate::Custom, custom);

    std::vector<Message> messages = {
        Message::user("Message 1"),
        Message::user("Message 2")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(*result, "Static text\nStatic text\n");
}

TEST_F(TemplateEngineTest, CustomTemplateNotProvided) {
    TemplateEngine engine(PromptTemplate::Custom);  // No custom template

    std::vector<Message> messages = {Message::user("Test")};

    auto result = engine.render(messages);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidTemplate);
}

// ============================================================================
// Error Cases
// ============================================================================

TEST_F(TemplateEngineTest, EmptyMessageList) {
    TemplateEngine engine(PromptTemplate::Llama3);
    std::vector<Message> empty;

    auto result = engine.render(empty);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

TEST_F(TemplateEngineTest, ChatMLEmptyMessageList) {
    TemplateEngine engine(PromptTemplate::ChatML);
    std::vector<Message> empty;

    auto result = engine.render(empty);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

TEST_F(TemplateEngineTest, CustomEmptyMessageList) {
    TemplateEngine engine(PromptTemplate::Custom, "{{role}}: {{content}}");
    std::vector<Message> empty;

    auto result = engine.render(empty);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidMessageSequence);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(TemplateEngineTest, EmptyMessageContent) {
    TemplateEngine engine(PromptTemplate::Llama3);
    std::vector<Message> messages = {Message::user("")};

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->length(), 0);  // Should still have template markup
}

TEST_F(TemplateEngineTest, VeryLongMessage) {
    TemplateEngine engine(PromptTemplate::Llama3);
    std::string long_content(10000, 'x');
    std::vector<Message> messages = {Message::user(long_content)};

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find(long_content), std::string::npos);
}

TEST_F(TemplateEngineTest, SpecialCharactersInContent) {
    TemplateEngine engine(PromptTemplate::Llama3);
    std::vector<Message> messages = {
        Message::user("Special chars: \n\t<>&\"'")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());
    // Should preserve special characters
    EXPECT_NE(result->find("Special chars: \n\t<>&\"'"), std::string::npos);
}

TEST_F(TemplateEngineTest, ToolMessages) {
    TemplateEngine engine(PromptTemplate::Llama3);
    std::vector<Message> messages = {
        Message::user("Call a tool"),
        Message::tool("Tool result", "call_123")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("tool"), std::string::npos);
}

// ============================================================================
// Template Type Tests
// ============================================================================

TEST_F(TemplateEngineTest, GetTemplate) {
    TemplateEngine llama3(PromptTemplate::Llama3);
    EXPECT_EQ(llama3.get_template(), PromptTemplate::Llama3);

    TemplateEngine chatml(PromptTemplate::ChatML);
    EXPECT_EQ(chatml.get_template(), PromptTemplate::ChatML);

    TemplateEngine custom(PromptTemplate::Custom, "test");
    EXPECT_EQ(custom.get_template(), PromptTemplate::Custom);
}

// ============================================================================
// Content Preservation Tests
// ============================================================================

TEST_F(TemplateEngineTest, Llama3PreservesMessageContent) {
    TemplateEngine engine(PromptTemplate::Llama3);
    std::vector<Message> messages = {
        Message::system("System prompt here"),
        Message::user("User query here"),
        Message::assistant("Assistant response here")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    EXPECT_NE(result->find("System prompt here"), std::string::npos);
    EXPECT_NE(result->find("User query here"), std::string::npos);
    EXPECT_NE(result->find("Assistant response here"), std::string::npos);
}

TEST_F(TemplateEngineTest, ChatMLPreservesMessageContent) {
    TemplateEngine engine(PromptTemplate::ChatML);
    std::vector<Message> messages = {
        Message::user("Question?"),
        Message::assistant("Answer!")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    EXPECT_NE(result->find("Question?"), std::string::npos);
    EXPECT_NE(result->find("Answer!"), std::string::npos);
}

TEST_F(TemplateEngineTest, CustomPreservesMessageContent) {
    TemplateEngine engine(PromptTemplate::Custom, "{{content}}\n---\n");
    std::vector<Message> messages = {
        Message::user("First"),
        Message::assistant("Second")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(*result, "First\n---\nSecond\n---\n");
}

// ============================================================================
// Conversation Ending Tests
// ============================================================================

TEST_F(TemplateEngineTest, Llama3LastMessageAssistant) {
    TemplateEngine engine(PromptTemplate::Llama3);
    std::vector<Message> messages = {
        Message::user("Hello"),
        Message::assistant("Hi there!")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // When last message is assistant, should NOT add another assistant header
    // Count assistant headers
    size_t pos = 0;
    int count = 0;
    while ((pos = result->find("<|start_header_id|>assistant<|end_header_id|>", pos)) != std::string::npos) {
        count++;
        pos += 45;
    }
    EXPECT_EQ(count, 1);  // Only one assistant header (from the message)
}

TEST_F(TemplateEngineTest, ChatMLLastMessageAssistant) {
    TemplateEngine engine(PromptTemplate::ChatML);
    std::vector<Message> messages = {
        Message::user("Hello"),
        Message::assistant("Hi there!")
    };

    auto result = engine.render(messages);
    ASSERT_TRUE(result.has_value());

    // When last message is assistant, should still add assistant start for continuation
    // Actually, check the implementation - it adds assistant start when last is NOT assistant
    size_t pos = 0;
    int count = 0;
    while ((pos = result->find("<|im_start|>assistant", pos)) != std::string::npos) {
        count++;
        pos += 21;
    }
    EXPECT_EQ(count, 1);  // Only the one from the message itself
}
