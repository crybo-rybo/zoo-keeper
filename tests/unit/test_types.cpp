/**
 * @file test_types.cpp
 * @brief Unit tests for shared core value types and validation helpers.
 */

#include "log.hpp"
#include "zoo/core/json.hpp"
#include "zoo/core/types.hpp"
#include "zoo/log.hpp"
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

class TempDir {
  public:
    TempDir() {
        const auto unique =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() / ("zoo-types-tests-" + unique);
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

  private:
    std::filesystem::path path_;
};

template <typename T>
concept CanValidateRoleSequence =
    requires(const T& history) { zoo::validate_role_sequence(history, zoo::Role::User); };

struct UnsupportedHistory {
    [[nodiscard]] size_t size() const noexcept {
        return 0;
    }

    [[nodiscard]] int operator[](size_t) const noexcept {
        return 0;
    }
};

static_assert(!CanValidateRoleSequence<UnsupportedHistory>);

} // namespace

TEST(RoleTest, RoleToString) {
    EXPECT_STREQ(zoo::role_to_string(zoo::Role::System), "system");
    EXPECT_STREQ(zoo::role_to_string(zoo::Role::User), "user");
    EXPECT_STREQ(zoo::role_to_string(zoo::Role::Assistant), "assistant");
    EXPECT_STREQ(zoo::role_to_string(zoo::Role::Tool), "tool");
}

TEST(ToolCallTest, OwnedToolCallProducesBorrowedView) {
    zoo::OwnedToolCall call{"call_1", "lookup_weather", R"({"city":"Boston"})"};
    const auto view = call.view();

    EXPECT_EQ(view.id, "call_1");
    EXPECT_EQ(view.name, "lookup_weather");
    EXPECT_EQ(view.arguments_json, R"({"city":"Boston"})");
}

TEST(ToolCallTest, ToolCallSpanSupportsBorrowedAndOwnedStorage) {
    const std::array<zoo::ToolCallView, 1> borrowed = {
        zoo::ToolCallView{"call_1", "lookup_weather", R"({"city":"Boston"})"}};
    const zoo::ToolCallSpan borrowed_span{std::span<const zoo::ToolCallView>(borrowed)};

    EXPECT_EQ(borrowed_span.size(), 1u);
    EXPECT_EQ(borrowed_span[0].name, "lookup_weather");

    const std::array<zoo::OwnedToolCall, 1> owned = {
        zoo::OwnedToolCall{"call_2", "sum", R"({"a":1})"}};
    const zoo::ToolCallSpan owned_span{std::span<const zoo::OwnedToolCall>(owned)};

    EXPECT_EQ(owned_span.size(), 1u);
    EXPECT_EQ(owned_span[0].id, "call_2");
}

TEST(ConversationViewTest, SupportsBorrowedAndOwnedStorage) {
    const std::array<zoo::MessageView, 1> borrowed = {zoo::MessageView{zoo::Role::User, "hello"}};
    const zoo::ConversationView borrowed_view{std::span<const zoo::MessageView>(borrowed)};

    EXPECT_EQ(borrowed_view.size(), 1u);
    EXPECT_EQ(borrowed_view[0].role, zoo::Role::User);
    EXPECT_EQ(borrowed_view[0].content, "hello");

    const std::array<zoo::OwnedMessage, 1> owned = {zoo::OwnedMessage::assistant("reply")};
    const zoo::ConversationView owned_view{std::span<const zoo::OwnedMessage>(owned)};

    EXPECT_EQ(owned_view.size(), 1u);
    EXPECT_EQ(owned_view[0].role, zoo::Role::Assistant);
    EXPECT_EQ(owned_view[0].content, "reply");
}

TEST(MessageTest, FactoryMethodsCreateOwnedMessages) {
    auto sys = zoo::OwnedMessage::system("System message");
    EXPECT_EQ(sys.role, zoo::Role::System);
    EXPECT_EQ(sys.content, "System message");
    EXPECT_TRUE(sys.tool_call_id.empty());

    auto user = zoo::OwnedMessage::user("User message");
    EXPECT_EQ(user.role, zoo::Role::User);

    auto assistant = zoo::OwnedMessage::assistant("Assistant message");
    EXPECT_EQ(assistant.role, zoo::Role::Assistant);

    auto tool = zoo::OwnedMessage::tool("Tool result", "call_123");
    EXPECT_EQ(tool.role, zoo::Role::Tool);
    EXPECT_EQ(tool.tool_call_id, "call_123");
}

TEST(MessageTest, ViewReflectsOwnedMessageWithoutCopyingFields) {
    zoo::OwnedMessage message = zoo::OwnedMessage::assistant_with_tool_calls(
        "Visible text", {zoo::OwnedToolCall{"call_1", "sum", R"({"a":1})"}});

    const zoo::MessageView view = message.view();

    EXPECT_EQ(view.role(), zoo::Role::Assistant);
    EXPECT_EQ(view.content(), "Visible text");
    ASSERT_EQ(view.tool_calls().size(), 1u);
    EXPECT_EQ(view.tool_calls()[0].name, "sum");
}

TEST(MessageTest, BorrowedMessageCanBeMaterialized) {
    const std::array<zoo::ToolCallView, 1> tool_calls = {
        zoo::ToolCallView{"call_2", "echo", R"({"text":"hello"})"}};
    const zoo::MessageView view{zoo::Role::Assistant, "hello", std::span(tool_calls)};

    const auto owned = zoo::OwnedMessage::from_view(view);

    EXPECT_EQ(owned.role, zoo::Role::Assistant);
    EXPECT_EQ(owned.content, "hello");
    ASSERT_EQ(owned.tool_calls.size(), 1u);
    EXPECT_EQ(owned.tool_calls[0].arguments_json, R"({"text":"hello"})");
}

TEST(ConversationViewTest, SupportsBorrowedAndOwnedStorage) {
    const std::array<zoo::MessageView, 2> borrowed = {
        zoo::MessageView{zoo::Role::System, "Be concise."},
        zoo::MessageView{zoo::Role::User, "Say hello."}};
    zoo::ConversationView borrowed_view{std::span<const zoo::MessageView>(borrowed)};

    EXPECT_EQ(borrowed_view.size(), 2u);
    EXPECT_EQ(borrowed_view[1].content(), "Say hello.");

    zoo::HistorySnapshot snapshot{
        {zoo::OwnedMessage::system("Prompt"), zoo::OwnedMessage::user("Ping")}};
    const auto owned_view = snapshot.view();

    EXPECT_EQ(owned_view.size(), 2u);
    EXPECT_EQ(owned_view[0].content(), "Prompt");
    EXPECT_EQ(owned_view[1].role(), zoo::Role::User);
}

TEST(ErrorTest, Construction) {
    zoo::Error err(zoo::ErrorCode::InvalidConfig, "Test error");
    EXPECT_EQ(err.code, zoo::ErrorCode::InvalidConfig);
    EXPECT_EQ(err.message, "Test error");
    EXPECT_FALSE(err.context.has_value());
}

TEST(ErrorTest, ToString) {
    zoo::Error err(zoo::ErrorCode::InvalidConfig, "Configuration is invalid");
    std::string str = err.to_string();
    EXPECT_NE(str.find("100"), std::string::npos);
    EXPECT_NE(str.find("Configuration is invalid"), std::string::npos);
}

TEST(ErrorTest, ToStringWithContext) {
    zoo::Error err(zoo::ErrorCode::InferenceFailed, "Inference failed", "Out of memory");
    std::string str = err.to_string();
    EXPECT_NE(str.find("Out of memory"), std::string::npos);
}

TEST(ErrorTest, Expected) {
    zoo::Expected<int> success = 42;
    EXPECT_TRUE(success.has_value());
    EXPECT_EQ(*success, 42);

    zoo::Expected<int> failure = std::unexpected(zoo::Error{zoo::ErrorCode::Unknown, "Failed"});
    EXPECT_FALSE(failure.has_value());
    EXPECT_EQ(failure.error().code, zoo::ErrorCode::Unknown);
}

TEST(SamplingParamsTest, DefaultsAndValidation) {
    zoo::SamplingParams params;
    EXPECT_FLOAT_EQ(params.temperature, 0.7f);
    EXPECT_FLOAT_EQ(params.top_p, 0.9f);
    EXPECT_EQ(params.top_k, 40);
    EXPECT_TRUE(params.validate().has_value());
}

TEST(SamplingParamsTest, ValidationRejectsInvalidFields) {
    zoo::SamplingParams params;
    params.temperature = -0.1f;
    EXPECT_FALSE(params.validate().has_value());

    params = {};
    params.top_p = 1.1f;
    EXPECT_FALSE(params.validate().has_value());

    params = {};
    params.top_k = 0;
    EXPECT_FALSE(params.validate().has_value());
}

TEST(SamplingParamsJsonTest, RoundTripsDefaultValues) {
    const zoo::SamplingParams params;
    const nlohmann::json json = params;

    EXPECT_EQ(json.at("temperature"), 0.7f);
    EXPECT_EQ(json.at("repeat_last_n"), 64);

    const auto round_trip = json.get<zoo::SamplingParams>();
    EXPECT_EQ(round_trip, params);
}

TEST(SamplingParamsJsonTest, RejectsUnknownKeys) {
    const nlohmann::json json = {{"temperature", 0.7f}, {"unsupported", true}};
    EXPECT_THROW((void)json.get<zoo::SamplingParams>(), std::invalid_argument);
}

TEST(ModelConfigTest, ValidationSuccess) {
    zoo::ModelConfig config;
    config.model_path = "/dev/null";
    EXPECT_TRUE(config.validate().has_value());
}

TEST(ModelConfigTest, ValidationRejectsNonExistentPath) {
    zoo::ModelConfig config;
    config.model_path = "/nonexistent/path/model.gguf";
    auto result = config.validate();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelPath);
}

TEST(ModelConfigTest, ValidationUsesNonThrowingFilesystemCheck) {
    TempDir temp_dir;
    const auto loop_path = temp_dir.path() / "loop";
    std::error_code ec;
    std::filesystem::create_directory_symlink(loop_path, loop_path, ec);
    if (ec) {
        GTEST_SKIP() << "Could not create symlink loop: " << ec.message();
    }

    zoo::ModelConfig config;
    config.model_path = loop_path.string();

    zoo::Expected<void> result;
    EXPECT_NO_THROW(result = config.validate());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelPath);
}

TEST(ModelConfigTest, ValidationRejectsBadFields) {
    zoo::ModelConfig config;
    EXPECT_FALSE(config.validate().has_value());

    config.model_path = "/dev/null";
    config.context_size = 0;
    EXPECT_FALSE(config.validate().has_value());
}

TEST(AgentConfigTest, DefaultsAndValidation) {
    zoo::AgentConfig config;
    EXPECT_EQ(config.max_history_messages, 64u);
    EXPECT_EQ(config.request_queue_capacity, 64u);
    EXPECT_EQ(config.max_tool_iterations, 5);
    EXPECT_EQ(config.max_tool_retries, 2);
    EXPECT_TRUE(config.validate().has_value());
}

TEST(AgentConfigTest, ValidationRejectsInvalidFields) {
    zoo::AgentConfig config;
    config.request_queue_capacity = 0;
    EXPECT_FALSE(config.validate().has_value());

    config = {};
    config.max_history_messages = 0;
    EXPECT_FALSE(config.validate().has_value());

    config = {};
    config.max_tool_iterations = 0;
    EXPECT_FALSE(config.validate().has_value());
}

TEST(GenerationOptionsTest, DefaultsAndValidation) {
    zoo::GenerationOptions options;
    EXPECT_EQ(options.max_tokens, -1);
    EXPECT_FALSE(options.record_tool_trace);
    EXPECT_TRUE(options.validate().has_value());
}

TEST(GenerationOptionsTest, ValidationRejectsZeroMaxTokens) {
    zoo::GenerationOptions options;
    options.max_tokens = 0;
    auto result = options.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

TEST(ModelConfigJsonTest, RoundTripsSerializableFields) {
    zoo::ModelConfig config;
    config.model_path = "/tmp/model.gguf";
    config.context_size = 4096;
    config.n_gpu_layers = 12;
    config.use_mmap = false;
    config.use_mlock = true;

    const nlohmann::json json = config;
    const auto round_trip = json.get<zoo::ModelConfig>();
    EXPECT_EQ(round_trip, config);
}

TEST(ModelConfigJsonTest, RejectsMissingModelPath) {
    const nlohmann::json json = {{"context_size", 4096}};
    EXPECT_THROW((void)json.get<zoo::ModelConfig>(), std::invalid_argument);
}

TEST(AgentConfigJsonTest, RoundTripsSerializableFields) {
    zoo::AgentConfig config;
    config.max_history_messages = 8;
    config.request_queue_capacity = 4;
    config.max_tool_iterations = 3;
    config.max_tool_retries = 1;

    const nlohmann::json json = config;
    const auto round_trip = json.get<zoo::AgentConfig>();
    EXPECT_EQ(round_trip, config);
}

TEST(GenerationOptionsJsonTest, RoundTripsSerializableFields) {
    zoo::GenerationOptions options;
    options.sampling.temperature = 0.2f;
    options.sampling.top_p = 0.8f;
    options.sampling.top_k = 12;
    options.max_tokens = 256;
    options.stop_sequences = {"</tool_call>", "User:"};
    options.record_tool_trace = true;

    const nlohmann::json json = options;
    EXPECT_EQ(json.at("sampling").at("top_k"), 12);
    EXPECT_EQ(json.at("record_tool_trace"), true);

    const auto round_trip = json.get<zoo::GenerationOptions>();
    EXPECT_EQ(round_trip, options);
}

TEST(RoleValidationTest, EmptyHistoryAcceptsUser) {
    std::vector<zoo::OwnedMessage> history;
    EXPECT_TRUE(zoo::validate_role_sequence(history, zoo::Role::User).has_value());
}

TEST(RoleValidationTest, EmptyHistoryAcceptsSystem) {
    std::vector<zoo::OwnedMessage> history;
    EXPECT_TRUE(zoo::validate_role_sequence(history, zoo::Role::System).has_value());
}

TEST(RoleValidationTest, EmptyHistoryRejectsTool) {
    std::vector<zoo::OwnedMessage> history;
    auto result = zoo::validate_role_sequence(history, zoo::Role::Tool);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidMessageSequence);
}

TEST(RoleValidationTest, SystemAllowedAfterNonSystemMessage) {
    std::vector<zoo::OwnedMessage> history = {zoo::OwnedMessage::user("Hello")};
    EXPECT_TRUE(zoo::validate_role_sequence(history, zoo::Role::System).has_value());
}

TEST(RoleValidationTest, ConsecutiveSystemMessagesFails) {
    std::vector<zoo::OwnedMessage> history = {zoo::OwnedMessage::system("Initial prompt")};
    auto result = zoo::validate_role_sequence(history, zoo::Role::System);
    EXPECT_FALSE(result.has_value());
}

TEST(RoleValidationTest, ConsecutiveSameRoleFails) {
    std::vector<zoo::OwnedMessage> history = {zoo::OwnedMessage::user("Hello")};
    auto result = zoo::validate_role_sequence(history, zoo::Role::User);
    EXPECT_FALSE(result.has_value());
}

TEST(RoleValidationTest, ConsecutiveToolAllowed) {
    std::vector<zoo::OwnedMessage> history = {zoo::OwnedMessage::user("Hello"),
                                              zoo::OwnedMessage::assistant("I'll use tools"),
                                              zoo::OwnedMessage::tool("result1", "id1")};
    EXPECT_TRUE(zoo::validate_role_sequence(history, zoo::Role::Tool).has_value());
}

TEST(RoleValidationTest, AcceptsHistorySnapshotAndConversationView) {
    zoo::HistorySnapshot snapshot{{zoo::OwnedMessage::user("Hello")}};
    EXPECT_TRUE(zoo::validate_role_sequence(snapshot, zoo::Role::Assistant).has_value());

    const auto view = snapshot.view();
    auto result = zoo::validate_role_sequence(view, zoo::Role::User);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidMessageSequence);
}

TEST(LoggingTest, CallbackReceivesFormattedZooLogs) {
    struct LogRecord {
        zoo::LogLevel level = zoo::LogLevel::Debug;
        std::string message;
        int calls = 0;
    } record;

    zoo::set_log_callback(
        [](zoo::LogLevel level, const char* message, void* user_data) {
            auto& target = *static_cast<LogRecord*>(user_data);
            target.level = level;
            target.message = message;
            ++target.calls;
        },
        &record);

    ZOO_LOG("warn", "downloaded %d bytes", 42);
    zoo::reset_log_callback();

    EXPECT_EQ(record.calls, 1);
    EXPECT_EQ(record.level, zoo::LogLevel::Warning);
    EXPECT_EQ(record.message, "downloaded 42 bytes");
}

TEST(TokenUsageTest, Defaults) {
    zoo::TokenUsage usage;
    EXPECT_EQ(usage.prompt_tokens, 0);
    EXPECT_EQ(usage.completion_tokens, 0);
    EXPECT_EQ(usage.total_tokens, 0);
}

TEST(ToolInvocationTest, DefaultsAndStatusStrings) {
    zoo::ToolInvocation invocation;
    EXPECT_TRUE(invocation.id.empty());
    EXPECT_TRUE(invocation.name.empty());
    EXPECT_TRUE(invocation.arguments_json.empty());
    EXPECT_EQ(invocation.status, zoo::ToolInvocationStatus::Succeeded);
    EXPECT_STREQ(zoo::to_string(zoo::ToolInvocationStatus::Succeeded), "succeeded");
    EXPECT_STREQ(zoo::to_string(zoo::ToolInvocationStatus::ValidationFailed), "validation_failed");
    EXPECT_STREQ(zoo::to_string(zoo::ToolInvocationStatus::ExecutionFailed), "execution_failed");
}

TEST(TextResponseTest, Defaults) {
    zoo::TextResponse response;
    EXPECT_TRUE(response.text.empty());
    EXPECT_FALSE(response.tool_trace.has_value());
}

TEST(ExtractionResponseTest, Defaults) {
    zoo::ExtractionResponse response;
    EXPECT_TRUE(response.text.empty());
    EXPECT_TRUE(response.data.is_null());
    EXPECT_FALSE(response.tool_trace.has_value());
}
