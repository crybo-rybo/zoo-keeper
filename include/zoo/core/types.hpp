#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <functional>
#include <expected>

namespace zoo {

// ============================================================================
// Message Types
// ============================================================================

enum class Role {
    System,
    User,
    Assistant,
    Tool
};

[[nodiscard]] inline const char* role_to_string(Role role) {
    switch (role) {
        case Role::System: return "system";
        case Role::User: return "user";
        case Role::Assistant: return "assistant";
        case Role::Tool: return "tool";
    }
    return "unknown";
}

struct Message {
    Role role;
    std::string content;
    std::optional<std::string> tool_call_id;

    static Message system(std::string content) {
        return Message{Role::System, std::move(content), std::nullopt};
    }

    static Message user(std::string content) {
        return Message{Role::User, std::move(content), std::nullopt};
    }

    static Message assistant(std::string content) {
        return Message{Role::Assistant, std::move(content), std::nullopt};
    }

    static Message tool(std::string content, std::string tool_call_id) {
        return Message{Role::Tool, std::move(content), std::move(tool_call_id)};
    }

    bool operator==(const Message& other) const = default;
};

// ============================================================================
// Error Types
// ============================================================================

enum class ErrorCode {
    // Configuration errors (100-199)
    InvalidConfig = 100,
    InvalidModelPath = 101,
    InvalidContextSize = 102,

    // Backend errors (200-299)
    BackendInitFailed = 200,
    ModelLoadFailed = 201,
    ContextCreationFailed = 202,
    InferenceFailed = 203,
    TokenizationFailed = 204,

    // Engine errors (300-399)
    ContextWindowExceeded = 300,
    InvalidMessageSequence = 301,
    TemplateRenderFailed = 302,

    // Runtime errors (400-499)
    AgentNotRunning = 400,
    RequestCancelled = 401,
    RequestTimeout = 402,
    QueueFull = 403,

    // Tool errors (500-599)
    ToolNotFound = 500,
    ToolExecutionFailed = 501,
    InvalidToolSignature = 502,
    ToolRetriesExhausted = 503,
    ToolLoopLimitReached = 504,

    Unknown = 999
};

struct Error {
    ErrorCode code;
    std::string message;
    std::optional<std::string> context;

    Error(ErrorCode code, std::string message, std::optional<std::string> context = std::nullopt)
        : code(code), message(std::move(message)), context(std::move(context)) {}

    std::string to_string() const {
        std::string result = "[" + std::to_string(static_cast<int>(code)) + "] " + message;
        if (context.has_value()) {
            result += " | Context: " + *context;
        }
        return result;
    }
};

template<typename T>
using Expected = std::expected<T, Error>;

// ============================================================================
// Sampling Configuration
// ============================================================================

struct SamplingParams {
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    int seed = -1;

    bool operator==(const SamplingParams& other) const = default;
};

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    std::string model_path;
    int context_size = 8192;
    int n_gpu_layers = -1;
    bool use_mmap = true;
    bool use_mlock = false;

    SamplingParams sampling;

    int max_tokens = -1;
    std::vector<std::string> stop_sequences;

    std::optional<std::string> system_prompt;

    size_t request_queue_capacity = 0;

    std::optional<TokenCallback> on_token;

    Expected<void> validate() const {
        if (model_path.empty()) {
            return std::unexpected(Error{ErrorCode::InvalidModelPath, "Model path cannot be empty"});
        }
        if (context_size <= 0) {
            return std::unexpected(Error{ErrorCode::InvalidContextSize, "Context size must be positive"});
        }
        if (max_tokens == 0 || (max_tokens < 0 && max_tokens != -1)) {
            return std::unexpected(Error{ErrorCode::InvalidConfig, "max_tokens must be positive or -1 (unlimited)"});
        }
        return {};
    }

    bool operator==(const Config& other) const {
        return model_path == other.model_path &&
               context_size == other.context_size &&
               n_gpu_layers == other.n_gpu_layers &&
               use_mmap == other.use_mmap &&
               use_mlock == other.use_mlock &&
               sampling == other.sampling &&
               max_tokens == other.max_tokens &&
               stop_sequences == other.stop_sequences &&
               system_prompt == other.system_prompt &&
               request_queue_capacity == other.request_queue_capacity;
    }
};

// ============================================================================
// Response Types
// ============================================================================

struct TokenUsage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;

    bool operator==(const TokenUsage& other) const = default;
};

struct Metrics {
    std::chrono::milliseconds latency_ms{0};
    std::chrono::milliseconds time_to_first_token_ms{0};
    double tokens_per_second = 0.0;

    bool operator==(const Metrics& other) const = default;
};

struct Response {
    std::string text;
    TokenUsage usage;
    Metrics metrics;
    std::vector<Message> tool_calls;

    bool operator==(const Response& other) const = default;
};

// ============================================================================
// Token Callback Types
// ============================================================================

enum class TokenAction {
    Continue,
    Stop
};

using TokenCallback = std::function<TokenAction(std::string_view)>;

// ============================================================================
// Request Types
// ============================================================================

using RequestId = uint64_t;

// ============================================================================
// Conversation Validation (pure logic, no model dependency)
// ============================================================================

[[nodiscard]] inline Expected<void> validate_role_sequence(
    const std::vector<Message>& messages, Role role
) {
    if (messages.empty()) {
        if (role == Role::Tool) {
            return std::unexpected(Error{
                ErrorCode::InvalidMessageSequence,
                "First message cannot be a tool response"
            });
        }
        return {};
    }

    if (role == Role::System) {
        return std::unexpected(Error{
            ErrorCode::InvalidMessageSequence,
            "System message only allowed at the beginning"
        });
    }

    const Role last_role = messages.back().role;
    if (role == last_role && role != Role::Tool) {
        return std::unexpected(Error{
            ErrorCode::InvalidMessageSequence,
            "Cannot have consecutive messages with the same role (except Tool)"
        });
    }

    return {};
}

} // namespace zoo
