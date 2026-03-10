/**
 * @file types.hpp
 * @brief Core value types shared across the zoo-keeper model, tools, and agent layers.
 */

#pragma once

#include <chrono>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace zoo {

/**
 * @brief Enumerates the conversation roles supported by the runtime.
 */
enum class Role {
    System,    ///< Instructional message that primes the assistant.
    User,      ///< End-user input awaiting a response.
    Assistant, ///< Assistant-authored response content.
    Tool       ///< Structured tool result associated with a prior tool call.
};

/**
 * @brief Returns the lowercase role label expected by llama.cpp chat templates.
 *
 * @param role Role to stringify.
 * @return Stable lowercase role name, or `"unknown"` for unexpected values.
 */
[[nodiscard]] inline const char* role_to_string(Role role) noexcept {
    switch (role) {
    case Role::System:
        return "system";
    case Role::User:
        return "user";
    case Role::Assistant:
        return "assistant";
    case Role::Tool:
        return "tool";
    }
    return "unknown";
}

/**
 * @brief Represents one message in the conversation history.
 */
struct Message {
    Role role;           ///< Speaker role associated with the message.
    std::string content; ///< Raw message content passed to the model.
    std::optional<std::string>
        tool_call_id; ///< Tool call correlation identifier for tool responses.

    /**
     * @brief Creates a system message.
     *
     * @param content Instructional content to prepend to the conversation.
     * @return A message tagged with `Role::System`.
     */
    static Message system(std::string content) {
        return Message{Role::System, std::move(content), std::nullopt};
    }

    /**
     * @brief Creates a user message.
     *
     * @param content User-authored content.
     * @return A message tagged with `Role::User`.
     */
    static Message user(std::string content) {
        return Message{Role::User, std::move(content), std::nullopt};
    }

    /**
     * @brief Creates an assistant message.
     *
     * @param content Assistant-authored content.
     * @return A message tagged with `Role::Assistant`.
     */
    static Message assistant(std::string content) {
        return Message{Role::Assistant, std::move(content), std::nullopt};
    }

    /**
     * @brief Creates a tool response message.
     *
     * @param content Serialized tool output.
     * @param tool_call_id Identifier of the originating tool call.
     * @return A message tagged with `Role::Tool`.
     */
    static Message tool(std::string content, std::string tool_call_id) {
        return Message{Role::Tool, std::move(content), std::move(tool_call_id)};
    }

    /// Compares two messages including role, content, and tool call correlation.
    bool operator==(const Message& other) const = default;
};

/**
 * @brief Categorizes runtime failures surfaced by zoo-keeper.
 */
enum class ErrorCode {
    // Configuration errors (100-199)
    InvalidConfig = 100,         ///< Generic configuration validation failure.
    InvalidModelPath = 101,      ///< Missing or invalid model path.
    InvalidContextSize = 102,    ///< Invalid context window configuration.
    InvalidSamplingParams = 103, ///< Invalid sampling configuration.

    // Backend errors (200-299)
    BackendInitFailed = 200,     ///< llama.cpp backend initialization failed.
    ModelLoadFailed = 201,       ///< Model weights could not be loaded.
    ContextCreationFailed = 202, ///< llama.cpp context creation failed.
    InferenceFailed = 203,       ///< Decode or generation failed.
    TokenizationFailed = 204,    ///< Text could not be tokenized.

    // Engine errors (300-399)
    ContextWindowExceeded = 300,  ///< Prompt or generation exceeded available context.
    InvalidMessageSequence = 301, ///< Conversation roles violate sequencing rules.
    TemplateRenderFailed = 302,   ///< Chat template rendering failed.

    // Runtime errors (400-499)
    AgentNotRunning = 400,  ///< A request targeted an agent that is not accepting work.
    RequestCancelled = 401, ///< The caller cancelled the request before completion.
    RequestTimeout = 402,   ///< The request exceeded its allowed runtime.
    QueueFull = 403,        ///< The request queue could not accept another item.

    // Tool errors (500-599)
    ToolNotFound = 500,         ///< A referenced tool name is not registered.
    ToolExecutionFailed = 501,  ///< A tool handler threw or returned an execution failure.
    InvalidToolSignature = 502, ///< Registered tool metadata does not match its callable signature.
    ToolRetriesExhausted = 503, ///< Validation retries for a tool call were exhausted.
    ToolLoopLimitReached = 504, ///< The agent exceeded its tool-iteration budget.
    InvalidToolSchema = 505,    ///< A manually supplied tool schema uses an unsupported construct.
    ToolValidationFailed = 506, ///< A parsed tool call failed schema-based argument validation.

    Unknown = 999 ///< Fallback code for uncategorized failures.
};

/**
 * @brief Rich error payload returned by fallible zoo-keeper operations.
 */
struct Error {
    ErrorCode code;                     ///< Machine-readable error category.
    std::string message;                ///< Human-readable error summary.
    std::optional<std::string> context; ///< Optional contextual detail for diagnostics.

    /**
     * @brief Constructs an error payload.
     *
     * @param code Error category.
     * @param message Human-readable description of the failure.
     * @param context Optional additional diagnostic context.
     */
    Error(ErrorCode code, std::string message, std::optional<std::string> context = std::nullopt)
        : code(code), message(std::move(message)), context(std::move(context)) {}

    /**
     * @brief Formats the error as a single log-friendly string.
     *
     * @return String containing the numeric code, message, and optional context.
     */
    std::string to_string() const {
        std::string result = "[" + std::to_string(static_cast<int>(code)) + "] " + message;
        if (context.has_value()) {
            result += " | Context: " + *context;
        }
        return result;
    }

    /// Compares two errors including code, message, and optional context.
    bool operator==(const Error& other) const = default;
};

/**
 * @brief Convenience alias for operations that either return `T` or a `zoo::Error`.
 */
template <typename T> using Expected = std::expected<T, Error>;

/**
 * @brief Sampling parameters used to configure text generation.
 */
struct SamplingParams {
    float temperature = 0.7f;    ///< Softmax temperature. `0.0f` behaves greedily.
    float top_p = 0.9f;          ///< Nucleus sampling probability cutoff in `[0.0, 1.0]`.
    int top_k = 40;              ///< Limits candidate selection to the top-k tokens.
    float repeat_penalty = 1.1f; ///< Penalty applied to recently seen tokens.
    int repeat_last_n = 64;      ///< Number of trailing tokens considered for repetition penalty.
    int seed = -1; ///< RNG seed. Negative values delegate seeding to the current time.

    /**
     * @brief Validates the sampling configuration.
     *
     * @return Empty success on valid parameters, or an `InvalidSamplingParams`
     *         error describing the first invalid field.
     */
    Expected<void> validate() const {
        if (temperature < 0.0f) {
            return std::unexpected(
                Error{ErrorCode::InvalidSamplingParams,
                      "temperature must be >= 0.0 (got " + std::to_string(temperature) + ")"});
        }
        if (top_p < 0.0f || top_p > 1.0f) {
            return std::unexpected(
                Error{ErrorCode::InvalidSamplingParams,
                      "top_p must be in [0.0, 1.0] (got " + std::to_string(top_p) + ")"});
        }
        if (top_k < 1) {
            return std::unexpected(Error{ErrorCode::InvalidSamplingParams,
                                         "top_k must be >= 1 (got " + std::to_string(top_k) + ")"});
        }
        if (repeat_penalty < 0.0f) {
            return std::unexpected(
                Error{ErrorCode::InvalidSamplingParams, "repeat_penalty must be >= 0.0 (got " +
                                                            std::to_string(repeat_penalty) + ")"});
        }
        if (repeat_last_n < 0) {
            return std::unexpected(
                Error{ErrorCode::InvalidSamplingParams,
                      "repeat_last_n must be >= 0 (got " + std::to_string(repeat_last_n) + ")"});
        }
        return {};
    }

    /// Compares two sampling configurations field-by-field.
    bool operator==(const SamplingParams& other) const = default;
};

/**
 * @brief Controls whether token streaming should continue.
 */
enum class TokenAction {
    Continue, ///< Continue decoding and streaming tokens.
    Stop      ///< Stop generation after the current token callback.
};

/**
 * @brief Callback invoked for streamed token fragments.
 */
using TokenCallback = std::function<TokenAction(std::string_view)>;

/**
 * @brief Callback queried by generation loops to decide whether work should stop.
 */
using CancellationCallback = std::function<bool()>;

/**
 * @brief Runtime configuration used to load a model and create an agent.
 */
struct Config {
    std::string model_path;  ///< Filesystem path to the GGUF model.
    int context_size = 8192; ///< Requested context window size in tokens.
    int n_gpu_layers =
        0; ///< Number of layers to offload to GPU. Defaults to CPU-only for portability.
    bool use_mmap = true;   ///< Whether to memory-map the model file.
    bool use_mlock = false; ///< Whether to lock model pages in memory.

    SamplingParams sampling; ///< Sampling behavior for generation.

    int max_tokens = -1; ///< Maximum number of generated tokens, or `-1` for no explicit cap.
    std::vector<std::string>
        stop_sequences; ///< User-defined stop sequences applied during generation.

    std::optional<std::string>
        system_prompt;                ///< Optional system prompt inserted at the start of history.
    size_t max_history_messages = 64; ///< Maximum number of non-system history messages retained.

    size_t request_queue_capacity = 64; ///< Maximum number of pending requests accepted by `Agent`.

    int max_tool_iterations = 5; ///< Maximum detect/execute/respond iterations per request.
    int max_tool_retries = 2;    ///< Maximum validation retries for malformed tool calls.

    std::optional<TokenCallback>
        on_token; ///< Optional default token callback used by direct model generation.

    /**
     * @brief Validates the configuration before model initialization.
     *
     * @return Empty success on valid configuration, or the first encountered
     *         validation error.
     */
    Expected<void> validate() const {
        if (model_path.empty()) {
            return std::unexpected(
                Error{ErrorCode::InvalidModelPath, "Model path cannot be empty"});
        }
        if (context_size <= 0) {
            return std::unexpected(
                Error{ErrorCode::InvalidContextSize, "Context size must be positive"});
        }
        if (max_tokens == 0 || (max_tokens < 0 && max_tokens != -1)) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "max_tokens must be positive or -1 (unlimited)"});
        }
        if (max_history_messages == 0) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "max_history_messages must be >= 1"});
        }
        if (auto result = sampling.validate(); !result) {
            return result;
        }
        if (max_tool_iterations < 1) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "max_tool_iterations must be >= 1 (got " +
                                                    std::to_string(max_tool_iterations) + ")"});
        }
        if (max_tool_retries < 0) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "max_tool_retries must be >= 0 (got " +
                                                    std::to_string(max_tool_retries) + ")"});
        }
        return {};
    }

    /**
     * @brief Compares two configurations, excluding transient callbacks.
     *
     * The `on_token` callback is intentionally omitted because function objects
     * are not meaningfully comparable.
     */
    bool operator==(const Config& other) const {
        return model_path == other.model_path && context_size == other.context_size &&
               n_gpu_layers == other.n_gpu_layers && use_mmap == other.use_mmap &&
               use_mlock == other.use_mlock && sampling == other.sampling &&
               max_tokens == other.max_tokens && stop_sequences == other.stop_sequences &&
               system_prompt == other.system_prompt &&
               max_history_messages == other.max_history_messages &&
               request_queue_capacity == other.request_queue_capacity &&
               max_tool_iterations == other.max_tool_iterations &&
               max_tool_retries == other.max_tool_retries;
    }
};

/**
 * @brief Token accounting for a completed generation.
 */
struct TokenUsage {
    int prompt_tokens = 0;     ///< Tokens consumed by the rendered prompt.
    int completion_tokens = 0; ///< Tokens emitted during generation.
    int total_tokens = 0;      ///< Sum of prompt and completion tokens.

    /// Compares two token-usage snapshots.
    bool operator==(const TokenUsage& other) const = default;
};

/**
 * @brief Timing and throughput metrics captured for a response.
 */
struct Metrics {
    std::chrono::milliseconds latency_ms{0};             ///< End-to-end request latency.
    std::chrono::milliseconds time_to_first_token_ms{0}; ///< Delay until the first streamed token.
    double tokens_per_second = 0.0; ///< Throughput measured after the first token arrives.

    /// Compares two metric snapshots.
    bool operator==(const Metrics& other) const = default;
};

/**
 * @brief Outcome recorded for one attempted tool invocation.
 */
enum class ToolInvocationStatus {
    Succeeded,        ///< Tool arguments validated and the handler returned a result.
    ValidationFailed, ///< Parsed arguments did not satisfy the registered schema.
    ExecutionFailed   ///< The handler returned or raised an execution failure.
};

/**
 * @brief Structured record of one attempted tool call during agent execution.
 */
struct ToolInvocation {
    std::string id; ///< Correlation identifier parsed from or derived for the tool call.
    std::string name; ///< Registered tool name the model attempted to invoke.
    std::string arguments_json; ///< Serialized arguments exactly as parsed from model output.
    ToolInvocationStatus status = ToolInvocationStatus::Succeeded; ///< Final outcome category.
    std::optional<std::string>
        result_json; ///< Serialized handler result when execution succeeded.
    std::optional<Error>
        error; ///< Validation or execution error details when the attempt did not succeed.

    /// Compares two tool invocation records field-by-field.
    bool operator==(const ToolInvocation& other) const = default;
};

/**
 * @brief Final response returned by model or agent generation.
 */
struct Response {
    std::string text; ///< Assistant-visible response text.
    TokenUsage usage; ///< Prompt and completion token usage.
    Metrics metrics;  ///< Latency and throughput data.
    std::vector<ToolInvocation>
        tool_invocations; ///< Explicit tool invocation attempts recorded during the agentic loop.

    /// Compares two responses field-by-field.
    bool operator==(const Response& other) const = default;
};

/**
 * @brief Monotonic identifier assigned to queued agent requests.
 */
using RequestId = uint64_t;

/**
 * @brief Validates whether a new message role can be appended to `messages`.
 *
 * Rules enforced today:
 * - a tool response cannot be the first message,
 * - system messages are only allowed at the start of history,
 * - non-tool roles may not repeat consecutively.
 *
 * @param messages Existing conversation history.
 * @param role Role to validate as the next appended message.
 * @return Empty success when the role may be appended, otherwise an
 *         `InvalidMessageSequence` error.
 */
[[nodiscard]] inline Expected<void> validate_role_sequence(const std::vector<Message>& messages,
                                                           Role role) {
    if (messages.empty()) {
        if (role == Role::Tool) {
            return std::unexpected(Error{ErrorCode::InvalidMessageSequence,
                                         "First message cannot be a tool response"});
        }
        return {};
    }

    if (role == Role::System) {
        return std::unexpected(Error{ErrorCode::InvalidMessageSequence,
                                     "System message only allowed at the beginning"});
    }

    const Role last_role = messages.back().role;
    if (role == last_role && role != Role::Tool) {
        return std::unexpected(
            Error{ErrorCode::InvalidMessageSequence,
                  "Cannot have consecutive messages with the same role (except Tool)"});
    }

    return {};
}

} // namespace zoo
