/**
 * @file types.hpp
 * @brief Core value types shared across the zoo-keeper model, tools, and agent layers.
 */

#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
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
 * @brief Lightweight non-owning callable reference for synchronous hot paths.
 *
 * `FunctionRef` does not extend the lifetime of the callable it references.
 * It is intended only for immediate use by the callee.
 */
template <typename Signature> class FunctionRef;

template <typename Result, typename... Args> class FunctionRef<Result(Args...)> {
  public:
    constexpr FunctionRef() noexcept = default;
    constexpr FunctionRef(std::nullptr_t) noexcept {}

    /// Prevent binding to temporaries that would dangle.
    template <typename Callable>
        requires(!std::same_as<std::remove_cvref_t<Callable>, FunctionRef> &&
                 !std::is_lvalue_reference_v<Callable>)
    FunctionRef(Callable&&) = delete;

    template <typename Callable>
        requires(!std::same_as<std::remove_cvref_t<Callable>, FunctionRef> &&
                 std::is_invocable_r_v<Result, Callable&, Args...>)
    FunctionRef(Callable& callable) noexcept
        : object_(static_cast<void*>(std::addressof(callable))),
          callback_([](void* object, Args... args) -> Result {
              return std::invoke(*static_cast<Callable*>(object), std::forward<Args>(args)...);
          }) {}

    [[nodiscard]] explicit operator bool() const noexcept {
        return callback_ != nullptr;
    }

    Result operator()(Args... args) const {
        assert(callback_ && "FunctionRef called with null callback");
        return callback_(object_, std::forward<Args>(args)...);
    }

  private:
    using Callback = Result (*)(void*, Args...);

    void* object_ = nullptr;
    Callback callback_ = nullptr;
};

/**
 * @brief Structured tool call view attached to assistant messages.
 */
struct ToolCallView {
    std::string_view id;
    std::string_view name;
    std::string_view arguments_json;

    bool operator==(const ToolCallView& other) const = default;
};

/**
 * @brief Owning structured tool call record.
 */
struct OwnedToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;

    [[nodiscard]] ToolCallView view() const noexcept {
        return ToolCallView{id, name, arguments_json};
    }

    [[nodiscard]] static OwnedToolCall from_view(const ToolCallView& view) {
        return OwnedToolCall{
            std::string(view.id),
            std::string(view.name),
            std::string(view.arguments_json),
        };
    }

    bool operator==(const OwnedToolCall& other) const = default;
};

/**
 * @brief Lightweight span over either borrowed or owned tool call metadata.
 */
class ToolCallSpan {
  public:
    constexpr ToolCallSpan() noexcept = default;
    constexpr explicit ToolCallSpan(std::span<const ToolCallView> borrowed) noexcept
        : storage_(borrowed) {}
    constexpr explicit ToolCallSpan(std::span<const OwnedToolCall> owned) noexcept
        : storage_(owned) {}

    [[nodiscard]] size_t size() const noexcept {
        return std::visit([](auto span) { return span.size(); }, storage_);
    }

    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    [[nodiscard]] ToolCallView operator[](size_t index) const noexcept {
        return std::visit(
            [index](auto span) -> ToolCallView {
                using Span = decltype(span);
                if constexpr (std::same_as<Span, std::span<const OwnedToolCall>>) {
                    return span[index].view();
                } else {
                    return span[index];
                }
            },
            storage_);
    }

  private:
    using Borrowed = std::span<const ToolCallView>;
    using Owned = std::span<const OwnedToolCall>;

    std::variant<Borrowed, Owned> storage_{Borrowed{}};
};

/**
 * @brief Non-owning view of one message in a conversation.
 */
class MessageView {
  public:
    constexpr MessageView() noexcept = default;
    constexpr MessageView(Role role, std::string_view content,
                          std::string_view tool_call_id = {}) noexcept
        : role_(role), content_(content), tool_call_id_(tool_call_id) {}
    constexpr MessageView(Role role, std::string_view content,
                          std::span<const ToolCallView> tool_calls) noexcept
        : role_(role), content_(content), tool_calls_(tool_calls) {}
    constexpr MessageView(Role role, std::string_view content, std::string_view tool_call_id,
                          std::span<const ToolCallView> tool_calls) noexcept
        : role_(role), content_(content), tool_call_id_(tool_call_id), tool_calls_(tool_calls) {}
    constexpr MessageView(Role role, std::string_view content,
                          std::span<const OwnedToolCall> tool_calls) noexcept
        : role_(role), content_(content), tool_calls_(tool_calls) {}
    constexpr MessageView(Role role, std::string_view content, std::string_view tool_call_id,
                          std::span<const OwnedToolCall> tool_calls) noexcept
        : role_(role), content_(content), tool_call_id_(tool_call_id), tool_calls_(tool_calls) {}

    [[nodiscard]] Role role() const noexcept {
        return role_;
    }

    [[nodiscard]] std::string_view content() const noexcept {
        return content_;
    }

    [[nodiscard]] std::string_view tool_call_id() const noexcept {
        return tool_call_id_;
    }

    [[nodiscard]] bool has_tool_call_id() const noexcept {
        return !tool_call_id_.empty();
    }

    [[nodiscard]] ToolCallSpan tool_calls() const noexcept {
        return tool_calls_;
    }

    bool operator==(const MessageView& other) const noexcept {
        if (role_ != other.role_ || content_ != other.content_ ||
            tool_call_id_ != other.tool_call_id_) {
            return false;
        }
        if (tool_calls_.size() != other.tool_calls_.size()) {
            return false;
        }
        for (size_t index = 0; index < tool_calls_.size(); ++index) {
            if (!(tool_calls_[index] == other.tool_calls_[index])) {
                return false;
            }
        }
        return true;
    }

  private:
    Role role_ = Role::User;
    std::string_view content_{};
    std::string_view tool_call_id_{};
    ToolCallSpan tool_calls_{};
};

/**
 * @brief Owning message stored in retained history or request snapshots.
 */
struct OwnedMessage {
    Role role = Role::User;                ///< Speaker role associated with the message.
    std::string content;                   ///< Raw message content passed to the model.
    std::string tool_call_id;              ///< Tool call correlation identifier for tool responses.
    std::vector<OwnedToolCall> tool_calls; ///< Structured tool calls for assistant messages.

    [[nodiscard]] MessageView view() const noexcept {
        return MessageView(role, content, tool_call_id, std::span<const OwnedToolCall>(tool_calls));
    }

    [[nodiscard]] static OwnedMessage from_view(const MessageView& view) {
        OwnedMessage owned;
        owned.role = view.role();
        owned.content = std::string(view.content());
        owned.tool_call_id = std::string(view.tool_call_id());
        owned.tool_calls.reserve(view.tool_calls().size());
        for (size_t index = 0; index < view.tool_calls().size(); ++index) {
            owned.tool_calls.push_back(OwnedToolCall::from_view(view.tool_calls()[index]));
        }
        return owned;
    }

    /**
     * @brief Creates a system message.
     */
    [[nodiscard]] static OwnedMessage system(std::string content) {
        return OwnedMessage{Role::System, std::move(content), {}, {}};
    }

    /**
     * @brief Creates a user message.
     */
    [[nodiscard]] static OwnedMessage user(std::string content) {
        return OwnedMessage{Role::User, std::move(content), {}, {}};
    }

    /**
     * @brief Creates an assistant message.
     */
    [[nodiscard]] static OwnedMessage assistant(std::string content) {
        return OwnedMessage{Role::Assistant, std::move(content), {}, {}};
    }

    /**
     * @brief Creates an assistant message with structured tool calls.
     */
    [[nodiscard]] static OwnedMessage assistant_with_tool_calls(std::string content,
                                                                std::vector<OwnedToolCall> calls) {
        return OwnedMessage{Role::Assistant, std::move(content), {}, std::move(calls)};
    }

    /**
     * @brief Creates a tool response message.
     */
    [[nodiscard]] static OwnedMessage tool(std::string content, std::string tool_call_id) {
        return OwnedMessage{Role::Tool, std::move(content), std::move(tool_call_id), {}};
    }

    bool operator==(const OwnedMessage& other) const = default;
};

/// Transitional alias retained for internal code and existing consumers.
using Message = OwnedMessage;
/// Transitional alias retained for internal code and existing consumers.
using ToolCallInfo = OwnedToolCall;

/**
 * @brief Borrowed read-only message sequence used by request-scoped APIs.
 */
class ConversationView {
  public:
    constexpr ConversationView() noexcept = default;
    constexpr explicit ConversationView(std::span<const MessageView> borrowed) noexcept
        : storage_(borrowed) {}
    constexpr explicit ConversationView(std::span<const OwnedMessage> owned) noexcept
        : storage_(owned) {}

    [[nodiscard]] size_t size() const noexcept {
        return std::visit([](auto span) { return span.size(); }, storage_);
    }

    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    [[nodiscard]] MessageView operator[](size_t index) const noexcept {
        return std::visit(
            [index](auto span) -> MessageView {
                using Span = decltype(span);
                if constexpr (std::same_as<Span, std::span<const OwnedMessage>>) {
                    return span[index].view();
                } else {
                    return span[index];
                }
            },
            storage_);
    }

  private:
    using Borrowed = std::span<const MessageView>;
    using Owned = std::span<const OwnedMessage>;

    std::variant<Borrowed, Owned> storage_{Borrowed{}};
};

/**
 * @brief Owning snapshot of retained history.
 */
struct HistorySnapshot {
    std::vector<OwnedMessage> messages;

    [[nodiscard]] ConversationView view() const noexcept {
        return ConversationView(std::span<const OwnedMessage>(messages));
    }

    [[nodiscard]] size_t size() const noexcept {
        return messages.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return messages.empty();
    }

    [[nodiscard]] const OwnedMessage& operator[](size_t index) const noexcept {
        return messages[index];
    }

    bool operator==(const HistorySnapshot& other) const = default;
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

    // Extraction errors (600-699)
    InvalidOutputSchema =
        600, ///< A supplied output schema is malformed or uses unsupported constructs.
    ExtractionFailed = 601, ///< Structured extraction from model output failed.

    // Hub layer errors (700-799). The hub layer mirrors these as `HubErrorCode`
    // for namespaced internal use; `to_error_code(HubErrorCode)` performs the
    // explicit one-to-one mapping when surfacing errors to callers.
    GgufReadFailed = 700,         ///< Could not open or parse a GGUF file for inspection.
    GgufMetadataNotFound = 701,   ///< An expected metadata key was missing from the GGUF file.
    ModelNotFound = 702,          ///< No model matched the given name, alias, or path.
    ModelAlreadyExists = 703,     ///< A model with the same path is already registered.
    DownloadFailed = 704,         ///< HTTP download of a model file failed.
    HuggingFaceApiError = 706,    ///< The HuggingFace API returned an error response.
    InvalidModelIdentifier = 707, ///< Could not parse the HuggingFace model identifier string.
    StoreCorrupted = 708,         ///< The model store catalog JSON is malformed.
    FilesystemError = 709,        ///< A filesystem operation failed.

    Unknown = 999 ///< Fallback code for uncategorized failures.
};

/**
 * @brief Rich error payload returned by fallible zoo-keeper operations.
 */
struct Error {
    ErrorCode code;                     ///< Machine-readable error category.
    std::string message;                ///< Human-readable error summary.
    std::optional<std::string> context; ///< Optional contextual detail for diagnostics.

    Error(ErrorCode code, std::string message, std::optional<std::string> context = std::nullopt)
        : code(code), message(std::move(message)), context(std::move(context)) {}

    [[nodiscard]] std::string to_string() const {
        std::string result = "[" + std::to_string(static_cast<int>(code)) + "] " + message;
        if (context.has_value()) {
            result += " | Context: " + *context;
        }
        return result;
    }

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
    int seed = -1;               ///< RNG seed. Negative values delegate seeding to the runtime.

    [[nodiscard]] Expected<void> validate() const {
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
 * @brief Callback invoked for streamed token fragments in synchronous generation.
 */
using TokenCallback = FunctionRef<TokenAction(std::string_view)>;

/**
 * @brief Callback queried by generation loops to decide whether work should stop.
 */
using CancellationCallback = FunctionRef<bool()>;

/**
 * @brief Async streaming callback stored by the agent runtime.
 */
using AsyncTextCallback = std::function<void(std::string_view)>;

/**
 * @brief Model loading and backend configuration.
 */
struct ModelConfig {
    std::string model_path;  ///< Filesystem path to the GGUF model.
    int context_size = 8192; ///< Requested context window size in tokens.
    int n_gpu_layers =
        0; ///< Number of layers to offload to GPU. Defaults to CPU-only for portability.
    bool use_mmap = true;   ///< Whether to memory-map the model file.
    bool use_mlock = false; ///< Whether to lock model pages in memory.

    [[nodiscard]] Expected<void> validate() const {
        if (model_path.empty()) {
            return std::unexpected(
                Error{ErrorCode::InvalidModelPath, "Model path cannot be empty"});
        }
        std::error_code ec;
        const bool model_exists = std::filesystem::exists(model_path, ec);
        if (ec) {
            return std::unexpected(Error{ErrorCode::InvalidModelPath,
                                         "Cannot access model path: " + model_path, ec.message()});
        }
        if (!model_exists) {
            return std::unexpected(
                Error{ErrorCode::InvalidModelPath, "Model file does not exist: " + model_path});
        }
        if (context_size <= 0) {
            return std::unexpected(
                Error{ErrorCode::InvalidContextSize, "Context size must be positive"});
        }
        return {};
    }

    bool operator==(const ModelConfig& other) const = default;
};

/**
 * @brief Agent queue, retention, and tool-loop policy configuration.
 */
struct AgentConfig {
    size_t max_history_messages = 64;   ///< Maximum number of non-system messages retained.
    size_t request_queue_capacity = 64; ///< Fixed number of request slots the agent may own.
    int max_tool_iterations = 5;        ///< Maximum detect/execute/respond iterations per request.
    int max_tool_retries = 2;           ///< Maximum validation retries for malformed tool calls.

    [[nodiscard]] Expected<void> validate() const {
        if (max_history_messages == 0) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "max_history_messages must be >= 1"});
        }
        if (request_queue_capacity == 0) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "request_queue_capacity must be >= 1"});
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

    bool operator==(const AgentConfig& other) const = default;
};

/**
 * @brief Per-call generation behavior shared by model and agent operations.
 */
struct GenerationOptions {
    SamplingParams sampling; ///< Sampling behavior for generation.
    int max_tokens = -1;     ///< Completion cap, or `-1` for the context-limited maximum.
    std::vector<std::string> stop_sequences; ///< User-defined stop sequences.
    bool record_tool_trace = false;          ///< When true, materialize detailed tool diagnostics.

    [[nodiscard]] Expected<void> validate() const {
        if (max_tokens == 0 || (max_tokens < 0 && max_tokens != -1)) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "max_tokens must be positive or -1 (unlimited)"});
        }
        return sampling.validate();
    }

    [[nodiscard]] bool is_default() const noexcept {
        return max_tokens == -1 && stop_sequences.empty() && !record_tool_trace &&
               sampling == SamplingParams{};
    }

    bool operator==(const GenerationOptions& other) const = default;
};

/**
 * @brief Token accounting for a completed generation.
 */
struct TokenUsage {
    int prompt_tokens = 0;     ///< Tokens consumed by the rendered prompt.
    int completion_tokens = 0; ///< Tokens emitted during generation.
    int total_tokens = 0;      ///< Sum of prompt and completion tokens.

    bool operator==(const TokenUsage& other) const = default;
};

/**
 * @brief Timing and throughput metrics captured for a response.
 */
struct Metrics {
    std::chrono::milliseconds latency_ms{0};             ///< End-to-end request latency.
    std::chrono::milliseconds time_to_first_token_ms{0}; ///< Delay until the first streamed token.
    double tokens_per_second = 0.0; ///< Throughput after the first token arrives.

    bool operator==(const Metrics& other) const = default;
};

/**
 * @brief Outcome recorded for one attempted tool invocation.
 */
enum class ToolInvocationStatus {
    Succeeded,        ///< Tool arguments validated and the handler returned a result.
    ValidationFailed, ///< Parsed arguments did not satisfy the registered schema.
    ExecutionFailed   ///< The handler returned an execution failure.
};

[[nodiscard]] inline const char* to_string(ToolInvocationStatus status) noexcept {
    switch (status) {
    case ToolInvocationStatus::Succeeded:
        return "succeeded";
    case ToolInvocationStatus::ValidationFailed:
        return "validation_failed";
    case ToolInvocationStatus::ExecutionFailed:
        return "execution_failed";
    }
    return "unknown";
}

/**
 * @brief Structured record of one attempted tool call.
 */
struct ToolInvocation {
    std::string id;             ///< Correlation identifier parsed from model output.
    std::string name;           ///< Registered tool name the model attempted to invoke.
    std::string arguments_json; ///< Serialized arguments exactly as parsed from model output.
    ToolInvocationStatus status = ToolInvocationStatus::Succeeded; ///< Final outcome category.
    std::optional<std::string> result_json; ///< Serialized handler result when execution succeeded.
    std::optional<Error> error; ///< Validation or execution error when the attempt failed.

    bool operator==(const ToolInvocation& other) const = default;
};

/**
 * @brief Optional diagnostic trace materialized only when explicitly requested.
 */
struct ToolTrace {
    std::vector<ToolInvocation> invocations;

    [[nodiscard]] bool empty() const noexcept {
        return invocations.empty();
    }

    bool operator==(const ToolTrace& other) const = default;
};

/**
 * @brief Text generation result for chat/complete/generate.
 */
struct TextResponse {
    std::string text;                    ///< Assistant-visible response text.
    TokenUsage usage;                    ///< Prompt and completion token usage.
    Metrics metrics;                     ///< Latency and throughput data.
    std::optional<ToolTrace> tool_trace; ///< Tool diagnostics when explicitly requested.

    bool operator==(const TextResponse& other) const = default;
};

/**
 * @brief Structured extraction result for `extract()`.
 */
struct ExtractionResponse {
    std::string text;                    ///< Raw generated JSON text.
    nlohmann::json data;                 ///< Parsed schema-conforming structured output.
    TokenUsage usage;                    ///< Prompt and completion token usage.
    Metrics metrics;                     ///< Latency and throughput data.
    std::optional<ToolTrace> tool_trace; ///< Tool diagnostics when explicitly requested.

    bool operator==(const ExtractionResponse& other) const = default;
};

/**
 * @brief Monotonic identifier assigned to queued agent requests.
 */
using RequestId = uint64_t;

/**
 * @brief Validates whether a new message role can be appended to an existing history.
 */
[[nodiscard]] inline Expected<void> validate_role_sequence(ConversationView messages, Role role) {
    if (messages.size() == 0) {
        if (role == Role::Tool) {
            return std::unexpected(Error{ErrorCode::InvalidMessageSequence,
                                         "First message cannot be a tool response"});
        }
        return {};
    }

    const Role last_role = messages[messages.size() - 1].role();
    if (role == last_role && role != Role::Tool) {
        return std::unexpected(
            Error{ErrorCode::InvalidMessageSequence,
                  "Cannot have consecutive messages with the same role (except Tool)"});
    }

    return {};
}

/**
 * @brief Validates whether a new message role can be appended to an existing history snapshot.
 */
[[nodiscard]] inline Expected<void> validate_role_sequence(const HistorySnapshot& messages,
                                                           Role role) {
    return validate_role_sequence(messages.view(), role);
}

/**
 * @brief Validates whether a new message role can be appended to an owned message span.
 */
[[nodiscard]] inline Expected<void> validate_role_sequence(std::span<const OwnedMessage> messages,
                                                           Role role) {
    return validate_role_sequence(ConversationView{messages}, role);
}

/**
 * @brief Validates whether a new message role can be appended to an owned message vector.
 */
[[nodiscard]] inline Expected<void>
validate_role_sequence(const std::vector<OwnedMessage>& messages, Role role) {
    return validate_role_sequence(std::span<const OwnedMessage>(messages), role);
}

/**
 * @brief Validates whether a new message role can be appended to a borrowed message span.
 */
[[nodiscard]] inline Expected<void> validate_role_sequence(std::span<const MessageView> messages,
                                                           Role role) {
    return validate_role_sequence(ConversationView{messages}, role);
}

/**
 * @brief Minimal tool description for template-driven tool calling at the core layer.
 *
 * This avoids a dependency from Layer 1 (core) on Layer 2 (tools).
 * The agent layer converts `tools::ToolMetadata` to this before passing to `Model`.
 */
struct CoreToolInfo {
    std::string name;            ///< Registered tool name exposed to the model.
    std::string description;     ///< Human-readable description used in the chat template.
    std::string parameters_json; ///< JSON Schema of the parameters as a string.
};

} // namespace zoo
