#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <functional>
#include <atomic>
#include <memory>
#include <future>
#include <tl/expected.hpp>

namespace zoo {

// ============================================================================
// Message Types
// ============================================================================

/**
 * @brief Message role in conversation flow
 *
 * Defines the source and purpose of a message in the conversation history.
 */
enum class Role {
    System,     ///< System instructions that guide model behavior
    User,       ///< Input from the end user
    Assistant,  ///< Model-generated response
    Tool        ///< Result from tool execution
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

/**
 * @brief Single message in conversation history
 *
 * Value type representing one turn in the conversation. Contains the role
 * (system, user, assistant, or tool), the text content, and an optional
 * tool_call_id for correlating tool responses.
 *
 * @threadsafety Safe to copy and pass by value across threads
 */
struct Message {
    Role role;                                 ///< Message role (system/user/assistant/tool)
    std::string content;                       ///< Text content of the message
    std::optional<std::string> tool_call_id;   ///< Tool correlation ID (tool responses only)

    // Factory methods
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

    // Equality for testing
    bool operator==(const Message& other) const {
        return role == other.role &&
               content == other.content &&
               tool_call_id == other.tool_call_id;
    }

    bool operator!=(const Message& other) const {
        return !(*this == other);
    }
};

// ============================================================================
// Error Types
// ============================================================================

/**
 * @brief Error codes organized by category range
 *
 * Error codes are grouped into ranges by category:
 * - 100-199: Configuration errors
 * - 200-299: Backend/model errors
 * - 300-399: Engine logic errors
 * - 400-499: Runtime/request errors
 * - 500-599: Tool system errors (reserved)
 */
enum class ErrorCode {
    // Configuration errors (100-199)
    InvalidConfig = 100,
    InvalidModelPath = 101,
    InvalidContextSize = 102,
    InvalidTemplate = 103,

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
    HistoryCorrupted = 303,

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

    // MCP errors (600-699)
    McpTransportFailed = 600,
    McpProtocolError = 601,
    McpServerError = 602,
    McpSessionFailed = 603,
    McpToolNotAvailable = 604,
    McpTimeout = 605,
    McpDisconnected = 606,

    // Unknown
    Unknown = 999
};

/**
 * @brief Error information with code, message, and optional context
 *
 * Value type representing a library error. Used with tl::expected for
 * composable error handling without exceptions.
 */
struct Error {
    ErrorCode code;                      ///< Categorized error code
    std::string message;                 ///< Human-readable error description
    std::optional<std::string> context;  ///< Additional context (e.g., file paths, values)

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

// Expected type alias
template<typename T>
using Expected = tl::expected<T, Error>;

// ============================================================================
// Sampling Configuration
// ============================================================================

/**
 * @brief Sampling parameters controlling model output randomness
 *
 * Value type containing standard sampling hyperparameters for LLM inference.
 * Default values provide balanced creativity vs coherence.
 *
 * @threadsafety Safe to copy and pass by value across threads
 */
struct SamplingParams {
    float temperature = 0.7f;        ///< Sampling temperature (0.0 = deterministic, higher = more random)
    float top_p = 0.9f;              ///< Nucleus sampling threshold (0.0-1.0)
    int top_k = 40;                  ///< Top-K sampling limit (0 = disabled)
    float repeat_penalty = 1.1f;     ///< Penalty for repeating tokens (1.0 = no penalty)
    int repeat_last_n = 64;          ///< Number of tokens to consider for repeat penalty
    int seed = -1;                   ///< Random seed (-1 = random seed per request)

    // Equality for testing
    bool operator==(const SamplingParams& other) const {
        return temperature == other.temperature &&
               top_p == other.top_p &&
               top_k == other.top_k &&
               repeat_penalty == other.repeat_penalty &&
               repeat_last_n == other.repeat_last_n &&
               seed == other.seed;
    }

    bool operator!=(const SamplingParams& other) const {
        return !(*this == other);
    }
};

// ============================================================================
// Prompt Template
// ============================================================================

enum class PromptTemplate {
    Llama3,
    ChatML,
    Custom
};

inline std::string template_to_string(PromptTemplate tmpl) {
    switch (tmpl) {
        case PromptTemplate::Llama3: return "Llama3";
        case PromptTemplate::ChatML: return "ChatML";
        case PromptTemplate::Custom: return "Custom";
    }
    return "unknown";
}

// ============================================================================
// Agent Configuration
// ============================================================================

/**
 * @brief Complete configuration for Agent initialization
 *
 * Value type containing all model, inference, and generation settings.
 * Must be validated via validate() before use. Immutable after Agent
 * construction.
 *
 * @threadsafety Safe to copy and pass by value across threads (excluding callbacks)
 */
struct Config {
    // Model settings
    std::string model_path;                                  ///< Path to GGUF model file (required)
    int context_size = 8192;                                 ///< Context window size in tokens (> 0)
    int n_gpu_layers = -1;                                   ///< GPU layers to offload (-1 = all, 0 = CPU only)
    bool use_mmap = true;                                    ///< Memory-map model file for faster loading
    bool use_mlock = false;                                  ///< Lock model in RAM (prevents swapping)

    // Sampling
    SamplingParams sampling;                                 ///< Sampling hyperparameters

    // Template
    PromptTemplate prompt_template = PromptTemplate::Llama3; ///< Chat template format
    std::optional<std::string> custom_template;              ///< Custom Jinja2 template (required if PromptTemplate::Custom)

    // Generation limits
    int max_tokens = 512;                                    ///< Maximum tokens to generate per response (> 0)
    std::vector<std::string> stop_sequences;                 ///< Additional stop strings to halt generation

    // System prompt
    std::optional<std::string> system_prompt;                ///< System message prepended to conversations

    // Memory estimation
    /// If true, Agent::create() will reduce context_size to fit estimated available memory.
    /// Requires the model file to be accessible for GGUF metadata reading.
    bool auto_tune_context = false;

    // Queue settings
    size_t request_queue_capacity = 0;                       ///< Maximum request queue size (0 = unlimited)

    // KV cache quantization type (ggml_type enum values as int, to avoid ggml dependency in public header).
    // Default: 1 = GGML_TYPE_F16 (matches current behavior, full precision)
    // Recommended for memory savings: 8 = GGML_TYPE_Q8_0 (half memory, near-lossless quality)
    // Aggressive: 2 = GGML_TYPE_Q4_0 (quarter memory, some quality loss)
    // Note: quantized V cache requires flash attention, which zoo-keeper already enables.
    int kv_cache_type_k = 1;  ///< KV cache K type (GGML_TYPE_F16 by default)
    int kv_cache_type_v = 1;  ///< KV cache V type (GGML_TYPE_F16 by default)

    // Callbacks
    using TokenCallback = std::function<void(std::string_view)>;
    std::optional<TokenCallback> on_token;                   ///< Per-token streaming callback (runs on inference thread)

    // Validation
    Expected<void> validate() const {
        if (model_path.empty()) {
            return tl::unexpected(Error{ErrorCode::InvalidModelPath, "Model path cannot be empty"});
        }
        if (context_size <= 0) {
            return tl::unexpected(Error{ErrorCode::InvalidContextSize, "Context size must be positive"});
        }
        if (max_tokens <= 0) {
            return tl::unexpected(Error{ErrorCode::InvalidConfig, "max_tokens must be positive"});
        }
        if (prompt_template == PromptTemplate::Custom && !custom_template.has_value()) {
            return tl::unexpected(Error{ErrorCode::InvalidTemplate, "Custom template string required for PromptTemplate::Custom"});
        }
        if (kv_cache_type_k < 0 || kv_cache_type_v < 0) {
            return tl::unexpected(Error{ErrorCode::InvalidConfig, "kv_cache_type_k and kv_cache_type_v must be >= 0"});
        }
        return {};
    }

    // Equality for testing (excluding callbacks)
    bool operator==(const Config& other) const {
        return model_path == other.model_path &&
               context_size == other.context_size &&
               n_gpu_layers == other.n_gpu_layers &&
               use_mmap == other.use_mmap &&
               use_mlock == other.use_mlock &&
               sampling == other.sampling &&
               prompt_template == other.prompt_template &&
               custom_template == other.custom_template &&
               max_tokens == other.max_tokens &&
               stop_sequences == other.stop_sequences &&
               system_prompt == other.system_prompt &&
               request_queue_capacity == other.request_queue_capacity &&
               kv_cache_type_k == other.kv_cache_type_k &&
               kv_cache_type_v == other.kv_cache_type_v &&
               auto_tune_context == other.auto_tune_context;
    }

    bool operator!=(const Config& other) const {
        return !(*this == other);
    }
};

// ============================================================================
// RAG Types
// ============================================================================

/**
 * @brief Retrieved context chunk used for RAG-grounded generation.
 *
 * Contains the chunk text, retrieval score, and optional source identifier.
 */
struct RagChunk {
    std::string id;                           ///< Stable chunk identifier
    std::string content;                      ///< Retrieved chunk text
    double score = 0.0;                       ///< Retrieval relevance score (higher is better)
    std::optional<std::string> source;        ///< Optional source identifier (doc ID, file path, URI)

    bool operator==(const RagChunk& other) const {
        return id == other.id &&
               content == other.content &&
               score == other.score &&
               source == other.source;
    }

    bool operator!=(const RagChunk& other) const {
        return !(*this == other);
    }
};

/**
 * @brief RAG behavior options for a single chat request.
 *
 * If enabled, the engine can either:
 * - Use `context_override` directly as ephemeral context, or
 * - Query a configured retriever and inject top-k chunks.
 *
 * Injected context is ephemeral and is not stored in long-term history.
 */
struct RagOptions {
    bool enabled = false;                             ///< Enable RAG for this request
    int top_k = 4;                                    ///< Number of chunks to retrieve when querying retriever
    std::optional<std::string> context_override;      ///< Optional precomputed context block to inject

    bool operator==(const RagOptions& other) const {
        return enabled == other.enabled &&
               top_k == other.top_k &&
               context_override == other.context_override;
    }

    bool operator!=(const RagOptions& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Per-request chat options.
 */
struct ChatOptions {
    RagOptions rag;                                   ///< Retrieval-augmented generation settings

    bool operator==(const ChatOptions& other) const {
        return rag == other.rag;
    }

    bool operator!=(const ChatOptions& other) const {
        return !(*this == other);
    }
};

// ============================================================================
// Response Types
// ============================================================================

/**
 * @brief Token usage statistics for a single request
 *
 * Tracks the number of tokens consumed during inference for cost
 * estimation and quota management.
 */
struct TokenUsage {
    int prompt_tokens = 0;      ///< Tokens in the input prompt
    int completion_tokens = 0;  ///< Tokens generated in the response
    int total_tokens = 0;       ///< Sum of prompt_tokens + completion_tokens

    bool operator==(const TokenUsage& other) const {
        return prompt_tokens == other.prompt_tokens &&
               completion_tokens == other.completion_tokens &&
               total_tokens == other.total_tokens;
    }

    bool operator!=(const TokenUsage& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Performance metrics for a single inference request
 *
 * Captures timing and throughput data for performance monitoring
 * and optimization.
 */
struct Metrics {
    std::chrono::milliseconds latency_ms{0};             ///< Total request duration (submit to completion)
    std::chrono::milliseconds time_to_first_token_ms{0}; ///< Time until first token streamed
    double tokens_per_second = 0.0;                      ///< Generation throughput (tokens/sec)

    bool operator==(const Metrics& other) const {
        return latency_ms == other.latency_ms &&
               time_to_first_token_ms == other.time_to_first_token_ms &&
               tokens_per_second == other.tokens_per_second;
    }

    bool operator!=(const Metrics& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Complete response from a chat() request
 *
 * Value type containing the generated text, token usage statistics,
 * performance metrics, and tool call history (if applicable).
 *
 * @threadsafety Safe to copy and pass by value across threads
 */
struct Response {
    std::string text;                 ///< Generated response text
    TokenUsage usage;                 ///< Token consumption statistics
    Metrics metrics;                  ///< Performance timing data
    std::vector<Message> tool_calls;  ///< Tool call and result history from agentic loop
    std::vector<RagChunk> rag_chunks; ///< Retrieved RAG chunks used for this turn (ephemeral context provenance)

    bool operator==(const Response& other) const {
        return text == other.text &&
               usage == other.usage &&
               metrics == other.metrics &&
               tool_calls == other.tool_calls &&
               rag_chunks == other.rag_chunks;
    }

    bool operator!=(const Response& other) const {
        return !(*this == other);
    }
};

// ============================================================================
// Request Types
// ============================================================================

/// Unique identifier for a chat request, used for per-request cancellation.
using RequestId = uint64_t;

/**
 * @brief Handle returned from Agent::chat() for per-request cancellation
 *
 * Contains a unique request ID and the future for the response.
 * Use the ID with Agent::cancel() to cancel a specific request.
 */
struct RequestHandle {
    RequestId id;                         ///< Unique request identifier
    std::future<Expected<Response>> future; ///< Future for the response

    // Move-only (std::future is not copyable)
    RequestHandle() : id(0) {}
    RequestHandle(RequestId id, std::future<Expected<Response>> future)
        : id(id), future(std::move(future)) {}
    RequestHandle(RequestHandle&&) = default;
    RequestHandle& operator=(RequestHandle&&) = default;
    RequestHandle(const RequestHandle&) = delete;
    RequestHandle& operator=(const RequestHandle&) = delete;
};

/**
 * @brief Internal request representation with metadata
 *
 * Wraps a user message with optional streaming callback and submission
 * timestamp for queue processing and metrics collection.
 *
 * @note This type is internal to the engine and not part of the public API
 */
struct Request {
    Message message;                                                  ///< User message to process
    ChatOptions options;                                              ///< Per-request options (RAG, etc.)
    std::optional<std::function<void(std::string_view)>> streaming_callback; ///< Per-token callback override
    std::chrono::steady_clock::time_point submitted_at;               ///< Timestamp for latency tracking
    std::shared_ptr<std::promise<Expected<Response>>> promise;        ///< Bundled promise for result delivery
    RequestId id = 0;                                                 ///< Unique request identifier
    std::shared_ptr<std::atomic<bool>> cancelled;                     ///< Per-request cancellation flag

    Request(
        Message msg,
        ChatOptions opts = {},
        std::optional<std::function<void(std::string_view)>> callback = std::nullopt
    )
        : message(std::move(msg))
        , options(std::move(opts))
        , streaming_callback(std::move(callback))
        , submitted_at(std::chrono::steady_clock::now())
        , cancelled(std::make_shared<std::atomic<bool>>(false))
    {}

    Request(
        Message msg,
        std::optional<std::function<void(std::string_view)>> callback
    )
        : Request(std::move(msg), ChatOptions{}, std::move(callback))
    {}
};

} // namespace zoo
