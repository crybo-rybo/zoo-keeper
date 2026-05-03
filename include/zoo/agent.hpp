/**
 * @file agent.hpp
 * @brief Asynchronous orchestration layer that coordinates model inference and tool execution.
 */

#pragma once

#include "core/types.hpp"
#include "tools/registry.hpp"
#include <chrono>
#include <concepts>
#include <initializer_list>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace zoo {

namespace internal::agent {
template <typename Result> class RequestStateBase;
} // namespace internal::agent

/**
 * @brief Move-only async handle for one queued agent request.
 *
 * The handle holds a shared_ptr to a polymorphic state object whose concrete
 * type lives in the runtime (see `src/agent/request_state.hpp`). All non-trivial
 * members are defined in `src/agent/request_handle.cpp` and explicitly
 * instantiated for `TextResponse` and `ExtractionResponse`.
 */
template <typename Result> class RequestHandle {
  public:
    RequestHandle() noexcept = default;

    RequestHandle(std::shared_ptr<internal::agent::RequestStateBase<Result>> state,
                  RequestId id) noexcept
        : id_(id), state_(std::move(state)) {}

    ~RequestHandle();

    RequestHandle(RequestHandle&& other) noexcept
        : id_(other.id_), state_(std::move(other.state_)) {
        other.id_ = 0;
    }

    RequestHandle& operator=(RequestHandle&& other) noexcept;

    RequestHandle(const RequestHandle&) = delete;
    RequestHandle& operator=(const RequestHandle&) = delete;

    [[nodiscard]] RequestId id() const noexcept {
        return id_;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return valid();
    }

    [[nodiscard]] bool valid() const noexcept {
        return static_cast<bool>(state_);
    }

    [[nodiscard]] bool ready() const;

    Expected<Result> await_result();

    template <typename Rep, typename Period>
    Expected<Result> await_result(std::chrono::duration<Rep, Period> timeout) {
        return await_result_for(std::chrono::duration_cast<std::chrono::nanoseconds>(timeout));
    }

    void reset() noexcept;

  private:
    Expected<Result> await_result_for(std::chrono::nanoseconds timeout);

    RequestId id_ = 0;
    std::shared_ptr<internal::agent::RequestStateBase<Result>> state_;
};

/**
 * @brief Async orchestration layer built on top of `zoo::core::Model`.
 */
class Agent {
  public:
    /**
     * @brief Creates and starts an agent from split model, agent, and generation settings.
     */
    static Expected<std::unique_ptr<Agent>>
    create(const ModelConfig& model_config, const AgentConfig& agent_config = AgentConfig{},
           const GenerationOptions& default_generation = GenerationOptions{});

    ~Agent();

    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;
    Agent(Agent&&) = delete;
    Agent& operator=(Agent&&) = delete;

    /**
     * @brief Queues a user message for asynchronous processing.
     */
    RequestHandle<TextResponse> chat(std::string_view user_message,
                                     const GenerationOptions& options = GenerationOptions{},
                                     AsyncTextCallback callback = {});

    /**
     * @brief Queues a structured single-message request for asynchronous processing.
     */
    RequestHandle<TextResponse> chat(MessageView message,
                                     const GenerationOptions& options = GenerationOptions{},
                                     AsyncTextCallback callback = {});

    /**
     * @brief Queues a stateless completion against the supplied full message history.
     */
    RequestHandle<TextResponse> complete(ConversationView messages,
                                         const GenerationOptions& options = GenerationOptions{},
                                         AsyncTextCallback callback = {});

    /**
     * @brief Queues a structured extraction request (stateful).
     */
    RequestHandle<ExtractionResponse>
    extract(const nlohmann::json& output_schema, std::string_view user_message,
            const GenerationOptions& options = GenerationOptions{},
            AsyncTextCallback callback = {});

    /**
     * @brief Queues a structured extraction request (stateful).
     */
    RequestHandle<ExtractionResponse>
    extract(const nlohmann::json& output_schema, MessageView message,
            const GenerationOptions& options = GenerationOptions{},
            AsyncTextCallback callback = {});

    /**
     * @brief Queues a structured extraction request (stateless).
     */
    RequestHandle<ExtractionResponse>
    extract(const nlohmann::json& output_schema, ConversationView messages,
            const GenerationOptions& options = GenerationOptions{},
            AsyncTextCallback callback = {});

    /**
     * @brief Requests cancellation of a queued or running request.
     */
    void cancel(RequestId id);

    /**
     * @brief Replaces the current system prompt on the underlying model.
     */
    void set_system_prompt(std::string_view prompt);

    /**
     * @brief Replaces the current system prompt, returning RequestTimeout if the command waits too
     * long.
     * @param prompt The new system prompt text.
     * @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
     */
    Expected<void> set_system_prompt(std::string_view prompt, std::chrono::nanoseconds timeout);

    /**
     * @brief Appends a system-role message to the conversation without replacing the initial
     * system prompt. Uses incremental history append (no KV cache flush).
     */
    Expected<void> add_system_message(std::string_view message);

    /**
     * @brief Appends a system-role message, returning RequestTimeout if the command waits too long.
     * @param message The system message text to append.
     * @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
     */
    Expected<void> add_system_message(std::string_view message, std::chrono::nanoseconds timeout);

    /// Stops the worker thread and prevents additional requests from being processed.
    void stop();

    /// Returns whether the background inference thread is still accepting work.
    [[nodiscard]] bool is_running() const noexcept;

    [[nodiscard]] const ModelConfig& model_config() const noexcept {
        return model_config_;
    }

    [[nodiscard]] const AgentConfig& agent_config() const noexcept {
        return agent_config_;
    }

    [[nodiscard]] const GenerationOptions& default_generation_options() const noexcept {
        return default_generation_options_;
    }

    /// Returns a history snapshot taken synchronously on the inference thread.
    [[nodiscard]] HistorySnapshot get_history() const;

    /// Returns a history snapshot, or `RequestTimeout` if the command waits too long.
    /// @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
    [[nodiscard]] Expected<HistorySnapshot> get_history(std::chrono::nanoseconds timeout) const;

    /// Clears history synchronously on the inference thread before later queued work.
    void clear_history();

    /// Clears history, returning `RequestTimeout` if the command waits too long.
    /// @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
    Expected<void> clear_history(std::chrono::nanoseconds timeout);

    /// @brief Registers a typed callable as a tool (initializer_list overload).
    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 std::initializer_list<std::string> param_names, Func func) {
        return register_tool(name, description, std::vector<std::string>(param_names),
                             std::move(func));
    }

    /// @brief Registers a typed callable as a tool with timeout (initializer_list overload).
    /// @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 std::initializer_list<std::string> param_names, Func func,
                                 std::chrono::nanoseconds timeout) {
        return register_tool(name, description, std::vector<std::string>(param_names),
                             std::move(func), timeout);
    }

    /// @brief Registers a typed callable as a tool (span overload).
    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 std::span<const std::string> param_names, Func func) {
        auto definition = tools::detail::make_tool_definition(
            name, description, std::vector<std::string>(param_names.begin(), param_names.end()),
            std::move(func));
        if (!definition) {
            return std::unexpected(definition.error());
        }
        return register_tool(std::move(*definition));
    }

    /// @brief Registers a typed callable as a tool with timeout (span overload).
    /// @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 std::span<const std::string> param_names, Func func,
                                 std::chrono::nanoseconds timeout) {
        auto definition = tools::detail::make_tool_definition(
            name, description, std::vector<std::string>(param_names.begin(), param_names.end()),
            std::move(func));
        if (!definition) {
            return std::unexpected(definition.error());
        }
        return register_tool(std::move(*definition), timeout);
    }

    /// @brief Registers a tool using a JSON Schema and a JSON-backed handler.
    template <typename Handler>
        requires tools::detail::is_json_handler_like_v<Handler>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 const nlohmann::json& schema, Handler handler) {
        return register_tool(name, description, nlohmann::json(schema),
                             tools::ToolHandler(std::move(handler)));
    }

    /// @brief Registers a tool using a JSON Schema and handler, with timeout.
    /// @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
    template <typename Handler>
        requires tools::detail::is_json_handler_like_v<Handler>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 const nlohmann::json& schema, Handler handler,
                                 std::chrono::nanoseconds timeout) {
        return register_tool(name, description, nlohmann::json(schema),
                             tools::ToolHandler(std::move(handler)), timeout);
    }

    /// @brief Registers a tool from a prebuilt JSON Schema and handler.
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 nlohmann::json schema, tools::ToolHandler handler);
    /// @brief Registers a tool from a prebuilt JSON Schema and handler, with timeout.
    /// @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 nlohmann::json schema, tools::ToolHandler handler,
                                 std::chrono::nanoseconds timeout);

    /**
     * @brief Registers multiple tool definitions in a single queued command.
     * @param definitions Tool definitions to register.
     * @return Void on success, or the first error encountered.
     */
    Expected<void> register_tools(std::vector<tools::ToolDefinition> definitions);
    /**
     * @brief Registers multiple tool definitions in a single queued command, with timeout.
     * @param definitions Tool definitions to register.
     * @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
     * @return Void on success, or the first error encountered.
     */
    Expected<void> register_tools(std::vector<tools::ToolDefinition> definitions,
                                  std::chrono::nanoseconds timeout);

    [[nodiscard]] size_t tool_count() const noexcept;

  private:
    struct Impl;

    Agent(ModelConfig model_config, AgentConfig agent_config, GenerationOptions default_generation,
          std::unique_ptr<Impl> impl);
    Expected<void> register_tool(tools::ToolDefinition definition);
    Expected<void> register_tool(tools::ToolDefinition definition,
                                 std::chrono::nanoseconds timeout);

    ModelConfig model_config_;
    AgentConfig agent_config_;
    GenerationOptions default_generation_options_;
    std::unique_ptr<Impl> impl_;
};

} // namespace zoo
