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
#include <type_traits>
#include <utility>
#include <vector>

namespace zoo {

namespace internal::agent {
template <typename Result> class RequestStateBase;

template <typename Result>
concept RequestHandleResult =
    std::same_as<Result, TextResponse> || std::same_as<Result, ExtractionResponse>;

template <typename Message>
concept ExtractMessage =
    std::same_as<std::remove_cvref_t<Message>, MessageView> ||
    requires(Message&& message) { std::string_view{std::forward<Message>(message)}; };
} // namespace internal::agent

/**
 * @brief Move-only async handle for one queued agent request.
 *
 * The handle holds a shared_ptr to a polymorphic state object whose concrete
 * type lives in the runtime (see `src/agent/request_state.hpp`). All non-trivial
 * members are defined in `src/agent/request_handle.cpp` and explicitly
 * instantiated for `TextResponse` and `ExtractionResponse`.
 */
template <internal::agent::RequestHandleResult Result> class RequestHandle {
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

    void cancel() const noexcept;

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
                                     AsyncTokenCallback callback = {});

    /**
     * @brief Queues a structured single-message request for asynchronous processing.
     */
    RequestHandle<TextResponse> chat(MessageView message,
                                     const GenerationOptions& options = GenerationOptions{},
                                     AsyncTokenCallback callback = {});

    /**
     * @brief Queues a stateless completion against the supplied full message history.
     */
    RequestHandle<TextResponse> complete(ConversationView messages,
                                         const GenerationOptions& options = GenerationOptions{},
                                         AsyncTokenCallback callback = {});

    /**
     * @brief Queues a structured extraction request (stateful).
     */
    template <internal::agent::ExtractMessage Message>
    RequestHandle<ExtractionResponse>
    extract(const nlohmann::json& output_schema, Message&& message,
            const GenerationOptions& options = {}, AsyncTokenCallback callback = {}) {
        if constexpr (std::same_as<std::remove_cvref_t<Message>, MessageView>) {
            return extract_stateful(output_schema, message, options, std::move(callback));
        } else {
            return extract_stateful(
                output_schema,
                MessageView{Role::User, std::string_view{std::forward<Message>(message)}}, options,
                std::move(callback));
        }
    }

    /**
     * @brief Queues a structured extraction request (stateless).
     */
    RequestHandle<ExtractionResponse>
    extract(const nlohmann::json& output_schema, ConversationView messages,
            const GenerationOptions& options = GenerationOptions{},
            AsyncTokenCallback callback = {});

    /**
     * @brief Requests cancellation of a queued or running request.
     */
    void cancel(RequestId id);

    /// Best-effort system-prompt replacement. Use `try_set_system_prompt()` to observe errors.
    void set_system_prompt(std::string_view prompt);

    /// Replaces the current system prompt, returning command-lane failures.
    Expected<void> try_set_system_prompt(std::string_view prompt);

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

    /// Best-effort history snapshot. Use `try_get_history()` to observe errors.
    [[nodiscard]] HistorySnapshot get_history() const;

    /// Returns a history snapshot, or an error if the command cannot run.
    [[nodiscard]] Expected<HistorySnapshot> try_get_history() const;

    /// Returns a history snapshot, or `RequestTimeout` if the command waits too long.
    /// @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
    [[nodiscard]] Expected<HistorySnapshot> get_history(std::chrono::nanoseconds timeout) const;

    /// Best-effort history clear. Use `try_clear_history()` to observe errors.
    void clear_history();

    /// Clears history, or returns an error if the command cannot run.
    Expected<void> try_clear_history();

    /// Clears history, returning `RequestTimeout` if the command waits too long.
    /// @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
    Expected<void> clear_history(std::chrono::nanoseconds timeout);

    /// @brief Registers a typed callable as a tool.
    template <typename Func>
    Expected<void> register_tool(std::string_view name, std::string_view description,
                                 std::span<const std::string> param_names, Func func,
                                 std::optional<std::chrono::nanoseconds> timeout = {}) {
        std::string tool_name{name};
        std::string tool_description{description};
        auto definition = tools::detail::make_tool_definition(
            tool_name, tool_description,
            std::vector<std::string>(param_names.begin(), param_names.end()), std::move(func));
        if (!definition) {
            return std::unexpected(definition.error());
        }
        return register_tool(std::move(*definition), timeout);
    }

    template <typename Func>
    Expected<void> register_tool(std::string_view name, std::string_view description,
                                 std::initializer_list<std::string> param_names, Func func,
                                 std::optional<std::chrono::nanoseconds> timeout = {}) {
        const std::vector<std::string> names(param_names);
        return register_tool(name, description, std::span<const std::string>(names),
                             std::move(func), timeout);
    }

    /// @brief Registers a tool using a JSON Schema and a JSON-backed handler.
    template <typename Handler>
        requires tools::detail::is_json_handler_like_v<Handler>
    Expected<void> register_tool(std::string_view name, std::string_view description,
                                 const nlohmann::json& schema, Handler handler,
                                 std::optional<std::chrono::nanoseconds> timeout = {}) {
        return register_tool(name, description, nlohmann::json(schema),
                             tools::ToolHandler(std::move(handler)), timeout);
    }

    /// @brief Registers a tool from a prebuilt JSON Schema and handler.
    Expected<void> register_tool(std::string_view name, std::string_view description,
                                 nlohmann::json schema, tools::ToolHandler handler,
                                 std::optional<std::chrono::nanoseconds> timeout = {});

    /**
     * @brief Registers multiple tool definitions in a single queued command.
     * @param definitions Tool definitions to register.
     * @param timeout Maximum time to wait; returns `RequestTimeout` on expiry.
     * @return Void on success, or the first error encountered.
     */
    Expected<void> register_tools(std::vector<tools::ToolDefinition> definitions,
                                  std::optional<std::chrono::nanoseconds> timeout = {});

    [[nodiscard]] size_t tool_count() const noexcept;

  private:
    struct Impl;

    Agent(ModelConfig model_config, AgentConfig agent_config, GenerationOptions default_generation,
          std::unique_ptr<Impl> impl);
    RequestHandle<ExtractionResponse> extract_stateful(const nlohmann::json& output_schema,
                                                       MessageView message,
                                                       const GenerationOptions& options,
                                                       AsyncTokenCallback callback);
    Expected<void> register_tool(tools::ToolDefinition definition,
                                 std::optional<std::chrono::nanoseconds> timeout = {});

    ModelConfig model_config_;
    AgentConfig agent_config_;
    GenerationOptions default_generation_options_;
    std::unique_ptr<Impl> impl_;
};

} // namespace zoo
