/**
 * @file agent.hpp
 * @brief Asynchronous orchestration layer that coordinates model inference and tool execution.
 */

#pragma once

#include "core/types.hpp"
#include "tools/registry.hpp"
#include <concepts>
#include <exception>
#include <future>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace zoo {

/**
 * @brief Handle returned by `Agent::chat()` for request tracking and result retrieval.
 */
struct RequestHandle {
    RequestId id;                           ///< Request identifier accepted by the agent.
    std::future<Expected<Response>> future; ///< Future resolved with the final response or error.

    /// Creates an empty handle with an invalid request id.
    RequestHandle() noexcept : id(0) {}
    /**
     * @brief Creates a handle bound to a live request future.
     *
     * @param id Assigned request identifier.
     * @param future Future that resolves when processing completes.
     */
    RequestHandle(RequestId id, std::future<Expected<Response>> future)
        : id(id), future(std::move(future)) {}
    /// Moves ownership of the result future.
    RequestHandle(RequestHandle&&) noexcept = default;
    /// Moves ownership of the result future.
    RequestHandle& operator=(RequestHandle&&) noexcept = default;
    /// Request handles are non-copyable because `std::future` is move-only.
    RequestHandle(const RequestHandle&) = delete;
    /// Request handles are non-copyable because `std::future` is move-only.
    RequestHandle& operator=(const RequestHandle&) = delete;
};

/**
 * @brief Async orchestration layer built on top of `zoo::core::Model`.
 *
 * `Agent` owns a background inference thread, a request queue, and a tool
 * registry. It implements the tool loop of detect, validate, execute, inject,
 * and re-generate until the assistant produces a final user-visible response.
 */
class Agent {
  public:
    /**
     * @brief Creates and starts an agent from the supplied configuration.
     *
     * @param config Runtime configuration used to load the underlying model.
     * @return A running agent, or an error if model creation fails.
     */
    static Expected<std::unique_ptr<Agent>> create(const Config& config);

    /// Stops the worker thread and releases owned resources.
    ~Agent();

    /// Agents own background state and cannot be copied.
    Agent(const Agent&) = delete;
    /// Agents own background state and cannot be copied.
    Agent& operator=(const Agent&) = delete;
    /// Agents own thread-affine state and cannot be moved.
    Agent(Agent&&) = delete;
    /// Agents own thread-affine state and cannot be moved.
    Agent& operator=(Agent&&) = delete;

    /**
     * @brief Queues a user message for asynchronous processing.
     *
     * @param message User message to append to the conversation.
     * @param callback Optional callback that receives streamed visible text.
     * @return Handle containing the request id and result future. If the agent
     *         is not running or the queue rejects the request, the future is
     *         resolved immediately with an error.
     */
    RequestHandle
    chat(Message message,
         std::optional<std::function<void(std::string_view)>> callback = std::nullopt);

    /**
     * @brief Requests cancellation of a queued or running request.
     *
     * Cancellation is cooperative. Requests that have already completed or have
     * been cleaned up are unaffected.
     *
     * @param id Request identifier returned by `chat()`.
     */
    void cancel(RequestId id);

    /**
     * @brief Replaces the current system prompt on the underlying model.
     *
     * This call blocks until the inference thread has applied the change. If a
     * request is currently generating, the prompt update runs before the next
     * queued request begins.
     *
     * @param prompt New system prompt content.
     */
    void set_system_prompt(const std::string& prompt);

    /// Stops the worker thread and prevents additional requests from being processed.
    void stop();

    /// Returns whether the background inference thread is still accepting work.
    bool is_running() const noexcept;

    /// Returns the immutable configuration used to create the agent.
    const Config& get_config() const noexcept {
        return config_;
    }

    /// Returns a history snapshot taken synchronously on the inference thread.
    std::vector<Message> get_history() const;

    /// Clears history synchronously on the inference thread before later queued work.
    void clear_history();

    /**
     * @brief Registers a strongly typed tool and refreshes grammar constraints.
     *
     * This call blocks until the inference thread has applied the grammar
     * refresh, so the tool is ready to use when the call returns.
     *
     * @tparam Func Callable type to register.
     * @param name Public tool name.
     * @param description Human-readable description for prompts and schemas.
     * @param param_names Parameter names in callable argument order.
     * @param func Callable implementation.
     * @return Empty success when registered, or the underlying registry error.
     */
    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 std::initializer_list<std::string> param_names, Func func) {
        return register_tool(name, description, std::vector<std::string>(param_names),
                             std::move(func));
    }

    /**
     * @brief Registers a strongly typed tool and refreshes grammar constraints.
     *
     * @tparam Func Callable type to register.
     * @param name Public tool name.
     * @param description Human-readable description for prompts and schemas.
     * @param param_names Parameter names in callable argument order.
     * @param func Callable implementation.
     * @return Empty success when registered, or the underlying registry error.
     */
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

    /**
     * @brief Registers a tool using a prebuilt JSON Schema and a JSON-backed callable.
     *
     * @tparam Handler Callable type that accepts one JSON object and returns `Expected<json>`.
     * @param name Public tool name.
     * @param description Human-readable description for prompts and schemas.
     * @param schema JSON Schema describing accepted arguments.
     * @param handler JSON-backed callable implementation.
     * @return Empty success when registered.
     */
    template <typename Handler>
        requires tools::detail::is_json_handler_like_v<Handler>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 const nlohmann::json& schema, Handler handler) {
        return register_tool(name, description, nlohmann::json(schema),
                             tools::ToolHandler(std::move(handler)));
    }

    /**
     * @brief Registers a tool using a prebuilt JSON Schema and JSON-backed handler.
     *
     * @param name Public tool name.
     * @param description Human-readable description for prompts and schemas.
     * @param schema JSON Schema describing accepted arguments.
     * @param handler JSON-backed callable implementation.
     * @return Empty success when registered.
     */
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 nlohmann::json schema, tools::ToolHandler handler);

    /// Returns the number of tools currently registered with the agent.
    size_t tool_count() const noexcept;

    /**
     * @brief Builds a system prompt that advertises the currently registered tools.
     *
     * When grammar-based tool calling is active the prompt describes the
     * sentinel-tagged format; otherwise it falls back to plain JSON instructions.
     *
     * @param base_prompt Base system prompt to extend.
     * @return Prompt text augmented with tool usage instructions and schemas.
     */
    std::string build_tool_system_prompt(const std::string& base_prompt) const;

  private:
    struct Impl;

    Agent(Config config, std::unique_ptr<Impl> impl);
    Expected<void> register_tool(tools::ToolDefinition definition);

    Config config_;
    std::unique_ptr<Impl> impl_;
};

} // namespace zoo
