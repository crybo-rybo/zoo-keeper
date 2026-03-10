/**
 * @file agent.hpp
 * @brief Asynchronous orchestration layer that coordinates model inference and tool execution.
 */

#pragma once

#include "core/types.hpp"
#include "tools/registry.hpp"
#include <exception>
#include <future>
#include <memory>
#include <optional>
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

    /// Returns a snapshot of the underlying model conversation history.
    std::vector<Message> get_history() const;

    /// Clears the underlying model conversation history.
    void clear_history();

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
                                 const std::vector<std::string>& param_names, Func func) {
        using traits = tools::detail::function_traits<Func>;
        using args_tuple = typename traits::args_tuple;

        if (param_names.size() != traits::arity) {
            return std::unexpected(Error{
                ErrorCode::InvalidToolSignature,
                "Parameter name count (" + std::to_string(param_names.size()) +
                    ") does not match function arity (" + std::to_string(traits::arity) + ")"});
        }

        nlohmann::json schema;
        if constexpr (traits::arity == 0) {
            schema = nlohmann::json{{"type", "object"},
                                    {"properties", nlohmann::json::object()},
                                    {"required", nlohmann::json::array()}};
        } else {
            schema = tools::detail::build_properties<args_tuple>(param_names);
        }

        auto captured_names = param_names;
        tools::ToolHandler handler = [f = std::move(func), names = std::move(captured_names)](
                                         const nlohmann::json& args) -> Expected<nlohmann::json> {
            try {
                if constexpr (traits::arity == 0) {
                    auto result = f();
                    return tools::detail::wrap_result(std::move(result));
                } else {
                    auto result =
                        tools::detail::invoke_with_json<decltype(f), args_tuple>(f, args, names);
                    return tools::detail::wrap_result(std::move(result));
                }
            } catch (const nlohmann::json::exception& e) {
                return std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                             std::string("JSON argument error: ") + e.what()});
            } catch (const std::exception& e) {
                return std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                             std::string("Tool execution failed: ") + e.what()});
            }
        };

        return register_tool(name, description, std::move(schema), std::move(handler));
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

    Config config_;
    std::unique_ptr<Impl> impl_;
};

} // namespace zoo
