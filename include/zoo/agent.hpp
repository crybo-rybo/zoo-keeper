/**
 * @file agent.hpp
 * @brief Asynchronous orchestration layer that coordinates model inference and tool execution.
 */

#pragma once

#include "core/types.hpp"
#include "tools/registry.hpp"
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

/**
 * @brief Move-only async handle for one queued agent request.
 */
template <typename Result> class RequestHandle {
  public:
    using AwaitFn = Expected<Result> (*)(void*, uint32_t, uint32_t);
    using ReadyFn = bool (*)(const void*, uint32_t, uint32_t);
    using ReleaseFn = void (*)(void*, uint32_t, uint32_t);

    RequestHandle() noexcept = default;

    RequestHandle(RequestId id, std::shared_ptr<void> state, uint32_t slot, uint32_t generation,
                  AwaitFn await_fn, ReadyFn ready_fn, ReleaseFn release_fn) noexcept
        : id_(id), state_(std::move(state)), slot_(slot), generation_(generation), await_(await_fn),
          ready_(ready_fn), release_(release_fn) {}

    ~RequestHandle() {
        reset();
    }

    RequestHandle(RequestHandle&& other) noexcept {
        *this = std::move(other);
    }
    RequestHandle& operator=(RequestHandle&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        reset();

        id_ = other.id_;
        state_ = std::move(other.state_);
        slot_ = other.slot_;
        generation_ = other.generation_;
        await_ = other.await_;
        ready_ = other.ready_;
        release_ = other.release_;

        other.id_ = 0;
        other.slot_ = 0;
        other.generation_ = 0;
        other.await_ = nullptr;
        other.ready_ = nullptr;
        other.release_ = nullptr;
        return *this;
    }

    RequestHandle(const RequestHandle&) = delete;
    RequestHandle& operator=(const RequestHandle&) = delete;

    [[nodiscard]] RequestId id() const noexcept {
        return id_;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return valid();
    }

    [[nodiscard]] bool valid() const noexcept {
        return static_cast<bool>(state_) && await_ != nullptr && release_ != nullptr;
    }

    [[nodiscard]] bool ready() const {
        return valid() && ready_(state_.get(), slot_, generation_);
    }

    Expected<Result> await_result() {
        if (!valid()) {
            return std::unexpected(
                Error{ErrorCode::AgentNotRunning, "Request handle is no longer valid"});
        }

        auto result = await_(state_.get(), slot_, generation_);
        id_ = 0;
        state_.reset();
        slot_ = 0;
        generation_ = 0;
        await_ = nullptr;
        ready_ = nullptr;
        release_ = nullptr;
        return result;
    }

    void reset() noexcept {
        if (!valid()) {
            return;
        }
        release_(state_.get(), slot_, generation_);
        id_ = 0;
        state_.reset();
        slot_ = 0;
        generation_ = 0;
        await_ = nullptr;
        ready_ = nullptr;
        release_ = nullptr;
    }

  private:
    RequestId id_ = 0;
    std::shared_ptr<void> state_;
    uint32_t slot_ = 0;
    uint32_t generation_ = 0;
    AwaitFn await_ = nullptr;
    ReadyFn ready_ = nullptr;
    ReleaseFn release_ = nullptr;
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

    /// Clears history synchronously on the inference thread before later queued work.
    void clear_history();

    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 std::initializer_list<std::string> param_names, Func func) {
        return register_tool(name, description, std::vector<std::string>(param_names),
                             std::move(func));
    }

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

    template <typename Handler>
        requires tools::detail::is_json_handler_like_v<Handler>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 const nlohmann::json& schema, Handler handler) {
        return register_tool(name, description, nlohmann::json(schema),
                             tools::ToolHandler(std::move(handler)));
    }

    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 nlohmann::json schema, tools::ToolHandler handler);

    [[nodiscard]] size_t tool_count() const noexcept;

  private:
    struct Impl;

    Agent(ModelConfig model_config, AgentConfig agent_config, GenerationOptions default_generation,
          std::unique_ptr<Impl> impl);
    Expected<void> register_tool(tools::ToolDefinition definition);

    ModelConfig model_config_;
    AgentConfig agent_config_;
    GenerationOptions default_generation_options_;
    std::unique_ptr<Impl> impl_;
};

} // namespace zoo
