/**
 * @file runtime.hpp
 * @brief Internal worker-owned runtime behind the public `zoo::Agent` facade.
 */

#pragma once

#include "backend.hpp"
#include "callback_dispatcher.hpp"
#include "mailbox.hpp"
#include "request_slots.hpp"
#include "zoo/agent.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>
#include <thread>

namespace zoo::internal::agent {

/**
 * @brief Internal runtime that owns the inference thread and orchestration loop.
 *
 * All backend/model access happens on the inference thread. Calling-thread
 * operations that need backend state are routed through the command lane.
 */
class AgentRuntime {
  public:
    AgentRuntime(ModelConfig model_config, AgentConfig agent_config,
                 GenerationOptions default_generation, std::unique_ptr<AgentBackend> backend);
    ~AgentRuntime();

    AgentRuntime(const AgentRuntime&) = delete;
    AgentRuntime& operator=(const AgentRuntime&) = delete;
    AgentRuntime(AgentRuntime&&) = delete;
    AgentRuntime& operator=(AgentRuntime&&) = delete;

    RequestHandle<TextResponse> chat(std::string_view user_message,
                                     const GenerationOptions& options = GenerationOptions{},
                                     AsyncTextCallback callback = {});
    RequestHandle<TextResponse> chat(MessageView message,
                                     const GenerationOptions& options = GenerationOptions{},
                                     AsyncTextCallback callback = {});
    RequestHandle<TextResponse> complete(ConversationView messages,
                                         const GenerationOptions& options = GenerationOptions{},
                                         AsyncTextCallback callback = {});
    RequestHandle<ExtractionResponse>
    extract(const nlohmann::json& output_schema, std::string_view user_message,
            const GenerationOptions& options = GenerationOptions{},
            AsyncTextCallback callback = {});
    RequestHandle<ExtractionResponse>
    extract(const nlohmann::json& output_schema, MessageView message,
            const GenerationOptions& options = GenerationOptions{},
            AsyncTextCallback callback = {});
    RequestHandle<ExtractionResponse>
    extract(const nlohmann::json& output_schema, ConversationView messages,
            const GenerationOptions& options = GenerationOptions{},
            AsyncTextCallback callback = {});

    void cancel(RequestId id);
    void set_system_prompt(std::string_view prompt);
    Expected<void> set_system_prompt(std::string_view prompt, std::chrono::nanoseconds timeout);
    Expected<void> add_system_message(std::string_view message);
    Expected<void> add_system_message(std::string_view message, std::chrono::nanoseconds timeout);
    void stop();
    bool is_running() const noexcept;

    HistorySnapshot get_history() const;
    Expected<HistorySnapshot> get_history(std::chrono::nanoseconds timeout) const;
    void clear_history();
    Expected<void> clear_history(std::chrono::nanoseconds timeout);

    Expected<void> register_tool(tools::ToolDefinition definition);
    Expected<void> register_tool(tools::ToolDefinition definition,
                                 std::chrono::nanoseconds timeout);
    Expected<void> register_tools(std::vector<tools::ToolDefinition> definitions);
    Expected<void> register_tools(std::vector<tools::ToolDefinition> definitions,
                                  std::chrono::nanoseconds timeout);
    size_t tool_count() const noexcept;

  private:
    void inference_loop();
    void handle_request(QueuedRequest request);
    void handle_command(Command& cmd);
    Expected<TextResponse> process_request(const ActiveRequest& request);
    Expected<ExtractionResponse> process_extraction_request(const ActiveRequest& request);

    void fail_pending(const Error& error);
    static void resolve_command_on_shutdown(Command& cmd);
    bool refresh_tool_calling_state();
    void enforce_history_limit();
    template <typename Result> RequestHandle<Result> make_immediate_error_handle(Error error);
    template <typename Result> RequestHandle<Result> enqueue_request(RequestPayload payload);

    /// Generic sync-command helper: build a Command carrying a fresh promise<Expected<R>>,
    /// push it on the mailbox, optionally bound by a timeout, and return the resolved result.
    template <typename Result, typename Maker>
    Expected<Result> send_sync_command(Maker&& make_cmd,
                                       std::optional<std::chrono::nanoseconds> timeout,
                                       std::string_view name);

    Expected<void> set_system_prompt_impl(std::string prompt,
                                          std::optional<std::chrono::nanoseconds> timeout);
    Expected<void> add_system_message_impl(std::string message,
                                           std::optional<std::chrono::nanoseconds> timeout);
    Expected<HistorySnapshot>
    get_history_impl(std::optional<std::chrono::nanoseconds> timeout) const;
    Expected<void> clear_history_impl(std::optional<std::chrono::nanoseconds> timeout);
    Expected<void> register_tool_impl(tools::ToolDefinition definition,
                                      std::optional<std::chrono::nanoseconds> timeout);
    Expected<void> register_tools_impl(std::vector<tools::ToolDefinition> definitions,
                                       std::optional<std::chrono::nanoseconds> timeout);

    GenerationOptions resolve_generation_options(const GenerationOptions& overrides) const;

    ModelConfig model_config_;
    AgentConfig agent_config_;
    GenerationOptions default_generation_options_;
    std::unique_ptr<AgentBackend> backend_;
    tools::ToolRegistry tool_registry_;
    std::shared_ptr<RequestSlots> request_slots_;
    mutable RuntimeMailbox request_mailbox_;
    std::thread inference_thread_;
    std::atomic<bool> running_{true};
    std::atomic<bool> tool_grammar_active_{false};
    CallbackDispatcher callback_dispatcher_;
};

} // namespace zoo::internal::agent
