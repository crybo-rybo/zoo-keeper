/**
 * @file runtime.hpp
 * @brief Internal worker-owned runtime behind the public `zoo::Agent` facade.
 */

#pragma once

#include "backend.hpp"
#include "mailbox.hpp"
#include "request_tracker.hpp"
#include "zoo/agent.hpp"
#include <atomic>
#include <memory>
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
    AgentRuntime(const Config& cfg, std::unique_ptr<AgentBackend> backend);
    ~AgentRuntime();

    AgentRuntime(const AgentRuntime&) = delete;
    AgentRuntime& operator=(const AgentRuntime&) = delete;
    AgentRuntime(AgentRuntime&&) = delete;
    AgentRuntime& operator=(AgentRuntime&&) = delete;

    RequestHandle
    chat(Message message,
         std::optional<std::function<void(std::string_view)>> callback = std::nullopt);

    RequestHandle
    complete(std::vector<Message> messages,
             std::optional<std::function<void(std::string_view)>> callback = std::nullopt);

    void cancel(RequestId id);
    void set_system_prompt(const std::string& prompt);
    void stop();
    bool is_running() const noexcept;

    std::vector<Message> get_history() const;
    void clear_history();

    Expected<void> register_tool(tools::ToolDefinition definition);
    size_t tool_count() const noexcept;
    std::string build_tool_system_prompt(const std::string& base_prompt) const;

  private:
    void inference_loop();
    void handle_request(Request& request);
    void handle_command(Command& cmd);
    Expected<Response> process_request(const Request& request);

    void fail_pending(const Error& error);
    static void resolve_command_on_shutdown(Command& cmd);
    void update_tool_grammar();

    Config config_;
    std::unique_ptr<AgentBackend> backend_;
    tools::ToolRegistry tool_registry_;
    RequestTracker request_tracker_;
    mutable RuntimeMailbox request_mailbox_;
    std::thread inference_thread_;
    std::atomic<bool> running_{true};
    std::atomic<bool> tool_grammar_active_{false};
};

} // namespace zoo::internal::agent
