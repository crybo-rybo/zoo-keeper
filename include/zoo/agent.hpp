#pragma once

#include "types.hpp"
#include "backend/IBackend.hpp"
#include "engine/history_manager.hpp"
#include "engine/request_queue.hpp"
#include "engine/tool_registry.hpp"
#include "engine/agentic_loop.hpp"
#ifdef ZOO_ENABLE_MCP
#include "mcp/mcp_client.hpp"
#endif
#include <thread>
#include <future>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <algorithm>

namespace zoo {

/**
 * @brief Main entry point for Zoo-Keeper Agent Engine
 *
 * The Agent class provides a high-level, thread-safe interface for LLM inference.
 * It manages an inference thread that processes requests asynchronously.
 *
 * Key Features (MVP):
 * - Asynchronous chat interface with std::future
 * - Conversation history management
 * - Configurable prompt templates
 * - Streaming token callbacks
 * - Graceful shutdown
 *
 * Thread Model:
 * - Calling Thread: Calls chat(), receives std::future<Response>
 * - Inference Thread: Processes requests, executes backend->generate()
 * - Callbacks: Execute on inference thread (user must handle synchronization)
 *
 * Example Usage:
 * @code
 * Config config;
 * config.model_path = "model.gguf";
 * config.context_size = 8192;
 * config.max_tokens = 512;
 *
 * auto agent_result = Agent::create(config);
 * if (!agent_result) {
 *     std::cerr << "Failed to create agent: " << agent_result.error().to_string() << std::endl;
 *     return 1;
 * }
 *
 * auto agent = std::move(*agent_result);  // Now a std::unique_ptr<Agent>
 *
 * auto future = agent->chat(Message::user("Hello!"));
 * auto response = future.get();
 * if (response) {
 *     std::cout << response->text << std::endl;
 * }
 * @endcode
 */
class Agent {
public:
    /**
     * @brief Factory method to create an Agent
     *
     * Validates configuration, initializes backend, and starts inference thread.
     *
     * @param config Agent configuration
     * @param backend Optional backend (for testing with MockBackend)
     * @return Expected<std::unique_ptr<Agent>> Unique pointer to Agent or initialization error
     */
    static Expected<std::unique_ptr<Agent>> create(
        const Config& config,
        std::unique_ptr<backend::IBackend> backend = nullptr
    ) {
        // Validate configuration
        if (auto result = config.validate(); !result) {
            return tl::unexpected(result.error());
        }

        // Create backend if not provided
        if (!backend) {
            backend = backend::create_backend();
            if (!backend) {
                return tl::unexpected(Error{
                    ErrorCode::BackendInitFailed,
                    "Failed to create backend"
                });
            }
        }

        // Initialize backend
        if (auto result = backend->initialize(config); !result) {
            return tl::unexpected(result.error());
        }

        // Create agent with initialized backend (use unique_ptr with private constructor)
        return std::unique_ptr<Agent>(new Agent(config, std::move(backend)));
    }

    /**
     * @brief Destructor - stops inference thread and waits for completion
     *
     * Ensures graceful shutdown by:
     * - Signaling running_ flag to false
     * - Shutting down request queue
     * - Cancelling agentic loop
     * - Joining inference thread (blocks until thread completes)
     */
    ~Agent() {
#ifdef ZOO_ENABLE_MCP
        // Disconnect MCP clients before stopping inference thread
        std::lock_guard<std::mutex> lock(mcp_mutex_);
        for (auto& client : mcp_clients_) {
            client->disconnect();
        }
        mcp_clients_.clear();
#endif
        stop();
    }

    // Non-copyable and non-movable (inference thread captures `this`)
    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;
    Agent(Agent&&) = delete;
    Agent& operator=(Agent&&) = delete;

    /**
     * @brief Submit a chat message and get a handle for the response
     *
     * This is the main public API for inference. It:
     * - Creates a request with the message and optional callback
     * - Pushes to the request queue
     * - Returns a RequestHandle with a unique ID and future
     *
     * The callback (if provided) executes on the inference thread.
     * The caller is responsible for thread-safety if accessing shared state.
     *
     * @param message User message to send
     * @param callback Optional streaming token callback (runs on inference thread)
     * @return RequestHandle Handle with request ID and future response
     */
    RequestHandle chat(
        Message message,
        std::optional<std::function<void(std::string_view)>> callback = std::nullopt
    ) {
        return chat(std::move(message), ChatOptions{}, std::move(callback));
    }

    /**
     * @brief Submit a chat message with per-request options (e.g., RAG)
     *
     * @param message User message to send
     * @param options Per-request behavior options
     * @param callback Optional streaming token callback (runs on inference thread)
     * @return RequestHandle Handle with request ID and future response
     */
    RequestHandle chat(
        Message message,
        ChatOptions options,
        std::optional<std::function<void(std::string_view)>> callback = std::nullopt
    ) {
        // Create promise for result and bundle it with the request
        auto promise = std::make_shared<std::promise<Expected<Response>>>();
        std::future<Expected<Response>> future = promise->get_future();

        // Assign unique request ID
        RequestId request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);

        // Check if running
        if (!running_.load(std::memory_order_acquire)) {
            promise->set_value(tl::unexpected(Error{
                ErrorCode::AgentNotRunning,
                "Agent is not running"
            }));
            return RequestHandle{request_id, std::move(future)};
        }

        // Create request with bundled promise and cancellation token
        // Bundled promise eliminates race condition (issue #20)
        Request request(std::move(message), std::move(options), std::move(callback));
        request.promise = promise;
        request.id = request_id;

        // Store cancellation token for cancel() API
        {
            std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
            cancel_tokens_[request_id] = request.cancelled;
        }

        // Push to request queue (promise travels with the request atomically)
        if (!request_queue_->push(std::move(request))) {
            promise->set_value(tl::unexpected(Error{
                ErrorCode::QueueFull,
                "Request queue is full or agent is shutting down"
            }));
            // Clean up cancel token
            {
                std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
                cancel_tokens_.erase(request_id);
            }
            return RequestHandle{request_id, std::move(future)};
        }

        // Promise travels with the request atomically (no separate promise queue needed)
        return RequestHandle{request_id, std::move(future)};
    }

    /**
     * @brief Cancel a specific chat request by ID
     *
     * Sets the per-request cancellation flag. If the request is queued,
     * it will be skipped when dequeued. If it is in-flight, the agentic
     * loop will detect the flag and abort at the next iteration boundary.
     *
     * Canceling an already-completed or unknown request is a no-op.
     *
     * @param id The request ID from the RequestHandle returned by chat()
     */
    void cancel(RequestId id) {
        std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
        auto it = cancel_tokens_.find(id);
        if (it != cancel_tokens_.end()) {
            it->second->store(true, std::memory_order_release);
        }
    }

    /**
     * @brief Set or update the system prompt
     *
     * Updates the system message in conversation history.
     * Thread-safe: Can be called from any thread at any time.
     * HistoryManager provides internal locking.
     *
     * @param prompt System prompt text
     */
    void set_system_prompt(const std::string& prompt) {
        history_->set_system_prompt(prompt);
    }

    /**
     * @brief Stop the agent and wait for inference thread to finish
     *
     * Gracefully shuts down:
     * - Signals request queue to stop accepting new requests
     * - Cancels ongoing inference loop
     * - Waits for inference thread to complete
     * - Can be called multiple times safely
     */
    void stop() {
        if (!running_.load(std::memory_order_acquire)) {
            return;  // Already stopped
        }

        // Signal shutdown
        running_.store(false, std::memory_order_release);
        request_queue_->shutdown();
        agentic_loop_->cancel();

        // Wait for thread
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
    }

    /**
     * @brief Check if agent is running
     *
     * @return bool True if inference thread is active
     */
    bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get current configuration
     *
     * @return const Config& Agent configuration
     */
    const Config& get_config() const {
        return config_;
    }

    /**
     * @brief Get conversation history
     *
     * Thread-safe: Returns a copy of the current conversation history.
     * Can be called from any thread at any time.
     * HistoryManager provides internal locking.
     *
     * @return std::vector<Message> Copy of message history
     */
    std::vector<Message> get_history() const {
        return history_->get_messages();
    }

    /**
     * @brief Clear conversation history
     *
     * Thread-safe: Can be called from any thread at any time.
     * HistoryManager provides internal locking.
     */
    void clear_history() {
        history_->clear();
    }

    /**
     * @brief Register a tool with automatic schema generation
     *
     * Registers a callable as a tool that the model can invoke during inference.
     * Parameter types are extracted from the function signature and used to
     * generate a JSON schema.
     *
     * Supported parameter types: int, float, double, bool, std::string
     *
     * @param name Tool name (used by model to invoke)
     * @param description Human-readable description of what the tool does
     * @param param_names Names for each parameter (must match function arity)
     * @param func Callable to invoke when tool is called
     */
    template<typename Func>
    void register_tool(const std::string& name, const std::string& description,
                       const std::vector<std::string>& param_names, Func func) {
        tool_registry_->register_tool(name, description, param_names, std::move(func));
    }

    /**
     * @brief Get the number of registered tools
     */
    size_t tool_count() const {
        return tool_registry_->size();
    }

    /**
     * @brief Configure the retriever used for RAG requests.
     */
    void set_retriever(std::shared_ptr<engine::IRetriever> retriever) {
        agentic_loop_->set_retriever(std::move(retriever));
    }

    /**
     * @brief Configure a durable SQLite context database for long conversations.
     *
     * When configured, old history is archived automatically and retrieved context
     * is injected ephemerally on future turns.
     */
    void set_context_database(std::shared_ptr<engine::ContextDatabase> context_database) {
        agentic_loop_->set_context_database(std::move(context_database));
    }

    /**
     * @brief Open and install a durable SQLite context database.
     */
    Expected<void> enable_context_database(const std::string& path) {
        auto db_result = engine::ContextDatabase::open(path);
        if (!db_result) {
            return tl::unexpected(db_result.error());
        }
        set_context_database(std::move(*db_result));
        return {};
    }

#ifdef ZOO_ENABLE_MCP
    struct McpServerSummary {
        std::string server_id;
        bool connected = false;
        size_t discovered_tool_count = 0;
    };

    /**
     * @brief Connect to an MCP server and register its tools.
     *
     * Creates an McpClient, connects to the server, discovers tools,
     * and registers them into the existing ToolRegistry. MCP tools are
     * prefixed with "mcp_<server_id>:" to avoid name collisions with
     * locally registered tools.
     *
     * @param config MCP client configuration (server ID, transport, session settings)
     * @return Expected<void> Success or error
     */
    Expected<void> add_mcp_server(const mcp::McpClient::Config& config) {
        std::lock_guard<std::mutex> lock(mcp_mutex_);
        auto duplicate = std::find_if(
            mcp_clients_.begin(), mcp_clients_.end(),
            [&config](const auto& client) {
                return client && client->get_server_id() == config.server_id;
            });
        if (duplicate != mcp_clients_.end()) {
            return tl::unexpected(Error{
                ErrorCode::McpSessionFailed,
                "MCP server_id already connected: " + config.server_id
            });
        }

        auto client_result = mcp::McpClient::create(config);
        if (!client_result) {
            return tl::unexpected(client_result.error());
        }

        auto client = std::move(*client_result);

        auto connect_result = client->connect();
        if (!connect_result) {
            return tl::unexpected(connect_result.error());
        }

        auto register_result = client->register_tools_with(*tool_registry_);
        if (!register_result) {
            client->disconnect();
            return tl::unexpected(register_result.error());
        }

        mcp_clients_.push_back(std::move(client));
        return {};
    }

    /**
     * @brief Get the number of connected MCP servers.
     */
    size_t mcp_server_count() const {
        std::lock_guard<std::mutex> lock(mcp_mutex_);
        return mcp_clients_.size();
    }

    /**
     * @brief Remove a connected MCP server by ID.
     */
    Expected<void> remove_mcp_server(const std::string& server_id) {
        std::lock_guard<std::mutex> lock(mcp_mutex_);
        auto it = std::find_if(
            mcp_clients_.begin(), mcp_clients_.end(),
            [&server_id](const auto& client) {
                return client && client->get_server_id() == server_id;
            });
        if (it == mcp_clients_.end()) {
            return tl::unexpected(Error{
                ErrorCode::McpToolNotAvailable,
                "MCP server not found: " + server_id
            });
        }

        (*it)->disconnect();
        mcp_clients_.erase(it);
        return {};
    }

    /**
     * @brief List connected MCP server summaries.
     */
    std::vector<McpServerSummary> list_mcp_servers() const {
        std::lock_guard<std::mutex> lock(mcp_mutex_);
        std::vector<McpServerSummary> servers;
        servers.reserve(mcp_clients_.size());
        for (const auto& client : mcp_clients_) {
            if (!client) {
                continue;
            }
            servers.push_back(McpServerSummary{
                client->get_server_id(),
                client->is_connected(),
                client->get_discovered_tools().size()
            });
        }
        return servers;
    }

    /**
     * @brief Get one MCP server summary by ID.
     */
    Expected<McpServerSummary> get_mcp_server(const std::string& server_id) const {
        std::lock_guard<std::mutex> lock(mcp_mutex_);
        auto it = std::find_if(
            mcp_clients_.begin(), mcp_clients_.end(),
            [&server_id](const auto& client) {
                return client && client->get_server_id() == server_id;
            });
        if (it == mcp_clients_.end()) {
            return tl::unexpected(Error{
                ErrorCode::McpToolNotAvailable,
                "MCP server not found: " + server_id
            });
        }

        return McpServerSummary{
            (*it)->get_server_id(),
            (*it)->is_connected(),
            (*it)->get_discovered_tools().size()
        };
    }
#endif

private:
    /**
     * @brief Private constructor - use create() factory method
     *
     * @param config Configuration
     * @param backend Backend implementation
     */
    Agent(const Config& config, std::unique_ptr<backend::IBackend> backend)
        : config_(config)
        , backend_(std::shared_ptr<backend::IBackend>(std::move(backend)))
        , history_(std::make_shared<engine::HistoryManager>(
            config.context_size,
            [b = backend_.get()](const std::string& text) -> int {
                auto result = b->tokenize(text);
                if (result && !result->empty()) {
                    return static_cast<int>(result->size());
                }
                return std::max(1, static_cast<int>(text.length() / 4));
            }
        ))
        , request_queue_(std::make_shared<engine::RequestQueue>(config.request_queue_capacity))
        , tool_registry_(std::make_shared<engine::ToolRegistry>())
        , agentic_loop_(std::make_shared<engine::AgenticLoop>(
            backend_,
            history_,
            config
        ))
        , running_(true)
    {
        // Wire tool registry to agentic loop
        agentic_loop_->set_tool_registry(tool_registry_);

        // Set initial system prompt if provided
        if (config.system_prompt) {
            history_->set_system_prompt(*config.system_prompt);
        }

        // Start inference thread
        inference_thread_ = std::thread([this]() {
            inference_loop();
        });
    }

    /**
     * @brief Main inference thread loop
     *
     * Processes requests from the queue until shutdown.
     */
    void inference_loop() {
        while (running_.load(std::memory_order_acquire)) {
            // Pop request (blocking) — promise is bundled with the request
            auto request_opt = request_queue_->pop();

            // Check shutdown
            if (!request_opt) {
                break;  // Queue shutdown
            }

            // Extract promise from request (guaranteed to exist for queued requests)
            auto promise = request_opt->promise;

            // Check per-request cancellation before processing
            if (request_opt->cancelled &&
                request_opt->cancelled->load(std::memory_order_acquire)) {
                if (promise) {
                    promise->set_value(tl::unexpected(Error{
                        ErrorCode::RequestCancelled,
                        "Request cancelled"
                    }));
                }
                // Clean up cancel token
                {
                    std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
                    cancel_tokens_.erase(request_opt->id);
                }
                continue;
            }

            // Process request (pass cancellation token to agentic loop)
            auto result = agentic_loop_->process_request(*request_opt, request_opt->cancelled);

            // Clean up cancel token
            {
                std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
                cancel_tokens_.erase(request_opt->id);
            }

            // Fulfill promise — always succeeds since promise travels with request
            if (promise) {
                promise->set_value(std::move(result));
            }
        }

        // Drain any remaining queued requests and fulfill their promises
        while (auto remaining = request_queue_->pop()) {
            if (remaining->promise) {
                remaining->promise->set_value(tl::unexpected(Error{
                    ErrorCode::AgentNotRunning,
                    "Agent stopped before request could be processed"
                }));
            }
        }
    }

    // Configuration
    Config config_;

    // Components (shared ownership for thread safety)
    std::shared_ptr<backend::IBackend> backend_;
    std::shared_ptr<engine::HistoryManager> history_;
    std::shared_ptr<engine::RequestQueue> request_queue_;
    std::shared_ptr<engine::ToolRegistry> tool_registry_;
    std::shared_ptr<engine::AgenticLoop> agentic_loop_;

    // Threading
    std::thread inference_thread_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> next_request_id_{1};  // Monotonically increasing request IDs

    // Per-request cancellation
    std::mutex cancel_tokens_mutex_;
    std::unordered_map<RequestId, std::shared_ptr<std::atomic<bool>>> cancel_tokens_;

#ifdef ZOO_ENABLE_MCP
    // MCP clients (one per connected server)
    mutable std::mutex mcp_mutex_;
    std::vector<std::shared_ptr<mcp::McpClient>> mcp_clients_;
#endif
};

} // namespace zoo
