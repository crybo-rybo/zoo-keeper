#pragma once

#include "types.hpp"
#include "backend/interface.hpp"
#include "engine/history_manager.hpp"
#include "engine/template_engine.hpp"
#include "engine/request_queue.hpp"
#include "engine/agentic_loop.hpp"
#include <thread>
#include <future>
#include <memory>
#include <atomic>
#include <queue>

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
 * Agent agent = std::move(*agent_result);
 *
 * auto future = agent.chat(Message::user("Hello!"));
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
     * @return Expected<Agent> Agent instance or initialization error
     */
    static Expected<Agent> create(
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

        // Create agent with initialized backend
        return Agent(config, std::move(backend));
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
        stop();
    }

    // Non-copyable (Agent owns an active thread and shared resources)
    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;

    /**
     * @brief Move constructor
     *
     * Transfers ownership of all resources including the running inference thread.
     * The moved-from object is left in a valid but stopped state.
     *
     * Thread Safety: The inference thread continues running and now operates
     * on the new object's resources. The moved-from object's running_ flag
     * is set to false to prevent double-shutdown.
     *
     * @param other Agent to move from
     */
    Agent(Agent&& other) noexcept
        : config_(std::move(other.config_))
        , backend_(std::move(other.backend_))
        , history_(std::move(other.history_))
        , template_engine_(std::move(other.template_engine_))
        , request_queue_(std::move(other.request_queue_))
        , agentic_loop_(std::move(other.agentic_loop_))
        , inference_thread_(std::move(other.inference_thread_))
        , running_(other.running_.load())
        // Note: mutexes are default-constructed (cannot be moved)
    {
        other.running_.store(false);

        // Fix the mutex pointer in agentic_loop to point to the new object's mutex
        if (agentic_loop_) {
            agentic_loop_->set_history_mutex(&history_mutex_);
        }
    }

    /**
     * @brief Move assignment operator
     *
     * Stops the current agent's inference thread before moving resources.
     * The moved-from object is left in a valid but stopped state.
     *
     * Thread Safety: Calls stop() to ensure current inference thread is
     * joined before transferring resources from other.
     *
     * @param other Agent to move from
     * @return Reference to this
     */
    Agent& operator=(Agent&& other) noexcept {
        if (this != &other) {
            stop();  // Stop and join current inference thread

            config_ = std::move(other.config_);
            backend_ = std::move(other.backend_);
            history_ = std::move(other.history_);
            template_engine_ = std::move(other.template_engine_);
            request_queue_ = std::move(other.request_queue_);
            agentic_loop_ = std::move(other.agentic_loop_);
            inference_thread_ = std::move(other.inference_thread_);
            running_.store(other.running_.load());
            other.running_.store(false);
            // Note: mutexes remain in place (cannot be moved)

            // Fix the mutex pointer in agentic_loop to point to the new object's mutex
            if (agentic_loop_) {
                agentic_loop_->set_history_mutex(&history_mutex_);
            }
        }
        return *this;
    }

    /**
     * @brief Submit a chat message and get a future for the response
     *
     * This is the main public API for inference. It:
     * - Creates a request with the message and optional callback
     * - Pushes to the request queue
     * - Returns a future that will be fulfilled when inference completes
     *
     * The callback (if provided) executes on the inference thread.
     * The caller is responsible for thread-safety if accessing shared state.
     *
     * @param message User message to send
     * @param callback Optional streaming token callback (runs on inference thread)
     * @return std::future<Expected<Response>> Future response
     */
    std::future<Expected<Response>> chat(
        Message message,
        std::optional<std::function<void(std::string_view)>> callback = std::nullopt
    ) {
        // Create promise for result
        auto promise = std::make_shared<std::promise<Expected<Response>>>();
        std::future<Expected<Response>> future = promise->get_future();

        // Check if running
        if (!running_.load(std::memory_order_acquire)) {
            promise->set_value(tl::unexpected(Error{
                ErrorCode::AgentNotRunning,
                "Agent is not running"
            }));
            return future;
        }

        // Create request
        Request request(std::move(message), std::move(callback));

        // Push to request queue first (before adding promise)
        // This prevents race where inference thread could pop promise before request is queued
        if (!request_queue_->push(std::move(request))) {
            promise->set_value(tl::unexpected(Error{
                ErrorCode::QueueFull,
                "Request queue is full or agent is shutting down"
            }));
            return future;
        }

        // Store promise for retrieval by inference thread
        // At this point, request is guaranteed to be in queue
        {
            std::lock_guard<std::mutex> lock(promises_mutex_);
            pending_promises_.push(promise);
        }

        return future;
    }

    /**
     * @brief Set or update the system prompt
     *
     * Updates the system message in conversation history.
     * Thread-safe: Can be called from any thread at any time.
     *
     * @param prompt System prompt text
     */
    void set_system_prompt(const std::string& prompt) {
        std::lock_guard<std::mutex> lock(history_mutex_);
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
     *
     * @return std::vector<Message> Copy of message history
     */
    std::vector<Message> get_history() const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        return history_->get_messages();
    }

    /**
     * @brief Clear conversation history
     *
     * Thread-safe: Can be called from any thread at any time.
     */
    void clear_history() {
        std::lock_guard<std::mutex> lock(history_mutex_);
        history_->clear();
    }

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
        , history_(std::make_shared<engine::HistoryManager>(config.context_size))
        , template_engine_(std::make_shared<engine::TemplateEngine>(
            config.prompt_template,
            config.custom_template
        ))
        , request_queue_(std::make_shared<engine::RequestQueue>())
        , agentic_loop_(std::make_shared<engine::AgenticLoop>(
            backend_,
            history_,
            template_engine_,
            config,
            &history_mutex_
        ))
        , running_(true)
    {
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
            // Pop request (blocking)
            auto request_opt = request_queue_->pop();

            // Check shutdown
            if (!request_opt) {
                break;  // Queue shutdown
            }

            // Get corresponding promise
            std::shared_ptr<std::promise<Expected<Response>>> promise;
            {
                std::lock_guard<std::mutex> lock(promises_mutex_);
                if (!pending_promises_.empty()) {
                    promise = pending_promises_.front();
                    pending_promises_.pop();
                }
            }

            // Process request
            auto result = agentic_loop_->process_request(*request_opt);

            // Fulfill promise
            if (promise) {
                promise->set_value(std::move(result));
            }
        }

        // Fulfill remaining promises with shutdown error
        std::lock_guard<std::mutex> lock(promises_mutex_);
        while (!pending_promises_.empty()) {
            auto promise = pending_promises_.front();
            pending_promises_.pop();
            promise->set_value(tl::unexpected(Error{
                ErrorCode::AgentNotRunning,
                "Agent stopped before request could be processed"
            }));
        }
    }

    // Configuration
    Config config_;

    // Components (shared ownership for thread safety)
    std::shared_ptr<backend::IBackend> backend_;
    std::shared_ptr<engine::HistoryManager> history_;
    std::shared_ptr<engine::TemplateEngine> template_engine_;
    std::shared_ptr<engine::RequestQueue> request_queue_;
    std::shared_ptr<engine::AgenticLoop> agentic_loop_;

    // Threading
    std::thread inference_thread_;
    std::atomic<bool> running_;

    // Synchronization
    std::mutex promises_mutex_;
    mutable std::mutex history_mutex_;  // Protects history_ from concurrent access
    std::queue<std::shared_ptr<std::promise<Expected<Response>>>> pending_promises_;
};

} // namespace zoo
