#pragma once

#include "zoo/backend/IBackend.hpp"
#include <map>
#include <queue>
#include <sstream>
#include <thread>
#include <chrono>

namespace zoo {
namespace testing {

/**
 * @brief Mock backend for unit testing
 *
 * Simulates llama.cpp behavior without actual model inference.
 * Supports:
 * - Configurable responses (pre-programmed or echoing input)
 * - Token-by-token streaming simulation
 * - Token counting estimation
 * - Error injection for negative testing
 * - KV cache state tracking
 */
class MockBackend : public backend::IBackend {
public:
    // Configuration
    bool should_fail_initialize = false;
    bool should_fail_generate = false;
    std::string error_message = "Mock error";

    // State tracking
    bool initialized = false;
    Config last_config;
    std::vector<int> last_prompt_tokens;
    int kv_cache_tokens = 0;
    int context_size = 8192;
    int training_context_size = 4096;
    int vocab_size = 32000;
    int clear_kv_cache_calls = 0;
    std::string last_formatted_prompt;

    // Response control
    enum class ResponseMode {
        Echo,           // Echo the input prompt
        Fixed,          // Return pre-programmed response
        TokenByToken    // Simulate streaming
    };

    ResponseMode mode = ResponseMode::Fixed;
    std::queue<std::string> response_queue;  // For ResponseMode::Fixed
    std::string default_response = "This is a test response.";

    // Callbacks tracking (for verification)
    int token_callback_count = 0;
    std::vector<std::string> streamed_tokens;

    // Delay control for testing queue backpressure
    int generation_delay_ms = 0;                             ///< Artificial delay in generate() (ms)

    Expected<void> initialize(const Config& config) override {
        if (should_fail_initialize) {
            return tl::unexpected(Error{ErrorCode::BackendInitFailed, error_message});
        }

        initialized = true;
        last_config = config;
        context_size = config.context_size;
        return {};
    }

    Expected<std::vector<int>> tokenize(const std::string& text) override {
        // Simple estimation: ~4 chars per token
        std::vector<int> tokens;
        int num_tokens = std::max(1, static_cast<int>(text.length() / 4));

        // Generate dummy token IDs
        for (int i = 0; i < num_tokens; ++i) {
            tokens.push_back(100 + i);
        }

        return tokens;
    }

    Expected<std::string> generate(
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        const std::vector<std::string>& stop_sequences,
        const std::optional<std::function<void(std::string_view)>>& on_token = std::nullopt
    ) override {
        if (!initialized) {
            return tl::unexpected(Error{ErrorCode::BackendInitFailed, "Backend not initialized"});
        }

        if (should_fail_generate) {
            return tl::unexpected(Error{ErrorCode::InferenceFailed, error_message});
        }

        // Artificial delay for testing queue backpressure
        if (generation_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(generation_delay_ms));
        }

        last_prompt_tokens = prompt_tokens;
        kv_cache_tokens += static_cast<int>(prompt_tokens.size());

        std::string response;

        // Determine response based on mode
        switch (mode) {
            case ResponseMode::Echo: {
                response = "Echo: " + std::to_string(prompt_tokens.size()) + " tokens";
                break;
            }

            case ResponseMode::Fixed: {
                if (!response_queue.empty()) {
                    response = response_queue.front();
                    response_queue.pop();
                } else {
                    response = default_response;
                }
                break;
            }

            case ResponseMode::TokenByToken: {
                response = default_response;
                break;
            }
        }

        // Check stop sequences and trim before streaming (simple substring check)
        for (const auto& stop : stop_sequences) {
            size_t pos = response.find(stop);
            if (pos != std::string::npos) {
                response = response.substr(0, pos);
                break;
            }
        }

        // Simulate streaming if callback provided — only stream the trimmed response
        if (on_token.has_value()) {
            std::istringstream iss(response);
            std::string word;
            streamed_tokens.clear();

            while (iss >> word) {
                word += " ";  // Add space back
                streamed_tokens.push_back(word);
                (*on_token)(word);
                token_callback_count++;
            }
        }

        // Respect max_tokens (roughly)
        kv_cache_tokens += std::min(max_tokens, static_cast<int>(response.length() / 4));

        return response;
    }

    int get_kv_cache_token_count() const override {
        return kv_cache_tokens;
    }

    void clear_kv_cache() override {
        kv_cache_tokens = 0;
        ++clear_kv_cache_calls;
    }

    Expected<std::string> format_prompt(const std::vector<Message>& messages) override {
        // Simple concatenation for testing
        std::ostringstream out;
        for (const auto& msg : messages) {
            out << role_to_string(msg.role) << ": " << msg.content << "\n";
        }
        last_formatted_prompt = out.str();
        return last_formatted_prompt;
    }

    void finalize_response(const std::vector<Message>&) override {
        // No-op for mock
    }

    int get_context_size() const override {
        return context_size;
    }

    int get_training_context_size() const override {
        return training_context_size;
    }

    int get_vocab_size() const override {
        return vocab_size;
    }

    // Test helpers
    void enqueue_response(const std::string& response) {
        response_queue.push(response);
    }

    void reset() {
        initialized = false;
        kv_cache_tokens = 0;
        token_callback_count = 0;
        streamed_tokens.clear();
        last_prompt_tokens.clear();
        clear_kv_cache_calls = 0;
        last_formatted_prompt.clear();
        while (!response_queue.empty()) response_queue.pop();
    }
};

} // namespace testing
} // namespace zoo
