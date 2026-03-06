#pragma once

#include "zoo/core/model.hpp"
#include <queue>
#include <sstream>
#include <thread>
#include <chrono>

namespace zoo::testing {

class MockBackend : public core::IBackend {
public:
    bool should_fail_initialize = false;
    bool should_fail_generate = false;
    std::string error_message = "Mock error";

    bool initialized = false;
    Config last_config;
    std::vector<int> last_prompt_tokens;
    int context_size = 8192;
    int clear_kv_cache_calls = 0;
    std::string last_formatted_prompt;

    enum class ResponseMode { Echo, Fixed, TokenByToken };
    ResponseMode mode = ResponseMode::Fixed;
    std::queue<std::string> response_queue;
    std::string default_response = "This is a test response.";

    int token_callback_count = 0;
    std::vector<std::string> streamed_tokens;
    int generation_delay_ms = 0;

    Expected<void> initialize(const Config& config) override {
        if (should_fail_initialize) {
            return std::unexpected(Error{ErrorCode::BackendInitFailed, error_message});
        }
        initialized = true;
        last_config = config;
        context_size = config.context_size;
        return {};
    }

    Expected<std::vector<int>> tokenize(const std::string& text) override {
        std::vector<int> tokens;
        int num_tokens = std::max(1, static_cast<int>(text.length() / 4));
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
            return std::unexpected(Error{ErrorCode::BackendInitFailed, "Backend not initialized"});
        }
        if (should_fail_generate) {
            return std::unexpected(Error{ErrorCode::InferenceFailed, error_message});
        }

        if (generation_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(generation_delay_ms));
        }

        last_prompt_tokens = prompt_tokens;

        std::string response;
        switch (mode) {
            case ResponseMode::Echo:
                response = "Echo: " + std::to_string(prompt_tokens.size()) + " tokens";
                break;
            case ResponseMode::Fixed:
                if (!response_queue.empty()) {
                    response = response_queue.front();
                    response_queue.pop();
                } else {
                    response = default_response;
                }
                break;
            case ResponseMode::TokenByToken:
                response = default_response;
                break;
        }

        for (const auto& stop : stop_sequences) {
            size_t pos = response.find(stop);
            if (pos != std::string::npos) {
                response = response.substr(0, pos);
                break;
            }
        }

        if (on_token.has_value()) {
            std::istringstream iss(response);
            std::string word;
            streamed_tokens.clear();
            while (iss >> word) {
                word += " ";
                streamed_tokens.push_back(word);
                (*on_token)(word);
                token_callback_count++;
            }
        }

        (void)max_tokens;
        return response;
    }

    int get_context_size() const override { return context_size; }

    void clear_kv_cache() override {
        ++clear_kv_cache_calls;
    }

    Expected<std::string> format_prompt(const std::vector<Message>& messages) override {
        std::ostringstream out;
        for (const auto& msg : messages) {
            out << role_to_string(msg.role) << ": " << msg.content << "\n";
        }
        last_formatted_prompt = out.str();
        return last_formatted_prompt;
    }

    void finalize_response(const std::vector<Message>&) override {}

    // Test helpers
    void enqueue_response(const std::string& response) {
        response_queue.push(response);
    }

    void reset() {
        initialized = false;
        token_callback_count = 0;
        streamed_tokens.clear();
        last_prompt_tokens.clear();
        clear_kv_cache_calls = 0;
        last_formatted_prompt.clear();
        while (!response_queue.empty()) response_queue.pop();
    }
};

} // namespace zoo::testing
