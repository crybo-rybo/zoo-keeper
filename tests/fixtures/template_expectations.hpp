#pragma once

#include <string>

namespace zoo {
namespace testing {
namespace fixtures {

/**
 * @brief Expected template outputs for testing
 *
 * Provides golden outputs for various template formats to ensure
 * template rendering is correct and consistent.
 */
class TemplateExpectations {
public:
    // ========================================================================
    // Llama3 Format Expectations
    // ========================================================================

    /**
     * @brief Llama3 format for simple conversation
     *
     * Conversation:
     * - System: "You are a helpful assistant."
     * - User: "Hello!"
     * - Assistant: "Hi! How can I help you?"
     */
    static std::string llama3_simple_conversation() {
        return
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Hello!<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "Hi! How can I help you?<|eot_id|>";
    }

    /**
     * @brief Llama3 format for minimal conversation
     *
     * Conversation:
     * - User: "What is 2+2?"
     */
    static std::string llama3_minimal_conversation() {
        return
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What is 2+2?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    /**
     * @brief Llama3 format with tool call
     *
     * Conversation:
     * - User: "What's the weather?"
     * - Assistant: "Let me check."
     * - Tool: "72F, Sunny"
     */
    static std::string llama3_with_tool() {
        return
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What's the weather?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "Let me check.<|eot_id|>"
            "<|start_header_id|>tool<|end_header_id|>\n\n"
            "72F, Sunny<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    // ========================================================================
    // ChatML Format Expectations
    // ========================================================================

    /**
     * @brief ChatML format for simple conversation
     *
     * Conversation:
     * - System: "You are a helpful assistant."
     * - User: "Hello!"
     * - Assistant: "Hi! How can I help you?"
     */
    static std::string chatml_simple_conversation() {
        return
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Hello!<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Hi! How can I help you?<|im_end|>\n";
    }

    /**
     * @brief ChatML format for minimal conversation
     *
     * Conversation:
     * - User: "What is 2+2?"
     */
    static std::string chatml_minimal_conversation() {
        return
            "<|im_start|>user\n"
            "What is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n";
    }

    /**
     * @brief ChatML format with tool call
     *
     * Conversation:
     * - User: "What's the weather?"
     * - Assistant: "Let me check."
     * - Tool: "72F, Sunny"
     */
    static std::string chatml_with_tool() {
        return
            "<|im_start|>user\n"
            "What's the weather?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Let me check.<|im_end|>\n"
            "<|im_start|>tool\n"
            "72F, Sunny<|im_end|>\n"
            "<|im_start|>assistant\n";
    }

    /**
     * @brief ChatML format with system prompt only
     */
    static std::string chatml_system_only() {
        return
            "<|im_start|>system\n"
            "Be helpful and concise.<|im_end|>\n"
            "<|im_start|>assistant\n";
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    /**
     * @brief Empty content message (Llama3)
     */
    static std::string llama3_empty_content() {
        return
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    /**
     * @brief Empty content message (ChatML)
     */
    static std::string chatml_empty_content() {
        return
            "<|im_start|>user\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n";
    }

    /**
     * @brief Multi-turn conversation (Llama3)
     */
    static std::string llama3_multi_turn() {
        return
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a math tutor.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What is 2+2?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "2+2 equals 4.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What about 3+3?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "3+3 equals 6.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "And 4+4?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    /**
     * @brief Multi-turn conversation (ChatML)
     */
    static std::string chatml_multi_turn() {
        return
            "<|im_start|>system\n"
            "You are a math tutor.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "2+2 equals 4.<|im_end|>\n"
            "<|im_start|>user\n"
            "What about 3+3?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "3+3 equals 6.<|im_end|>\n"
            "<|im_start|>user\n"
            "And 4+4?<|im_end|>\n"
            "<|im_start|>assistant\n";
    }
};

} // namespace fixtures
} // namespace testing
} // namespace zoo
