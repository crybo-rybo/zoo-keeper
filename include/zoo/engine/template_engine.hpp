#pragma once

#include "../types.hpp"
#include <sstream>
#include <algorithm>

namespace zoo {
namespace engine {

/**
 * @brief Renders conversation history into model-specific prompt formats
 *
 * Supports:
 * - Llama3: <|begin_of_text|><|start_header_id|>role<|end_header_id|>content<|eot_id|>
 * - ChatML: <|im_start|>role\ncontent<|im_end|>
 * - Custom: User-provided template string with placeholders
 *
 * Template placeholders:
 * - {{role}}: Message role (system/user/assistant)
 * - {{content}}: Message content
 * - {{bos}}: Beginning of sequence
 * - {{eos}}: End of sequence
 */
class TemplateEngine {
public:
    /**
     * @brief Construct with template type
     *
     * @param tmpl Template type
     * @param custom_template Optional custom template string
     */
    explicit TemplateEngine(PromptTemplate tmpl, const std::optional<std::string>& custom_template = std::nullopt)
        : template_(tmpl)
        , custom_template_(custom_template)
    {}

    /**
     * @brief Render conversation history into a prompt
     *
     * @param messages Conversation messages to render
     * @return Expected<std::string> Formatted prompt or error
     */
    Expected<std::string> render(const std::vector<Message>& messages) const {
        if (messages.empty()) {
            return tl::unexpected(Error{ErrorCode::InvalidMessageSequence, "Cannot render empty message list"});
        }

        std::ostringstream prompt;

        switch (template_) {
            case PromptTemplate::Llama3:
                prompt << render_llama3(messages);
                break;

            case PromptTemplate::ChatML:
                prompt << render_chatml(messages);
                break;

            case PromptTemplate::Custom:
                if (!custom_template_.has_value()) {
                    return tl::unexpected(Error{ErrorCode::InvalidTemplate, "Custom template not provided"});
                }
                prompt << render_custom(messages, *custom_template_);
                break;
        }

        return prompt.str();
    }

    /**
     * @brief Get template type
     */
    PromptTemplate get_template() const {
        return template_;
    }

private:
    PromptTemplate template_;
    std::optional<std::string> custom_template_;

    // Llama3 format implementation
    std::string render_llama3(const std::vector<Message>& messages) const {
        std::ostringstream out;
        out << "<|begin_of_text|>";

        for (const auto& msg : messages) {
            out << "<|start_header_id|>" << role_to_string(msg.role) << "<|end_header_id|>\n\n"
                << msg.content << "<|eot_id|>";
        }

        // Add assistant header to prompt for response
        if (!messages.empty() && messages.back().role != Role::Assistant) {
            out << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }

        return out.str();
    }

    // ChatML format implementation
    std::string render_chatml(const std::vector<Message>& messages) const {
        std::ostringstream out;

        for (const auto& msg : messages) {
            out << "<|im_start|>" << role_to_string(msg.role) << "\n"
                << msg.content << "<|im_end|>\n";
        }

        // Add assistant start tag to prompt for response
        if (!messages.empty() && messages.back().role != Role::Assistant) {
            out << "<|im_start|>assistant\n";
        }

        return out.str();
    }

    // Custom template implementation (simple placeholder replacement)
    std::string render_custom(const std::vector<Message>& messages, const std::string& tmpl) const {
        std::ostringstream out;

        for (const auto& msg : messages) {
            std::string formatted = tmpl;

            // Replace placeholders
            replace_all(formatted, "{{role}}", role_to_string(msg.role));
            replace_all(formatted, "{{content}}", msg.content);

            out << formatted;
        }

        return out.str();
    }

    // Helper: Replace all occurrences
    static void replace_all(std::string& str, const std::string& from, const std::string& to) {
        if (from.empty()) return;

        size_t pos = 0;
        while ((pos = str.find(from, pos)) != std::string::npos) {
            str.replace(pos, from.length(), to);
            pos += to.length();
        }
    }
};

} // namespace engine
} // namespace zoo
