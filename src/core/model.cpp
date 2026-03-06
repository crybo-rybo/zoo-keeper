#include "zoo/core/model.hpp"
#include <algorithm>
#include <chrono>

namespace zoo::core {

Model::Model(const Config& config, std::unique_ptr<IBackend> backend)
    : config_(config)
    , backend_(std::move(backend))
{
    if (config.system_prompt) {
        set_system_prompt(*config.system_prompt);
    }
}

Expected<std::unique_ptr<Model>> Model::load(
    const Config& config,
    std::unique_ptr<IBackend> backend
) {
    if (auto result = config.validate(); !result) {
        return std::unexpected(result.error());
    }

    if (!backend) {
        backend = create_llama_backend();
        if (!backend) {
            return std::unexpected(Error{
                ErrorCode::BackendInitFailed,
                "Failed to create backend"
            });
        }
    }

    if (auto result = backend->initialize(config); !result) {
        return std::unexpected(result.error());
    }

    return std::unique_ptr<Model>(new Model(config, std::move(backend)));
}

Expected<Response> Model::generate(
    const std::string& user_message,
    std::optional<std::function<void(std::string_view)>> on_token
) {
    auto start_time = std::chrono::steady_clock::now();

    auto add_result = add_message(Message::user(user_message));
    if (!add_result) {
        return std::unexpected(add_result.error());
    }

    std::chrono::steady_clock::time_point first_token_time;
    bool first_token_received = false;
    int completion_tokens = 0;

    auto wrapped_callback = [&](std::string_view token) {
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        if (on_token) {
            (*on_token)(token);
        }
    };

    auto prompt_result = backend_->format_prompt(messages_);
    if (!prompt_result) {
        messages_.pop_back();
        return std::unexpected(prompt_result.error());
    }

    auto tokens_result = backend_->tokenize(*prompt_result);
    if (!tokens_result) {
        messages_.pop_back();
        return std::unexpected(tokens_result.error());
    }

    int prompt_tokens = static_cast<int>(tokens_result->size());

    auto generate_result = backend_->generate(
        *tokens_result,
        config_.max_tokens,
        config_.stop_sequences,
        std::optional<std::function<void(std::string_view)>>(wrapped_callback)
    );

    if (!generate_result) {
        messages_.pop_back();
        return std::unexpected(generate_result.error());
    }

    std::string generated_text = std::move(*generate_result);
    messages_.push_back(Message::assistant(generated_text));
    estimated_tokens_ += estimate_tokens(generated_text) + kTemplateOverheadPerMessage;
    backend_->finalize_response(messages_);

    auto end_time = std::chrono::steady_clock::now();

    Response response;
    response.text = std::move(generated_text);
    response.usage.prompt_tokens = prompt_tokens;
    response.usage.completion_tokens = completion_tokens;
    response.usage.total_tokens = prompt_tokens + completion_tokens;

    response.metrics.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    if (first_token_received) {
        response.metrics.time_to_first_token_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                first_token_time - start_time);
        auto generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - first_token_time);
        if (generation_time.count() > 0) {
            response.metrics.tokens_per_second =
                (completion_tokens * 1000.0) / generation_time.count();
        }
    }

    return response;
}

Expected<std::string> Model::generate_from_history(
    std::optional<std::function<void(std::string_view)>> on_token
) {
    auto prompt_result = backend_->format_prompt(messages_);
    if (!prompt_result) {
        return std::unexpected(prompt_result.error());
    }

    auto tokens_result = backend_->tokenize(*prompt_result);
    if (!tokens_result) {
        return std::unexpected(tokens_result.error());
    }

    return backend_->generate(
        *tokens_result,
        config_.max_tokens,
        config_.stop_sequences,
        on_token
    );
}

void Model::set_system_prompt(const std::string& prompt) {
    Message sys_msg = Message::system(prompt);

    if (!messages_.empty() && messages_[0].role == Role::System) {
        estimated_tokens_ -= estimate_tokens(messages_[0].content) + kTemplateOverheadPerMessage;
        messages_[0] = sys_msg;
    } else {
        messages_.insert(messages_.begin(), sys_msg);
    }

    estimated_tokens_ += estimate_tokens(prompt) + kTemplateOverheadPerMessage;
}

Expected<void> Model::add_message(const Message& message) {
    auto err = validate_role_sequence(message.role);
    if (!err) {
        return std::unexpected(err.error());
    }

    messages_.push_back(message);
    estimated_tokens_ += estimate_tokens(message.content) + kTemplateOverheadPerMessage;
    return {};
}

std::vector<Message> Model::get_history() const {
    return messages_;
}

void Model::clear_history() {
    messages_.clear();
    estimated_tokens_ = 0;
    backend_->clear_kv_cache();
}

int Model::context_size() const {
    return config_.context_size;
}

int Model::estimated_tokens() const {
    return estimated_tokens_;
}

bool Model::is_context_exceeded() const {
    return estimated_tokens_ > config_.context_size;
}

int Model::estimate_tokens(const std::string& text) const {
    auto result = backend_->tokenize(text);
    if (result && !result->empty()) {
        return static_cast<int>(result->size());
    }
    return std::max(1, static_cast<int>(text.length() / 4));
}

Expected<void> Model::validate_role_sequence(Role role) const {
    if (messages_.empty()) {
        if (role == Role::Tool) {
            return std::unexpected(Error{
                ErrorCode::InvalidMessageSequence,
                "First message cannot be a tool response"
            });
        }
        return {};
    }

    if (role == Role::System) {
        return std::unexpected(Error{
            ErrorCode::InvalidMessageSequence,
            "System message only allowed at the beginning"
        });
    }

    const Role last_role = messages_.back().role;
    if (role == last_role && role != Role::Tool) {
        return std::unexpected(Error{
            ErrorCode::InvalidMessageSequence,
            "Cannot have consecutive messages with the same role (except Tool)"
        });
    }

    return {};
}

} // namespace zoo::core
