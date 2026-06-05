/**
 * @file model_harness.cpp
 * @brief Stateless completion and schema extraction for `zoo::core::Model`.
 */

#include "core/model_impl.hpp"
#include "tools/grammar.hpp"
#include "zoo/core/model.hpp"
#include "zoo/tools/registry.hpp"
#include "zoo/tools/validation.hpp"

#include <chrono>
#include <nlohmann/json.hpp>
#include <utility>

namespace zoo::core {

namespace {

template <typename Callback> class ScopeExit {
  public:
    explicit ScopeExit(Callback callback) : callback_(std::move(callback)) {}
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ~ScopeExit() {
        if (active_) {
            callback_();
        }
    }
    void dismiss() noexcept {
        active_ = false;
    }

  private:
    [[no_unique_address]] Callback callback_;
    bool active_ = true;
};

template <typename Callback> ScopeExit(Callback) -> ScopeExit<Callback>;

HistorySnapshot snapshot_from_view(ConversationView messages) {
    HistorySnapshot snapshot;
    snapshot.messages.reserve(messages.size());
    for (size_t index = 0; index < messages.size(); ++index) {
        snapshot.messages.push_back(OwnedMessage::from_view(messages[index]));
    }
    return snapshot;
}

std::string response_text_from_generation(const Model::GenerationResult& generated) {
    if (generated.tool_call_detected) {
        return generated.parsed_content;
    }
    return generated.text;
}

Expected<TextResponse> generate_response_from_history(Model& model, GenerationOverride generation,
                                                      TokenCallback on_token,
                                                      CancellationCallback should_cancel) {
    const auto start_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point first_token_time;
    bool first_token_received = false;
    int completion_tokens = 0;

    auto wrapped_callback = [&](std::string_view token) -> TokenAction {
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        if (on_token) {
            return on_token(token);
        }
        return TokenAction::Continue;
    };

    auto generated =
        model.generate_from_history(generation, TokenCallback(wrapped_callback), should_cancel);
    if (!generated) {
        return std::unexpected(generated.error());
    }

    const auto end_time = std::chrono::steady_clock::now();

    TextResponse response;
    response.text = response_text_from_generation(*generated);
    response.usage.prompt_tokens = generated->prompt_tokens;
    response.usage.completion_tokens = completion_tokens;
    response.usage.total_tokens = generated->prompt_tokens + completion_tokens;
    response.metrics.latency_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (first_token_received) {
        response.metrics.time_to_first_token_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        const auto generation_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - first_token_time);
        if (generation_time.count() > 0) {
            response.metrics.tokens_per_second =
                (completion_tokens * 1000.0) / generation_time.count();
        }
    }
    return response;
}

Expected<std::vector<tools::ToolParameter>> normalize_output_schema(const nlohmann::json& schema) {
    auto params = tools::detail::normalize_schema(schema);
    if (!params) {
        return std::unexpected(Error{ErrorCode::InvalidOutputSchema, params.error().message});
    }
    return params;
}

void restore_sampler_policy(Model::Impl& impl, Model::Impl::SamplerPolicy previous_policy) {
    impl.session_.sampler_policy = std::move(previous_policy);
    if (impl.session_.sampler_policy.mode == Model::Impl::SamplerPolicy::Mode::Plain) {
        impl.session_.sampler = create_sampler_chain(impl);
        return;
    }
    (void)ensure_grammar_sampler_for_pass(impl);
}

Expected<ExtractionResponse> extract_from_history(Model& model, Model::Impl& impl,
                                                  const std::vector<tools::ToolParameter>& params,
                                                  GenerationOverride generation,
                                                  TokenCallback on_token,
                                                  CancellationCallback should_cancel) {
    const auto previous_policy = impl.session_.sampler_policy;
    const std::string grammar = tools::GrammarBuilder::build_schema(params);
    if (!model.set_schema_grammar(grammar)) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed, "Failed to initialize schema grammar"});
    }
    auto restore_policy =
        ScopeExit([&impl, previous_policy] { restore_sampler_policy(impl, previous_policy); });

    auto response = generate_response_from_history(model, generation, on_token, should_cancel);
    if (!response) {
        return std::unexpected(response.error());
    }

    nlohmann::json extracted;
    try {
        extracted = nlohmann::json::parse(response->text);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed,
                  std::string("Failed to parse extraction output as JSON: ") + e.what()});
    }

    if (auto validation = tools::validate_json_against_schema(extracted, params); !validation) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed,
                  "Extracted JSON failed schema validation: " + validation.error().message});
    }

    ExtractionResponse extraction;
    extraction.text = std::move(response->text);
    extraction.data = std::move(extracted);
    extraction.usage = response->usage;
    extraction.metrics = response->metrics;
    return extraction;
}

} // namespace

Expected<TextResponse> Model::complete(ConversationView messages, GenerationOverride generation,
                                       TokenCallback on_token, CancellationCallback should_cancel) {
    auto previous = swap_history(snapshot_from_view(messages));
    auto restore_history = ScopeExit(
        [this, previous = std::move(previous)]() mutable { replace_history(std::move(previous)); });

    return generate_response_from_history(*this, generation, on_token, should_cancel);
}

Expected<ExtractionResponse> Model::extract(const nlohmann::json& output_schema,
                                            std::string_view user_message,
                                            GenerationOverride generation, TokenCallback on_token,
                                            CancellationCallback should_cancel) {
    return extract(output_schema, MessageView{Role::User, user_message}, generation, on_token,
                   should_cancel);
}

Expected<ExtractionResponse> Model::extract(const nlohmann::json& output_schema,
                                            MessageView message, GenerationOverride generation,
                                            TokenCallback on_token,
                                            CancellationCallback should_cancel) {
    auto params = normalize_output_schema(output_schema);
    if (!params) {
        return std::unexpected(params.error());
    }

    auto add_result = add_message(message);
    if (!add_result) {
        return std::unexpected(add_result.error());
    }
    auto rollback_user = ScopeExit([this] { rollback_last_message(*impl_); });

    auto response =
        extract_from_history(*this, *impl_, *params, generation, on_token, should_cancel);
    if (!response) {
        return std::unexpected(response.error());
    }

    auto assistant_result = add_message(OwnedMessage::assistant(response->text).view());
    if (!assistant_result) {
        return std::unexpected(assistant_result.error());
    }
    finalize_response();
    rollback_user.dismiss();
    return response;
}

Expected<ExtractionResponse> Model::extract(const nlohmann::json& output_schema,
                                            ConversationView messages,
                                            GenerationOverride generation, TokenCallback on_token,
                                            CancellationCallback should_cancel) {
    auto params = normalize_output_schema(output_schema);
    if (!params) {
        return std::unexpected(params.error());
    }

    auto previous = swap_history(snapshot_from_view(messages));
    auto restore_history = ScopeExit(
        [this, previous = std::move(previous)]() mutable { replace_history(std::move(previous)); });

    return extract_from_history(*this, *impl_, *params, generation, on_token, should_cancel);
}

} // namespace zoo::core
