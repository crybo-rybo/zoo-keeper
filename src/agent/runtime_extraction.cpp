/**
 * @file runtime_extraction.cpp
 * @brief Structured output extraction: schema-constrained single-pass generation.
 */

#include "agent/runtime.hpp"

#include "agent/runtime_helpers.hpp"
#include "log.hpp"
#include "tools/grammar.hpp"
#include "zoo/core/model.hpp"
#include "zoo/tools/registry.hpp"
#include "zoo/tools/validation.hpp"
#include <chrono>

namespace zoo::internal::agent {

Expected<ExtractionResponse>
AgentRuntime::process_extraction_request(const ActiveRequest& request) {
    auto start_time = std::chrono::steady_clock::now();

    // Normalize schema and build GBNF grammar
    auto params = tools::detail::normalize_schema(request.extraction_schema->value());
    if (!params) {
        return std::unexpected(Error{ErrorCode::InvalidOutputSchema, params.error().message});
    }

    std::string grammar_str = tools::GrammarBuilder::build_schema(*params);

    auto history_scope =
        RequestHistoryScope::enter(*backend_, request.history_mode, *request.messages,
                                   agent_config_.max_history_messages, "extraction");
    if (!history_scope) {
        return std::unexpected(history_scope.error());
    }

    // Set schema grammar, restore previous tool calling state on exit
    const bool had_tool_calling = tool_grammar_active_.load(std::memory_order_acquire);

    if (!backend_->set_schema_grammar(grammar_str)) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed, "Failed to initialize schema grammar"});
    }

    ScopeExit grammar_guard([this, had_tool_calling] {
        if (had_tool_calling) {
            refresh_tool_calling_state();
        } else {
            backend_->clear_tool_grammar();
            tool_grammar_active_.store(false, std::memory_order_release);
        }
    });

    GenerationStats stats(start_time);
    GenerationRunner generation_runner(*backend_, callback_dispatcher_);
    auto cancellation_check = [&request]() {
        return request.cancelled && request.cancelled->load(std::memory_order_acquire);
    };
    auto pass = generation_runner.run(*request.options, request.streaming_callback,
                                      CancellationCallback(cancellation_check), stats);
    if (!pass) {
        return std::unexpected(pass.error());
    }
    auto generated = std::move(pass->generation);

    // Parse and validate extracted JSON
    nlohmann::json extracted;
    try {
        extracted = nlohmann::json::parse(generated.text);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed,
                  std::string("Failed to parse extraction output as JSON: ") + e.what()});
    }

    if (auto validation = tools::validate_json_against_schema(extracted, *params); !validation) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed,
                  "Extracted JSON failed schema validation: " + validation.error().message});
    }

    // Commit the assistant response to history
    backend_->add_message(Message::assistant(generated.text).view());
    backend_->finalize_response();

    auto end_time = std::chrono::steady_clock::now();

    ExtractionResponse response;
    response.text = std::move(generated.text);
    response.data = std::move(extracted);
    response.usage = stats.usage();
    response.metrics = stats.metrics(end_time);

    return response;
}

} // namespace zoo::internal::agent
