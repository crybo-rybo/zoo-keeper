/**
 * @file test_backend_contract.cpp
 * @brief Compile-time contract checks for AgentBackend and core::Model parity.
 */

#include "agent/backend.hpp"
#include "agent/backend_model.hpp"

#include <gtest/gtest.h>
#include <type_traits>
#include <zoo/core/model.hpp>

namespace {

using zoo::CancellationCallback;
using zoo::CoreToolInfo;
using zoo::Expected;
using zoo::GenerationOptions;
using zoo::HistorySnapshot;
using zoo::MessageView;
using zoo::TokenCallback;
using zoo::core::Model;
using zoo::internal::agent::AgentBackend;
using zoo::internal::agent::GenerationResult;
using zoo::internal::agent::ParsedToolResponse;

template<typename ModelLike>
concept ModelMirrorsAgentBackend = requires(ModelLike& model) {
    { model.add_message(std::declval<MessageView>()) } -> std::same_as<Expected<void>>;
    { model.generate_from_history(std::declval<const GenerationOptions&>(),
                                  std::declval<TokenCallback>(),
                                  std::declval<CancellationCallback>()) }
        -> std::same_as<Expected<GenerationResult>>;
    { model.finalize_response() } -> std::same_as<void>;
    { model.set_system_prompt(std::declval<std::string_view>()) } -> std::same_as<void>;
    { model.get_history() } -> std::same_as<HistorySnapshot>;
    { model.clear_history() } -> std::same_as<void>;
    { model.swap_history(std::declval<HistorySnapshot>()) } -> std::same_as<HistorySnapshot>;
    { model.trim_history(std::declval<size_t>()) } -> std::same_as<void>;
    { model.set_tool_calling(std::declval<const std::vector<CoreToolInfo>&>()) }
        -> std::same_as<bool>;
    { model.set_schema_grammar(std::declval<const std::string&>()) } -> std::same_as<bool>;
    { model.clear_tool_grammar() } -> std::same_as<void>;
    { model.parse_tool_response(std::declval<std::string_view>()) } -> std::same_as<ParsedToolResponse>;
    { model.tool_calling_format_name() } -> std::same_as<const char*>;
};

static_assert(ModelMirrorsAgentBackend<Model>);
static_assert(std::same_as<GenerationResult, Model::GenerationResult>);
static_assert(std::same_as<ParsedToolResponse, Model::ParsedResponse>);

TEST(BackendContractTest, MakeModelBackendReturnsAgentBackend) {
    static_assert(std::is_invocable_r_v<std::unique_ptr<AgentBackend>,
                                        decltype(zoo::internal::agent::make_model_backend),
                                        std::unique_ptr<Model>>);
    SUCCEED();
}

} // namespace
