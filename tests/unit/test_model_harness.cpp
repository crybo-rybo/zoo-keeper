/**
 * @file test_model_harness.cpp
 * @brief Unit checks for the top-level Model harness API.
 */

#include "core/model_test_access.hpp"
#include "zoo/model.hpp"

#include <array>
#include <gtest/gtest.h>
#include <span>
#include <type_traits>

namespace {

static_assert(std::is_same_v<zoo::Model, zoo::core::Model>);

zoo::ModelConfig make_config() {
    zoo::ModelConfig config;
    config.model_path = "unused.gguf";
    return config;
}

} // namespace

TEST(ModelHarnessTest, InvalidExtractSchemaFailsBeforeHistoryMutation) {
    auto model = zoo::core::ModelTestAccess::make(make_config(), zoo::GenerationOptions{});
    const auto before = model->get_history();

    const nlohmann::json bad_schema = {{"type", "array"}};
    auto result = model->extract(bad_schema, "extract something");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidOutputSchema);
    EXPECT_EQ(model->get_history(), before);
}

TEST(ModelHarnessTest, CompleteRestoresHistoryWhenGenerationValidationFails) {
    auto model = zoo::core::ModelTestAccess::make(make_config(), zoo::GenerationOptions{});
    model->set_system_prompt("Retained prompt");
    ASSERT_TRUE(model->add_message(zoo::OwnedMessage::user("Retained question").view()));
    const auto before = model->get_history();
    const int before_tokens = model->estimated_tokens();

    const std::array<zoo::MessageView, 2> scoped_messages = {
        zoo::MessageView{zoo::Role::System, "Scoped prompt"},
        zoo::MessageView{zoo::Role::User, "Scoped question"},
    };
    zoo::GenerationOptions invalid;
    invalid.max_tokens = 0;

    auto result =
        model->complete(zoo::ConversationView{std::span<const zoo::MessageView>(scoped_messages)},
                        zoo::GenerationOverride::explicit_options(invalid));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
    EXPECT_EQ(model->get_history(), before);
    EXPECT_EQ(model->estimated_tokens(), before_tokens);
}

TEST(ModelHarnessTest, ToolSpecCarriesJsonSchemaForTemplateSetup) {
    zoo::ToolSpec spec;
    spec.name = "search";
    spec.description = "Search local documents";
    spec.parameters_schema = {{"type", "object"},
                              {"properties", {{"query", {{"type", "string"}}}}},
                              {"required", nlohmann::json::array({"query"})}};

    EXPECT_EQ(spec.name, "search");
    EXPECT_EQ(spec.parameters_schema["properties"]["query"]["type"], "string");
}
