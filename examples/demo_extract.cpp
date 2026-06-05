/**
 * @file demo_extract.cpp
 * @brief Demonstrates Model::extract() for grammar-constrained structured output.
 */

#include <zoo/zoo.hpp>

#include <array>
#include <iostream>
#include <string>

static void print_separator(const std::string& title) {
    std::cout << "\n-- " << title << " " << std::string(55 - title.size(), '-') << "\n";
}

static void print_result(const zoo::Expected<zoo::ExtractionResponse>& response) {
    if (!response) {
        std::cerr << "Error: " << response.error().to_string() << "\n";
        return;
    }
    std::cout << "Result: " << response->data.dump(2) << "\n";
    std::cout << "Tokens: " << response->usage.completion_tokens << " completion, "
              << response->usage.prompt_tokens << " prompt\n";
}

static void run_entity_extraction(zoo::Model& model) {
    print_separator("Scenario 1: entity extraction (stateful)");

    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
        {"required", nlohmann::json::array({"name", "age"})},
        {"additionalProperties", false}};

    std::cout << "Input:  \"Alice Chen is the lead engineer. She turned 34 last Tuesday.\"\n";
    print_result(
        model.extract(schema, "Alice Chen is the lead engineer. She turned 34 last Tuesday."));
}

static void run_sentiment_classification(zoo::Model& model) {
    print_separator("Scenario 2: sentiment classification (stateless, enum)");

    nlohmann::json schema = {
        {"type", "object"},
        {"properties",
         {{"sentiment",
           {{"type", "string"},
            {"enum", nlohmann::json::array({"positive", "negative", "neutral"})}}}}},
        {"required", nlohmann::json::array({"sentiment"})},
        {"additionalProperties", false}};

    const std::string review =
        "The cinematography was stunning but the pacing dragged in the second act.";
    std::cout << "Input:  \"" << review << "\"\n";

    const std::array<zoo::MessageView, 2> messages = {
        zoo::MessageView{zoo::Role::System, "Classify the overall sentiment of the review."},
        zoo::MessageView{zoo::Role::User, review},
    };
    print_result(
        model.extract(schema, zoo::ConversationView{std::span<const zoo::MessageView>(messages)}));
}

static void run_numeric_extraction(zoo::Model& model) {
    print_separator("Scenario 3: numeric extraction (streaming)");

    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"count", {{"type", "integer"}}}, {"unit", {{"type", "string"}}}}},
        {"required", nlohmann::json::array({"count", "unit"})},
        {"additionalProperties", false}};

    std::cout << "Input:  \"The delivery contains 48 individual cartons.\"\n";
    std::cout << "Stream: ";
    std::cout.flush();

    auto on_token = [](std::string_view token) {
        std::cout << token << std::flush;
        return zoo::TokenAction::Continue;
    };
    auto response =
        model.extract(schema, "The delivery contains 48 individual cartons.", {}, on_token);

    std::cout << "\n";
    print_result(response);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }

    zoo::ModelConfig model_config;
    model_config.model_path = argv[1];
    model_config.context_size = 4096;
    model_config.n_gpu_layers = 0;

    zoo::GenerationOptions generation;
    generation.max_tokens = 64;
    generation.sampling.temperature = 0.0f;
    generation.sampling.top_p = 1.0f;
    generation.sampling.top_k = 1;
    generation.sampling.seed = 42;

    std::cout << "Loading model: " << model_config.model_path << "\n";
    auto model_result = zoo::Model::load(model_config, generation);
    if (!model_result) {
        std::cerr << "Error: " << model_result.error().to_string() << "\n";
        return 1;
    }
    auto model = std::move(*model_result);
    model->set_system_prompt("You are a precise extraction assistant. "
                             "Extract exactly the fields requested and nothing else.");

    run_entity_extraction(*model);
    run_sentiment_classification(*model);
    run_numeric_extraction(*model);

    std::cout << "\nDone.\n";
    return 0;
}
