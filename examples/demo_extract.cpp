/**
 * @file demo_extract.cpp
 * @brief Demonstrates Agent::extract() for grammar-constrained structured output.
 *
 * Runs three self-contained extraction scenarios against a single model to show
 * the stateful, stateless, and streaming extract() overloads.
 *
 * Usage:
 *   ./demo_extract <model.gguf>
 */

#include <zoo/zoo.hpp>

#include <iomanip>
#include <iostream>
#include <string>

// ============================================================================
// Helpers
// ============================================================================

static void print_separator(const std::string& title) {
    std::cout << "\n-- " << title << " " << std::string(55 - title.size(), '-') << "\n";
}

static void print_result(const zoo::Expected<zoo::Response>& response) {
    if (!response) {
        std::cerr << "Error: " << response.error().to_string() << "\n";
        return;
    }
    if (!response->extracted_data) {
        std::cerr << "Error: extracted_data is empty\n";
        return;
    }
    std::cout << "Result: " << response->extracted_data->dump(2) << "\n";
    std::cout << "Tokens: " << response->usage.completion_tokens << " completion, "
              << response->usage.prompt_tokens << " prompt\n";
}

// ============================================================================
// Scenarios
// ============================================================================

/**
 * @brief Scenario 1 — stateful extraction.
 *
 * Extracts a person's name and age from a natural-language sentence. The
 * message is appended to the agent's retained history.
 */
static void run_entity_extraction(zoo::Agent& agent) {
    print_separator("Scenario 1: entity extraction (stateful)");

    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
        {"required", nlohmann::json::array({"name", "age"})},
        {"additionalProperties", false}};

    std::cout << "Input:  \"Alice Chen is the lead engineer. She turned 34 last Tuesday.\"\n";
    auto handle = agent.extract(
        schema, zoo::Message::user("Alice Chen is the lead engineer. She turned 34 last Tuesday."));
    print_result(handle.future.get());
}

/**
 * @brief Scenario 2 — stateless extraction with an enum constraint.
 *
 * Classifies the sentiment of a movie review using an explicit message list.
 * The agent's history is not modified.
 */
static void run_sentiment_classification(zoo::Agent& agent) {
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

    auto handle = agent.extract(
        schema, {zoo::Message::system("Classify the overall sentiment of the review."),
                 zoo::Message::user(review)});
    print_result(handle.future.get());
}

/**
 * @brief Scenario 3 — streaming extraction.
 *
 * Extracts a numeric count while forwarding tokens to stdout via the on_token
 * callback so you can watch the constrained output build character by character.
 */
static void run_numeric_extraction(zoo::Agent& agent) {
    print_separator("Scenario 3: numeric extraction (streaming)");

    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"count", {{"type", "integer"}}}, {"unit", {{"type", "string"}}}}},
        {"required", nlohmann::json::array({"count", "unit"})},
        {"additionalProperties", false}};

    std::cout << "Input:  \"The delivery contains 48 individual cartons.\"\n";
    std::cout << "Stream: ";
    std::cout.flush();

    auto handle =
        agent.extract(schema, zoo::Message::user("The delivery contains 48 individual cartons."),
                      [](std::string_view token) { std::cout << token << std::flush; });

    auto response = handle.future.get();
    std::cout << "\n";
    print_result(response);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }

    zoo::Config config;
    config.model_path = argv[1];
    config.context_size = 4096;
    config.max_tokens = 64;
    config.n_gpu_layers = 0;
    config.sampling.temperature = 0.0f;
    config.sampling.top_p = 1.0f;
    config.sampling.top_k = 1;
    config.sampling.seed = 42;

    std::cout << "Loading model: " << config.model_path << "\n";
    auto agent_result = zoo::Agent::create(config);
    if (!agent_result) {
        std::cerr << "Error: " << agent_result.error().to_string() << "\n";
        return 1;
    }
    auto agent = std::move(*agent_result);
    agent->set_system_prompt("You are a precise extraction assistant. "
                             "Extract exactly the fields requested and nothing else.");

    run_entity_extraction(*agent);
    run_sentiment_classification(*agent);
    run_numeric_extraction(*agent);

    std::cout << "\nDone.\n";
    return 0;
}
