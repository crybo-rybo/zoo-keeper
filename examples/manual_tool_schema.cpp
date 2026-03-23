/**
 * @file manual_tool_schema.cpp
 * @brief Registers a manual-schema tool through `zoo::Agent`.
 */

#include <zoo/zoo.hpp>

#include <algorithm>
#include <iostream>
#include <string>

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
    generation.max_tokens = 256;
    generation.record_tool_trace = true;
    generation.sampling.temperature = 0.0f;
    generation.sampling.top_p = 1.0f;
    generation.sampling.top_k = 1;

    auto agent_result = zoo::Agent::create(model_config, zoo::AgentConfig{}, generation);
    if (!agent_result) {
        std::cerr << agent_result.error().to_string() << '\n';
        return 1;
    }

    auto agent = std::move(*agent_result);

    nlohmann::json schema = {
        {"type", "object"},
        {"properties",
         {{"query", {{"type", "string"}, {"description", "Search query"}}},
          {"limit", {{"type", "integer"}, {"enum", nlohmann::json::array({5, 10, 20})}}},
          {"scope", {{"type", "string"}, {"enum", nlohmann::json::array({"docs", "issues"})}}}}},
        {"required", nlohmann::json::array({"query"})},
        {"additionalProperties", false}};

    auto register_result = agent->register_tool(
        "search_documents", "Search a tiny in-memory document index for matching snippets.", schema,
        [](const nlohmann::json& args) -> zoo::Expected<nlohmann::json> {
            const std::string query = args.at("query").get<std::string>();
            const int limit = args.value("limit", 5);
            const std::string scope = args.value("scope", "docs");

            nlohmann::json matches = nlohmann::json::array();
            const int count = std::min(limit, 3);
            for (int i = 0; i < count; ++i) {
                matches.push_back(scope + " match " + std::to_string(i + 1) + " for " + query);
            }

            return nlohmann::json{{"query", query},
                                  {"scope", scope},
                                  {"limit", limit},
                                  {"matches", std::move(matches)}};
        });

    if (!register_result) {
        std::cerr << register_result.error().to_string() << '\n';
        return 1;
    }

    agent->set_system_prompt("You are a retrieval assistant. Use tools when they are relevant.");

    auto handle = agent->chat(
        "Search the docs for llama.cpp. Use a limit of 5 results and keep the answer brief.");
    auto response = handle.await_result();
    if (!response) {
        std::cerr << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << response->text << "\n\n";
    if (response->tool_trace) {
        for (const auto& invocation : response->tool_trace->invocations) {
            std::cout << invocation.name << " [" << zoo::to_string(invocation.status) << "]\n";
            std::cout << "args: " << invocation.arguments_json << '\n';
            if (invocation.result_json) {
                std::cout << "result: " << *invocation.result_json << '\n';
            }
            if (invocation.error) {
                std::cout << "error: " << invocation.error->to_string() << '\n';
            }
            std::cout << '\n';
        }
    }

    return 0;
}
