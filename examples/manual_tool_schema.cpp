/**
 * @file manual_tool_schema.cpp
 * @brief Configures model-facing tool schemas and parses native tool calls.
 */

#include <zoo/zoo.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

zoo::ToolSpec to_tool_spec(const zoo::tools::ToolMetadata& metadata) {
    return zoo::ToolSpec{metadata.name, metadata.description, metadata.parameters_schema};
}

zoo::tools::ToolCall to_tool_call(const zoo::OwnedToolCall& call) {
    return zoo::tools::ToolCall{call.id, call.name,
                                nlohmann::json::parse(call.arguments_json, nullptr, false)};
}

} // namespace

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
    generation.sampling.temperature = 0.0f;
    generation.sampling.top_p = 1.0f;
    generation.sampling.top_k = 1;

    auto model_result = zoo::Model::load(model_config, generation);
    if (!model_result) {
        std::cerr << model_result.error().to_string() << '\n';
        return 1;
    }
    auto model = std::move(*model_result);

    nlohmann::json schema = {
        {"type", "object"},
        {"properties",
         {{"query", {{"type", "string"}, {"description", "Search query"}}},
          {"limit", {{"type", "integer"}, {"enum", nlohmann::json::array({5, 10, 20})}}},
          {"scope", {{"type", "string"}, {"enum", nlohmann::json::array({"docs", "issues"})}}}}},
        {"required", nlohmann::json::array({"query"})},
        {"additionalProperties", false}};

    zoo::tools::ToolRegistry registry;
    auto register_result = registry.register_tool(
        "search_documents", "Search a tiny in-memory document index for matching snippets.", schema,
        [](const nlohmann::json& args) -> zoo::Expected<nlohmann::json> {
            return nlohmann::json{{"query", args.at("query")},
                                  {"scope", args.value("scope", "docs")},
                                  {"limit", args.value("limit", 5)}};
        });
    if (!register_result) {
        std::cerr << register_result.error().to_string() << '\n';
        return 1;
    }

    std::vector<zoo::ToolSpec> specs;
    for (const auto& metadata : registry.get_all_tool_metadata()) {
        specs.push_back(to_tool_spec(metadata));
    }
    if (!model->set_tool_calling(specs)) {
        std::cerr << "Selected model/template does not support native tool calling.\n";
        return 1;
    }

    model->set_system_prompt("You are a retrieval assistant. Emit a tool call when useful.");
    auto prompt_result = model->add_message(
        zoo::OwnedMessage::user("Search the docs for llama.cpp. Use a limit of 5 results.").view());
    if (!prompt_result) {
        std::cerr << prompt_result.error().to_string() << '\n';
        return 1;
    }
    auto generated = model->generate_from_history();
    if (!generated) {
        std::cerr << generated.error().to_string() << '\n';
        return 1;
    }

    std::cout << "Visible content:\n" << generated->parsed_content << "\n\n";
    for (const auto& call : generated->tool_calls) {
        auto tool_call = to_tool_call(call);
        auto validation = zoo::tools::ToolArgumentsValidator{}.validate(tool_call, registry);
        std::cout << "Tool call: " << call.name << "\n";
        std::cout << "Arguments: " << call.arguments_json << "\n";
        std::cout << "Validation: " << (validation ? "ok" : validation.error().to_string())
                  << "\n\n";
    }

    return 0;
}
