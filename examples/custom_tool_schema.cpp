/**
 * @file custom_tool_schema.cpp
 * @brief Registers and invokes a manual-schema tool with nested arguments.
 */

#include <zoo/tools/registry.hpp>

#include <iostream>

int main() {
    zoo::tools::ToolRegistry registry;

    nlohmann::json schema = {
        {"type", "object"},
        {"properties",
         {{"query",
           {{"type", "object"},
            {"properties",
             {{"term", {{"type", "string"}}},
              {"limit", {{"type", "integer"}, {"minimum", 1}, {"maximum", 10}}}}},
            {"required", nlohmann::json::array({"term"})},
            {"additionalProperties", false}}}}},
        {"required", nlohmann::json::array({"query"})},
        {"additionalProperties", false}};

    zoo::tools::ToolHandler handler =
        [](const nlohmann::json& args) -> zoo::Expected<nlohmann::json> {
        const auto& query = args.at("query");
        const std::string term = query.at("term").get<std::string>();
        const int limit = query.value("limit", 3);

        return nlohmann::json{
            {"term", term},
            {"matches", nlohmann::json::array({"Match 1 for " + term, "Match 2 for " + term})},
            {"limit", limit}};
    };

    registry.register_tool("search_documents",
                           "Search a local document index using a structured query object.",
                           std::move(schema), std::move(handler));

    auto result =
        registry.invoke("search_documents", {{"query", {{"term", "llama.cpp"}, {"limit", 2}}}});

    if (!result) {
        std::cerr << result.error().to_string() << '\n';
        return 1;
    }

    std::cout << result->dump(2) << '\n';
    return 0;
}
