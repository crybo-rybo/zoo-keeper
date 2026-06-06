/**
 * @file registry.hpp
 * @brief Tool schema registration and normalization helpers.
 */

#pragma once

#include "types.hpp"
#include <cstddef>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zoo::tools {

namespace detail {

/**
 * @brief Builds the normalized JSON Schema object for a parameter list.
 */
[[nodiscard]] nlohmann::json build_parameters_schema(const std::vector<ToolParameter>& parameters);

[[nodiscard]] Expected<ToolValueType> parse_tool_value_type(std::string_view value);

[[nodiscard]] bool json_matches_type(const nlohmann::json& value, ToolValueType type);

[[nodiscard]] Expected<void> validate_root_schema_keys(const nlohmann::json& schema,
                                                       const std::string& tool_name);

[[nodiscard]] Expected<void> validate_property_schema_keys(const nlohmann::json& property,
                                                           const std::string& tool_name,
                                                           const std::string& param_name);

[[nodiscard]] Expected<void> validate_enum_values(const std::vector<nlohmann::json>& values,
                                                  ToolValueType type, const std::string& tool_name,
                                                  const std::string& param_name);

/**
 * @brief Normalizes one tool's parameter schema into canonical parameter metadata.
 */
[[nodiscard]] Expected<std::vector<ToolParameter>>
normalize_tool_parameters(const std::string& tool_name, const nlohmann::json& schema);

/**
 * @brief Normalizes a JSON Schema object into a parameter vector.
 *
 * Same validation logic as manual tool registration, but without requiring
 * a tool name or description context. Uses "schema" as the context name
 * for error messages.
 *
 * @param schema JSON Schema object with type "object".
 * @return Normalized parameter vector in canonical order (required first, then optional).
 */
[[nodiscard]] Expected<std::vector<ToolParameter>> normalize_schema(const nlohmann::json& schema);

} // namespace detail

/**
 * @brief Registry of model-facing tool schemas.
 *
 * The registry owns normalized tool specs, exposes deterministic JSON Schema
 * definitions for prompt construction, and keeps the normalized parameter
 * metadata used by validators. Tool execution is not stored here; callers map
 * validated tool-call names to application executors in their own code.
 */
class ToolRegistry {
  public:
    /**
     * @brief Registers a tool schema from name, description, and JSON Schema.
     */
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 const nlohmann::json& schema);

    /**
     * @brief Registers a model-facing tool spec.
     *
     * Existing entries with the same name are replaced in place while
     * preserving registration order.
     */
    Expected<void> register_tool(ToolSpec spec);

    /**
     * @brief Registers multiple tool specs as one ordered batch.
     *
     * Existing entries with the same name are replaced in place while
     * preserving registration order.
     */
    Expected<void> register_tools(std::vector<ToolSpec> specs);

    /**
     * @brief Reports whether a tool name is currently registered.
     */
    [[nodiscard]] bool has_tool(const std::string& name) const;

    /**
     * @brief Returns the OpenAI-style tool schema for one registered tool.
     */
    [[nodiscard]] nlohmann::json get_tool_schema(const std::string& name) const;

    /**
     * @brief Returns the raw normalized JSON parameter schema for a registered tool.
     */
    [[nodiscard]] std::optional<nlohmann::json>
    get_parameters_schema(const std::string& name) const;

    /**
     * @brief Returns the model-facing tool spec for a registered tool.
     */
    [[nodiscard]] std::optional<ToolSpec> get_tool_spec(const std::string& name) const;

    /**
     * @brief Returns normalized parameters for validation.
     */
    [[nodiscard]] std::optional<std::vector<ToolParameter>>
    get_tool_parameters(const std::string& name) const;

    /**
     * @brief Returns every registered tool spec in registration order.
     */
    [[nodiscard]] std::vector<ToolSpec> get_all_tool_specs() const;

    /**
     * @brief Returns schemas for every registered tool in registration order.
     */
    [[nodiscard]] nlohmann::json get_all_schemas() const;

    /**
     * @brief Returns the names of every registered tool in registration order.
     */
    [[nodiscard]] std::vector<std::string> get_tool_names() const;

    /// Returns the number of registered tools.
    [[nodiscard]] size_t size() const;

  private:
    struct RegisteredTool {
        ToolSpec spec;
        std::vector<ToolParameter> parameters;
    };

    [[nodiscard]] static Expected<RegisteredTool> normalize(ToolSpec spec);

    /// Converts tool metadata into the schema shape consumed by prompts.
    [[nodiscard]] static nlohmann::json build_schema_json(const ToolSpec& spec);

    std::vector<RegisteredTool> tools_;
    std::unordered_map<std::string, size_t> index_by_name_;
};

} // namespace zoo::tools
