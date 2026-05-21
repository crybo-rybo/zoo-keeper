/**
 * @file registry.cpp
 * @brief Out-of-line definitions for non-template registry and schema helpers.
 */

#include "zoo/tools/registry.hpp"

#include <unordered_set>

namespace zoo::tools {
namespace detail {

nlohmann::json build_parameters_schema(const std::vector<ToolParameter>& parameters) {
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json required = nlohmann::json::array();

    for (const auto& parameter : parameters) {
        nlohmann::json property = {{"type", tool_value_type_name(parameter.type)}};
        if (!parameter.description.empty()) {
            property["description"] = parameter.description;
        }
        if (!parameter.enum_values.empty()) {
            property["enum"] = parameter.enum_values;
        }

        properties[parameter.name] = std::move(property);
        if (parameter.required) {
            required.push_back(parameter.name);
        }
    }

    return nlohmann::json{{"type", "object"},
                          {"properties", std::move(properties)},
                          {"required", std::move(required)},
                          {"additionalProperties", false}};
}

Expected<ToolValueType> parse_tool_value_type(std::string_view value) {
    if (value == "integer") {
        return ToolValueType::Integer;
    }
    if (value == "number") {
        return ToolValueType::Number;
    }
    if (value == "string") {
        return ToolValueType::String;
    }
    if (value == "boolean") {
        return ToolValueType::Boolean;
    }

    return std::unexpected(Error{ErrorCode::InvalidToolSchema,
                                 "Unsupported tool parameter type: " + std::string(value)});
}

bool json_matches_type(const nlohmann::json& value, ToolValueType type) {
    switch (type) {
    case ToolValueType::Integer:
        return value.is_number_integer();
    case ToolValueType::Number:
        return value.is_number();
    case ToolValueType::String:
        return value.is_string();
    case ToolValueType::Boolean:
        return value.is_boolean();
    }
    return false;
}

Expected<void> validate_root_schema_keys(const nlohmann::json& schema,
                                         const std::string& tool_name) {
    static const std::unordered_set<std::string> kAllowedKeys = {
        "type", "properties", "required", "additionalProperties", "description"};

    for (const auto& [key, _] : schema.items()) {
        if (kAllowedKeys.contains(key)) {
            continue;
        }
        return std::unexpected(
            Error{ErrorCode::InvalidToolSchema,
                  "Unsupported tool schema keyword '" + key + "' for tool '" + tool_name + "'"});
    }

    return {};
}

Expected<void> validate_property_schema_keys(const nlohmann::json& property,
                                             const std::string& tool_name,
                                             const std::string& param_name) {
    static const std::unordered_set<std::string> kAllowedKeys = {"type", "description", "enum"};

    for (const auto& [key, _] : property.items()) {
        if (kAllowedKeys.contains(key)) {
            continue;
        }
        return std::unexpected(Error{ErrorCode::InvalidToolSchema,
                                     "Unsupported schema keyword '" + key + "' for parameter '" +
                                         param_name + "' on tool '" + tool_name + "'"});
    }

    return {};
}

Expected<void> validate_enum_values(const std::vector<nlohmann::json>& values, ToolValueType type,
                                    const std::string& tool_name, const std::string& param_name) {
    for (const auto& value : values) {
        if (!json_matches_type(value, type)) {
            return std::unexpected(Error{ErrorCode::InvalidToolSchema,
                                         "Enum value for parameter '" + param_name + "' on tool '" +
                                             tool_name + "' does not match declared type '" +
                                             tool_value_type_name(type) + "'"});
        }
    }

    return {};
}

Expected<std::vector<std::string>> collect_required_names(const nlohmann::json& schema,
                                                          const nlohmann::json& properties,
                                                          const std::string& tool_name) {
    std::vector<std::string> required_names;
    auto required_it = schema.find("required");
    if (required_it == schema.end()) {
        return required_names;
    }
    if (!required_it->is_array()) {
        return std::unexpected(
            Error{ErrorCode::InvalidToolSchema,
                  "Tool schema for '" + tool_name + "' must use an array for 'required'"});
    }

    std::unordered_set<std::string> seen;
    for (const auto& value : *required_it) {
        if (!value.is_string()) {
            return std::unexpected(
                Error{ErrorCode::InvalidToolSchema,
                      "Tool schema for '" + tool_name + "' must use string entries in 'required'"});
        }

        const std::string required_name = value.get<std::string>();
        if (!seen.insert(required_name).second) {
            return std::unexpected(
                Error{ErrorCode::InvalidToolSchema,
                      "Tool schema for '" + tool_name +
                          "' contains duplicate names in 'required': " + required_name});
        }

        if (!properties.contains(required_name)) {
            return std::unexpected(
                Error{ErrorCode::InvalidToolSchema,
                      "Tool schema for '" + tool_name +
                          "' lists a missing property in 'required': " + required_name});
        }

        required_names.push_back(required_name);
    }
    return required_names;
}

Expected<ToolMetadata> normalize_manual_tool_metadata(const std::string& name,
                                                      const std::string& description,
                                                      const nlohmann::json& schema) {
    if (!schema.is_object()) {
        return std::unexpected(
            Error{ErrorCode::InvalidToolSchema, "Tool schema must be a JSON object"});
    }

    if (auto result = validate_root_schema_keys(schema, name); !result) {
        return std::unexpected(result.error());
    }

    auto type_it = schema.find("type");
    if (type_it == schema.end() || !type_it->is_string() || *type_it != "object") {
        return std::unexpected(
            Error{ErrorCode::InvalidToolSchema,
                  "Tool schema for '" + name + "' must declare top-level type 'object'"});
    }

    auto props_it = schema.find("properties");
    if (props_it == schema.end() || !props_it->is_object()) {
        return std::unexpected(
            Error{ErrorCode::InvalidToolSchema,
                  "Tool schema for '" + name + "' must contain an object-valued 'properties'"});
    }

    auto additional_it = schema.find("additionalProperties");
    if (additional_it != schema.end() &&
        (!additional_it->is_boolean() || additional_it->get<bool>())) {
        return std::unexpected(Error{ErrorCode::InvalidToolSchema,
                                     "Tool schema for '" + name +
                                         "' must omit 'additionalProperties' or set it to false"});
    }

    auto required_names = collect_required_names(schema, *props_it, name);
    if (!required_names) {
        return std::unexpected(required_names.error());
    }
    const std::unordered_set<std::string> required_lookup(required_names->begin(),
                                                          required_names->end());

    std::unordered_map<std::string, ToolParameter> parameters_by_name;
    for (const auto& [param_name, property] : props_it->items()) {
        if (!property.is_object()) {
            return std::unexpected(
                Error{ErrorCode::InvalidToolSchema,
                      "Property '" + param_name + "' on tool '" + name + "' must be an object"});
        }

        if (auto result = validate_property_schema_keys(property, name, param_name); !result) {
            return std::unexpected(result.error());
        }

        auto property_type_it = property.find("type");
        if (property_type_it == property.end() || !property_type_it->is_string()) {
            return std::unexpected(
                Error{ErrorCode::InvalidToolSchema, "Property '" + param_name + "' on tool '" +
                                                        name + "' must declare a string 'type'"});
        }

        auto type = parse_tool_value_type(property_type_it->get_ref<const std::string&>());
        if (!type) {
            auto error = type.error();
            error.context = "parameter=" + param_name + ", tool=" + name;
            return std::unexpected(std::move(error));
        }

        ToolParameter parameter;
        parameter.name = param_name;
        parameter.type = *type;
        parameter.required = required_lookup.contains(param_name);

        if (auto description_it = property.find("description"); description_it != property.end()) {
            if (!description_it->is_string()) {
                return std::unexpected(Error{ErrorCode::InvalidToolSchema,
                                             "Property '" + param_name + "' on tool '" + name +
                                                 "' must use a string 'description'"});
            }
            parameter.description = description_it->get<std::string>();
        }

        if (auto enum_it = property.find("enum"); enum_it != property.end()) {
            if (!enum_it->is_array()) {
                return std::unexpected(Error{ErrorCode::InvalidToolSchema,
                                             "Property '" + param_name + "' on tool '" + name +
                                                 "' must use an array for 'enum'"});
            }
            parameter.enum_values = enum_it->get<std::vector<nlohmann::json>>();
            if (auto result =
                    validate_enum_values(parameter.enum_values, parameter.type, name, param_name);
                !result) {
                return std::unexpected(result.error());
            }
        }

        parameters_by_name.emplace(param_name, std::move(parameter));
    }

    // required params first, then optional in nlohmann::json property order (alphabetical by
    // default)
    std::vector<ToolParameter> parameters;
    parameters.reserve(parameters_by_name.size());

    for (const auto& required_name : *required_names) {
        auto node = parameters_by_name.extract(required_name);
        parameters.push_back(std::move(node.mapped()));
    }
    for (const auto& [param_name, _] : props_it->items()) {
        if (!required_lookup.contains(param_name)) {
            auto node = parameters_by_name.extract(param_name);
            parameters.push_back(std::move(node.mapped()));
        }
    }

    ToolMetadata metadata;
    metadata.name = name;
    metadata.description = description;
    metadata.parameters = std::move(parameters);
    metadata.parameters_schema = build_parameters_schema(metadata.parameters);
    return metadata;
}

Expected<std::vector<ToolParameter>> normalize_schema(const nlohmann::json& schema) {
    auto result = normalize_manual_tool_metadata("schema", "", schema);
    if (!result) {
        return std::unexpected(result.error());
    }
    return std::move(result->parameters);
}

Expected<ToolDefinition> make_tool_definition(const std::string& name,
                                              const std::string& description,
                                              const nlohmann::json& schema, ToolHandler handler) {
    auto metadata = normalize_manual_tool_metadata(name, description, schema);
    if (!metadata) {
        return std::unexpected(metadata.error());
    }
    return ToolDefinition{std::move(*metadata), std::move(handler)};
}

} // namespace detail

Expected<ToolDefinition> make_tool_definition(const std::string& name,
                                              const std::string& description,
                                              const nlohmann::json& schema, ToolHandler handler) {
    return detail::make_tool_definition(name, description, schema, std::move(handler));
}

nlohmann::json ToolRegistry::build_schema_json(const ToolMetadata& metadata) {
    return nlohmann::json{{"type", "function"},
                          {"function",
                           {{"name", metadata.name},
                            {"description", metadata.description},
                            {"parameters", metadata.parameters_schema}}}};
}

Expected<void> ToolRegistry::register_tool(const std::string& name, const std::string& description,
                                           const nlohmann::json& schema, ToolHandler handler) {
    auto definition = make_tool_definition(name, description, schema, std::move(handler));
    if (!definition) {
        return std::unexpected(definition.error());
    }
    return register_tool(std::move(*definition));
}

Expected<void> ToolRegistry::register_tool(ToolDefinition definition) {
    auto it = index_by_name_.find(definition.metadata.name);
    if (it != index_by_name_.end()) {
        tools_[it->second] = std::move(definition);
        return {};
    }

    const size_t index = tools_.size();
    index_by_name_.emplace(definition.metadata.name, index);
    tools_.push_back(std::move(definition));
    return {};
}

Expected<void> ToolRegistry::register_tools(std::vector<ToolDefinition> definitions) {
    for (auto& definition : definitions) {
        auto it = index_by_name_.find(definition.metadata.name);
        if (it != index_by_name_.end()) {
            tools_[it->second] = std::move(definition);
        } else {
            const size_t index = tools_.size();
            index_by_name_.emplace(definition.metadata.name, index);
            tools_.push_back(std::move(definition));
        }
    }
    return {};
}

bool ToolRegistry::has_tool(const std::string& name) const {
    return index_by_name_.contains(name);
}

Expected<nlohmann::json> ToolRegistry::invoke(const std::string& name,
                                              const nlohmann::json& args) const {
    auto it = index_by_name_.find(name);
    if (it == index_by_name_.end()) {
        return std::unexpected(Error{ErrorCode::ToolNotFound, "Tool not found: " + name});
    }
    return tools_[it->second].handler(args);
}

std::optional<ToolHandler> ToolRegistry::find_handler(const std::string& name) const {
    auto it = index_by_name_.find(name);
    if (it == index_by_name_.end()) {
        return std::nullopt;
    }
    return tools_[it->second].handler;
}

nlohmann::json ToolRegistry::get_tool_schema(const std::string& name) const {
    auto metadata = get_tool_metadata(name);
    if (!metadata) {
        return nlohmann::json{};
    }
    return build_schema_json(*metadata);
}

std::optional<nlohmann::json> ToolRegistry::get_parameters_schema(const std::string& name) const {
    auto metadata = get_tool_metadata(name);
    if (!metadata) {
        return std::nullopt;
    }
    return metadata->parameters_schema;
}

std::optional<ToolMetadata> ToolRegistry::get_tool_metadata(const std::string& name) const {
    auto it = index_by_name_.find(name);
    if (it == index_by_name_.end()) {
        return std::nullopt;
    }
    return tools_[it->second].metadata;
}

std::vector<ToolMetadata> ToolRegistry::get_all_tool_metadata() const {
    std::vector<ToolMetadata> metadata;
    metadata.reserve(tools_.size());
    for (const auto& tool : tools_) {
        metadata.push_back(tool.metadata);
    }
    return metadata;
}

nlohmann::json ToolRegistry::get_all_schemas() const {
    nlohmann::json schemas = nlohmann::json::array();
    for (const auto& tool : tools_) {
        schemas.push_back(build_schema_json(tool.metadata));
    }
    return schemas;
}

std::vector<std::string> ToolRegistry::get_tool_names() const {
    std::vector<std::string> names;
    names.reserve(tools_.size());
    for (const auto& tool : tools_) {
        names.push_back(tool.metadata.name);
    }
    return names;
}

size_t ToolRegistry::size() const {
    return tools_.size();
}

} // namespace zoo::tools
