/**
 * @file registry.hpp
 * @brief Thread-safe tool registration, schema normalization, and invocation helpers.
 */

#pragma once

#include "types.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace zoo::tools {

namespace detail {

/**
 * @brief Maps supported C++ parameter types to normalized tool value types.
 */
template <typename T> struct tool_value_type;

template <> struct tool_value_type<int> {
    static constexpr ToolValueType value = ToolValueType::Integer;
};
template <> struct tool_value_type<float> {
    static constexpr ToolValueType value = ToolValueType::Number;
};
template <> struct tool_value_type<double> {
    static constexpr ToolValueType value = ToolValueType::Number;
};
template <> struct tool_value_type<bool> {
    static constexpr ToolValueType value = ToolValueType::Boolean;
};
template <> struct tool_value_type<std::string> {
    static constexpr ToolValueType value = ToolValueType::String;
};

template <typename T>
inline constexpr bool supported_tool_arg_v =
    std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, bool> || std::is_same_v<T, std::string>;

/**
 * @brief Extracts callable signature information for tool registration.
 */
template <typename T> struct function_traits;

template <typename R, typename... Args> struct function_traits<R (*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

template <typename C, typename R, typename... Args> struct function_traits<R (C::*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

template <typename T> struct function_traits : function_traits<decltype(&T::operator())> {};

template <typename R, typename... Args> struct function_traits<std::function<R(Args...)>> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

template <typename Handler, typename = void> struct is_json_handler_like : std::false_type {};

template <typename Handler>
struct is_json_handler_like<
    Handler, std::void_t<typename function_traits<std::remove_cvref_t<Handler>>::args_tuple,
                         typename function_traits<std::remove_cvref_t<Handler>>::return_type>>
    : std::bool_constant<[] {
        using traits = function_traits<std::remove_cvref_t<Handler>>;
        using args_tuple = typename traits::args_tuple;
        if constexpr (traits::arity != 1) {
            return false;
        } else {
            return std::is_same_v<std::tuple_element_t<0, args_tuple>, nlohmann::json> &&
                   std::is_same_v<typename traits::return_type, Expected<nlohmann::json>>;
        }
    }()> {};

template <typename Handler>
inline constexpr bool is_json_handler_like_v = is_json_handler_like<Handler>::value;

template <typename Tuple, size_t... Is>
consteval bool tuple_types_supported_impl(std::index_sequence<Is...>) {
    return (supported_tool_arg_v<std::tuple_element_t<Is, Tuple>> && ...);
}

template <typename Tuple> consteval bool tuple_types_supported() {
    return tuple_types_supported_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

/**
 * @brief Builds the normalized JSON Schema object for a parameter list.
 */
inline nlohmann::json build_parameters_schema(const std::vector<ToolParameter>& parameters) {
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

/**
 * @brief Extracts and converts one named JSON argument.
 */
template <typename T> T extract_arg(const nlohmann::json& args, const std::string& name) {
    return args.at(name).get<T>();
}

/**
 * @brief Invokes a callable by expanding JSON arguments in declaration order.
 */
template <typename Func, typename Tuple, size_t... Is>
auto invoke_with_json_impl(const Func& func, const nlohmann::json& args,
                           const std::vector<std::string>& param_names,
                           std::index_sequence<Is...>) {
    return func(extract_arg<std::tuple_element_t<Is, Tuple>>(args, param_names[Is])...);
}

/**
 * @brief Invokes a callable by expanding JSON arguments in declaration order.
 */
template <typename Func, typename Tuple>
auto invoke_with_json(const Func& func, const nlohmann::json& args,
                      const std::vector<std::string>& param_names) {
    constexpr size_t N = std::tuple_size_v<Tuple>;
    return invoke_with_json_impl<Func, Tuple>(func, args, param_names,
                                              std::make_index_sequence<N>{});
}

/**
 * @brief Wraps a tool result in the standardized `{"result": ...}` envelope.
 */
template <typename T> nlohmann::json wrap_result(T&& value) {
    return nlohmann::json{{"result", std::forward<T>(value)}};
}

[[nodiscard]] inline Expected<ToolValueType> parse_tool_value_type(std::string_view value) {
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

[[nodiscard]] inline bool json_matches_type(const nlohmann::json& value, ToolValueType type) {
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

[[nodiscard]] inline Expected<void> validate_root_schema_keys(const nlohmann::json& schema,
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

[[nodiscard]] inline Expected<void> validate_property_schema_keys(const nlohmann::json& property,
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

[[nodiscard]] inline Expected<void> validate_enum_values(const std::vector<nlohmann::json>& values,
                                                         ToolValueType type,
                                                         const std::string& tool_name,
                                                         const std::string& param_name) {
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

[[nodiscard]] inline Expected<ToolMetadata>
normalize_manual_tool_metadata(const std::string& name, const std::string& description,
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

    std::vector<std::string> required_names;
    std::unordered_set<std::string> required_lookup;
    if (auto required_it = schema.find("required"); required_it != schema.end()) {
        if (!required_it->is_array()) {
            return std::unexpected(
                Error{ErrorCode::InvalidToolSchema,
                      "Tool schema for '" + name + "' must use an array for 'required'"});
        }

        for (const auto& value : *required_it) {
            if (!value.is_string()) {
                return std::unexpected(
                    Error{ErrorCode::InvalidToolSchema,
                          "Tool schema for '" + name + "' must use string entries in 'required'"});
            }

            const std::string required_name = value.get<std::string>();
            if (!required_lookup.insert(required_name).second) {
                return std::unexpected(
                    Error{ErrorCode::InvalidToolSchema,
                          "Tool schema for '" + name +
                              "' contains duplicate names in 'required': " + required_name});
            }

            if (!props_it->contains(required_name)) {
                return std::unexpected(
                    Error{ErrorCode::InvalidToolSchema,
                          "Tool schema for '" + name +
                              "' lists a missing property in 'required': " + required_name});
            }

            required_names.push_back(required_name);
        }
    }

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

    // Build canonical parameter order: required params first (in the order
    // listed in the "required" array), then optional params (in nlohmann::json
    // object iteration order, which is alphabetical by default).
    std::vector<ToolParameter> parameters;
    parameters.reserve(parameters_by_name.size());

    for (const auto& required_name : required_names) {
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

template <typename Tuple, size_t... Is>
std::vector<ToolParameter> build_typed_parameters_impl(const std::vector<std::string>& param_names,
                                                       std::index_sequence<Is...>) {
    return {ToolParameter{
        param_names[Is], tool_value_type<std::tuple_element_t<Is, Tuple>>::value, true, {}, {}}...};
}

template <typename Tuple>
std::vector<ToolParameter> build_typed_parameters(const std::vector<std::string>& param_names) {
    constexpr size_t N = std::tuple_size_v<Tuple>;
    return build_typed_parameters_impl<Tuple>(param_names, std::make_index_sequence<N>{});
}

template <typename Func>
Expected<ToolDefinition>
make_tool_definition(const std::string& name, const std::string& description,
                     const std::vector<std::string>& param_names, Func func) {
    using traits = function_traits<Func>;
    using args_tuple = typename traits::args_tuple;

    static_assert(
        tuple_types_supported<args_tuple>(),
        "register_tool only supports int, float, double, bool, and std::string parameters");

    if (param_names.size() != traits::arity) {
        return std::unexpected(Error{ErrorCode::InvalidToolSignature,
                                     "Parameter name count (" + std::to_string(param_names.size()) +
                                         ") does not match function arity (" +
                                         std::to_string(traits::arity) + ")"});
    }

    ToolMetadata metadata;
    metadata.name = name;
    metadata.description = description;
    metadata.parameters = build_typed_parameters<args_tuple>(param_names);
    metadata.parameters_schema = build_parameters_schema(metadata.parameters);

    auto captured_names = param_names;
    ToolHandler handler = [f = std::move(func), names = std::move(captured_names)](
                              const nlohmann::json& args) -> Expected<nlohmann::json> {
        try {
            if constexpr (traits::arity == 0) {
                auto result = f();
                return wrap_result(std::move(result));
            } else {
                auto result = invoke_with_json<decltype(f), args_tuple>(f, args, names);
                return wrap_result(std::move(result));
            }
        } catch (const nlohmann::json::exception& e) {
            return std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                         std::string("JSON argument error: ") + e.what()});
        } catch (const std::exception& e) {
            return std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                         std::string("Tool execution failed: ") + e.what()});
        }
    };

    return ToolDefinition{std::move(metadata), std::move(handler)};
}

inline Expected<ToolDefinition> make_tool_definition(const std::string& name,
                                                     const std::string& description,
                                                     const nlohmann::json& schema,
                                                     ToolHandler handler) {
    auto metadata = normalize_manual_tool_metadata(name, description, schema);
    if (!metadata) {
        return std::unexpected(metadata.error());
    }
    return ToolDefinition{std::move(*metadata), std::move(handler)};
}

} // namespace detail

/**
 * @brief Thread-safe registry of tools available to the agent runtime.
 *
 * The registry owns normalized tool metadata, exposes deterministic JSON Schema
 * definitions for prompt construction and grammar generation, and provides
 * synchronized invocation of registered handlers.
 */
class ToolRegistry {
  public:
    /**
     * @brief Registers a strongly typed callable as a tool.
     */
    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 std::initializer_list<std::string> param_names, Func func) {
        return register_tool(name, description, std::vector<std::string>(param_names),
                             std::move(func));
    }

    /**
     * @brief Registers a strongly typed callable as a tool.
     */
    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 std::span<const std::string> param_names, Func func) {
        auto definition = detail::make_tool_definition(
            name, description, std::vector<std::string>(param_names.begin(), param_names.end()),
            std::move(func));
        if (!definition) {
            return std::unexpected(definition.error());
        }
        return register_tool(std::move(*definition));
    }

    /**
     * @brief Registers a tool using a prebuilt JSON Schema and a JSON-backed callable.
     */
    template <typename Handler>
        requires detail::is_json_handler_like_v<Handler>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 const nlohmann::json& schema, Handler handler) {
        return register_tool(name, description, nlohmann::json(schema),
                             ToolHandler(std::move(handler)));
    }

    /**
     * @brief Registers a tool using a prebuilt JSON Schema and handler.
     */
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 nlohmann::json schema, ToolHandler handler) {
        auto definition =
            detail::make_tool_definition(name, description, schema, std::move(handler));
        if (!definition) {
            return std::unexpected(definition.error());
        }
        return register_tool(std::move(*definition));
    }

    /**
     * @brief Registers a normalized tool definition.
     *
     * Existing entries with the same name are replaced in place while
     * preserving registration order.
     */
    Expected<void> register_tool(ToolDefinition definition) {
        std::unique_lock lock(mutex_);
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

    /**
     * @brief Reports whether a tool name is currently registered.
     */
    bool has_tool(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return index_by_name_.find(name) != index_by_name_.end();
    }

    /**
     * @brief Invokes a registered tool with JSON arguments.
     */
    Expected<nlohmann::json> invoke(const std::string& name, const nlohmann::json& args) const {
        std::shared_lock lock(mutex_);
        auto it = index_by_name_.find(name);
        if (it == index_by_name_.end()) {
            return std::unexpected(Error{ErrorCode::ToolNotFound, "Tool not found: " + name});
        }
        return tools_[it->second].handler(args);
    }

    /**
     * @brief Returns the OpenAI-style tool schema for one registered tool.
     */
    nlohmann::json get_tool_schema(const std::string& name) const {
        auto metadata = get_tool_metadata(name);
        if (!metadata) {
            return nlohmann::json{};
        }
        return build_schema_json(*metadata);
    }

    /**
     * @brief Returns the raw normalized JSON parameter schema for a registered tool.
     */
    std::optional<nlohmann::json> get_parameters_schema(const std::string& name) const {
        auto metadata = get_tool_metadata(name);
        if (!metadata) {
            return std::nullopt;
        }
        return metadata->parameters_schema;
    }

    /**
     * @brief Returns normalized metadata for a registered tool.
     */
    std::optional<ToolMetadata> get_tool_metadata(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = index_by_name_.find(name);
        if (it == index_by_name_.end()) {
            return std::nullopt;
        }
        return tools_[it->second].metadata;
    }

    /**
     * @brief Returns normalized metadata for every registered tool in registration order.
     */
    std::vector<ToolMetadata> get_all_tool_metadata() const {
        std::shared_lock lock(mutex_);
        std::vector<ToolMetadata> metadata;
        metadata.reserve(tools_.size());
        for (const auto& tool : tools_) {
            metadata.push_back(tool.metadata);
        }
        return metadata;
    }

    /**
     * @brief Returns schemas for every registered tool in registration order.
     */
    nlohmann::json get_all_schemas() const {
        std::shared_lock lock(mutex_);
        nlohmann::json schemas = nlohmann::json::array();
        for (const auto& tool : tools_) {
            schemas.push_back(build_schema_json(tool.metadata));
        }
        return schemas;
    }

    /**
     * @brief Returns the names of every registered tool in registration order.
     */
    std::vector<std::string> get_tool_names() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> names;
        names.reserve(tools_.size());
        for (const auto& tool : tools_) {
            names.push_back(tool.metadata.name);
        }
        return names;
    }

    /// Returns the number of registered tools.
    size_t size() const {
        std::shared_lock lock(mutex_);
        return tools_.size();
    }

  private:
    /// Converts tool metadata into the schema shape consumed by prompts.
    static nlohmann::json build_schema_json(const ToolMetadata& metadata) {
        return nlohmann::json{{"type", "function"},
                              {"function",
                               {{"name", metadata.name},
                                {"description", metadata.description},
                                {"parameters", metadata.parameters_schema}}}};
    }

    std::vector<ToolDefinition> tools_;
    std::unordered_map<std::string, size_t> index_by_name_;
    mutable std::shared_mutex mutex_;
};

} // namespace zoo::tools
