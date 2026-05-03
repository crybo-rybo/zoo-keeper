/**
 * @file registry.hpp
 * @brief Tool registration, schema normalization, and invocation helpers.
 */

#pragma once

#include "types.hpp"
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
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

[[nodiscard]] nlohmann::json build_parameters_schema(const std::vector<ToolParameter>& parameters);

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
[[nodiscard]] Expected<ToolMetadata> normalize_manual_tool_metadata(const std::string& name,
                                                                    const std::string& description,
                                                                    const nlohmann::json& schema);

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

Expected<ToolDefinition> make_tool_definition(const std::string& name,
                                              const std::string& description,
                                              const nlohmann::json& schema, ToolHandler handler);

} // namespace detail

/**
 * @brief Registry of tools available to the agent runtime.
 *
 * The registry owns normalized tool metadata, exposes deterministic JSON Schema
 * definitions for prompt construction and grammar generation, and invokes
 * registered handlers. Callers serialize registration before later reads.
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
                                 nlohmann::json schema, ToolHandler handler);

    /**
     * @brief Registers a normalized tool definition.
     *
     * Existing entries with the same name are replaced in place while
     * preserving registration order.
     */
    Expected<void> register_tool(ToolDefinition definition);

    /**
     * @brief Registers multiple tool definitions under a single lock acquisition.
     *
     * Existing entries with the same name are replaced in place while
     * preserving registration order.
     *
     * @param definitions Tool definitions to register.
     * @return Void on success.
     */
    Expected<void> register_tools(std::vector<ToolDefinition> definitions);

    /**
     * @brief Reports whether a tool name is currently registered.
     */
    bool has_tool(const std::string& name) const;

    /**
     * @brief Invokes a registered tool with JSON arguments.
     */
    Expected<nlohmann::json> invoke(const std::string& name, const nlohmann::json& args) const;

    /**
     * @brief Returns the OpenAI-style tool schema for one registered tool.
     */
    nlohmann::json get_tool_schema(const std::string& name) const;

    /**
     * @brief Returns the raw normalized JSON parameter schema for a registered tool.
     */
    std::optional<nlohmann::json> get_parameters_schema(const std::string& name) const;

    /**
     * @brief Returns normalized metadata for a registered tool.
     */
    std::optional<ToolMetadata> get_tool_metadata(const std::string& name) const;

    /**
     * @brief Returns normalized metadata for every registered tool in registration order.
     */
    std::vector<ToolMetadata> get_all_tool_metadata() const;

    /**
     * @brief Returns schemas for every registered tool in registration order.
     */
    nlohmann::json get_all_schemas() const;

    /**
     * @brief Returns the names of every registered tool in registration order.
     */
    std::vector<std::string> get_tool_names() const;

    /// Returns the number of registered tools.
    size_t size() const;

  private:
    /// Converts tool metadata into the schema shape consumed by prompts.
    static nlohmann::json build_schema_json(const ToolMetadata& metadata);

    std::vector<ToolDefinition> tools_;
    std::unordered_map<std::string, size_t> index_by_name_;
};

} // namespace zoo::tools
