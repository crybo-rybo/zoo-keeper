/**
 * @file registry.hpp
 * @brief Thread-safe tool registration, schema generation, and invocation helpers.
 */

#pragma once

#include "types.hpp"
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace zoo::tools {

namespace detail {

/**
 * @brief Maps supported C++ parameter types to JSON Schema primitive names.
 */
template <typename T> struct json_type_name;

template <> struct json_type_name<int> {
    static constexpr const char* type = "integer";
};
template <> struct json_type_name<float> {
    static constexpr const char* type = "number";
};
template <> struct json_type_name<double> {
    static constexpr const char* type = "number";
};
template <> struct json_type_name<bool> {
    static constexpr const char* type = "boolean";
};
template <> struct json_type_name<std::string> {
    static constexpr const char* type = "string";
};

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

/**
 * @brief Builds a JSON Schema object from a tuple of C++ argument types.
 *
 * @param param_names Parameter names in declaration order.
 * @return JSON Schema object containing `properties` and `required`.
 */
template <typename Tuple, size_t... Is>
nlohmann::json build_properties_impl(const std::vector<std::string>& param_names,
                                     std::index_sequence<Is...>) {
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json required = nlohmann::json::array();
    ((properties[param_names[Is]] =
          nlohmann::json{{"type", json_type_name<std::tuple_element_t<Is, Tuple>>::type}},
      required.push_back(param_names[Is])),
     ...);
    return nlohmann::json{{"type", "object"}, {"properties", properties}, {"required", required}};
}

/**
 * @brief Builds a JSON Schema object from a tuple of C++ argument types.
 *
 * @param param_names Parameter names in declaration order.
 * @return JSON Schema object describing the callable's argument object.
 */
template <typename Tuple>
nlohmann::json build_properties(const std::vector<std::string>& param_names) {
    constexpr size_t N = std::tuple_size_v<Tuple>;
    return build_properties_impl<Tuple>(param_names, std::make_index_sequence<N>{});
}

/**
 * @brief Extracts and converts one named JSON argument.
 *
 * @param args JSON argument object.
 * @param name Name of the field to extract.
 * @return Converted C++ value of type `T`.
 */
template <typename T> T extract_arg(const nlohmann::json& args, const std::string& name) {
    return args.at(name).get<T>();
}

/**
 * @brief Invokes a callable by expanding JSON arguments in declaration order.
 *
 * @param func Callable to invoke.
 * @param args JSON argument object.
 * @param param_names Parameter names in declaration order.
 * @return Result returned by `func`.
 */
template <typename Func, typename Tuple, size_t... Is>
auto invoke_with_json_impl(const Func& func, const nlohmann::json& args,
                           const std::vector<std::string>& param_names,
                           std::index_sequence<Is...>) {
    return func(extract_arg<std::tuple_element_t<Is, Tuple>>(args, param_names[Is])...);
}

/**
 * @brief Invokes a callable by expanding JSON arguments in declaration order.
 *
 * @param func Callable to invoke.
 * @param args JSON argument object.
 * @param param_names Parameter names in declaration order.
 * @return Result returned by `func`.
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
 *
 * @param value Tool return value to serialize.
 * @return JSON object containing the serialized result.
 */
template <typename T> nlohmann::json wrap_result(T&& value) {
    return nlohmann::json{{"result", std::forward<T>(value)}};
}

} // namespace detail

/**
 * @brief Thread-safe registry of tools available to the agent runtime.
 *
 * The registry owns tool metadata, exposes JSON Schema definitions for prompt
 * construction and grammar generation, and provides synchronized invocation of
 * registered handlers.
 */
class ToolRegistry {
  public:
    /**
     * @brief Registers a strongly typed callable as a tool.
     *
     * The callable signature is introspected to build a JSON Schema parameter
     * object and a JSON-backed invocation wrapper.
     *
     * @tparam Func Callable type to register.
     * @param name Public tool name presented to the model.
     * @param description Human-readable tool description.
     * @param param_names Parameter names in callable argument order.
     * @param func Callable implementation to execute.
     * @return Empty success when the tool is registered, or an
     *         `InvalidToolSignature` error when `param_names` does not match the
     *         callable arity.
     */
    template <typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                                 const std::vector<std::string>& param_names, Func func) {
        using traits = detail::function_traits<Func>;
        using args_tuple = typename traits::args_tuple;

        if (param_names.size() != traits::arity) {
            return std::unexpected(Error{
                ErrorCode::InvalidToolSignature,
                "Parameter name count (" + std::to_string(param_names.size()) +
                    ") does not match function arity (" + std::to_string(traits::arity) + ")"});
        }

        nlohmann::json schema;
        if constexpr (traits::arity == 0) {
            schema = nlohmann::json{{"type", "object"},
                                    {"properties", nlohmann::json::object()},
                                    {"required", nlohmann::json::array()}};
        } else {
            schema = detail::build_properties<args_tuple>(param_names);
        }

        auto captured_names = param_names;
        ToolHandler handler = [f = std::move(func), names = std::move(captured_names)](
                                  const nlohmann::json& args) -> Expected<nlohmann::json> {
            try {
                if constexpr (traits::arity == 0) {
                    auto result = f();
                    return detail::wrap_result(std::move(result));
                } else {
                    auto result = detail::invoke_with_json<decltype(f), args_tuple>(f, args, names);
                    return detail::wrap_result(std::move(result));
                }
            } catch (const nlohmann::json::exception& e) {
                return std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                             std::string("JSON argument error: ") + e.what()});
            } catch (const std::exception& e) {
                return std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                             std::string("Tool execution failed: ") + e.what()});
            }
        };

        register_tool(name, description, std::move(schema), std::move(handler));
        return {};
    }

    /**
     * @brief Registers a tool using a prebuilt JSON Schema and handler.
     *
     * Existing entries with the same name are overwritten atomically.
     *
     * @param name Public tool name presented to the model.
     * @param description Human-readable tool description.
     * @param schema JSON Schema describing accepted arguments.
     * @param handler JSON-backed callable implementation.
     */
    void register_tool(const std::string& name, const std::string& description,
                       nlohmann::json schema, ToolHandler handler) {
        std::unique_lock lock(mutex_);
        tools_.insert_or_assign(
            name, ToolEntry{name, description, std::move(schema), std::move(handler)});
    }

    /**
     * @brief Reports whether a tool name is currently registered.
     *
     * @param name Tool name to query.
     * @return `true` when a matching tool exists.
     */
    bool has_tool(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return tools_.find(name) != tools_.end();
    }

    /**
     * @brief Invokes a registered tool with JSON arguments.
     *
     * @param name Tool name to execute.
     * @param args JSON argument object passed to the handler.
     * @return Handler result payload, or `ToolNotFound` / `ToolExecutionFailed`.
     */
    Expected<nlohmann::json> invoke(const std::string& name, const nlohmann::json& args) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return std::unexpected(Error{ErrorCode::ToolNotFound, "Tool not found: " + name});
        }
        return it->second.handler(args);
    }

    /**
     * @brief Returns the OpenAI-style tool schema for one registered tool.
     *
     * @param name Tool name to serialize.
     * @return Tool schema object, or an empty JSON value when the tool is missing.
     */
    nlohmann::json get_tool_schema(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return nlohmann::json{};
        }
        return build_schema_json(it->second);
    }

    /**
     * @brief Returns the raw JSON parameter schema for a registered tool.
     *
     * @param name Tool name to query.
     * @return Parameter schema when present, otherwise `std::nullopt`.
     */
    std::optional<nlohmann::json> get_parameters_schema(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return std::nullopt;
        }
        return it->second.parameters_schema;
    }

    /**
     * @brief Returns schemas for every registered tool.
     *
     * @return JSON array of tool schema objects.
     */
    nlohmann::json get_all_schemas() const {
        std::shared_lock lock(mutex_);
        nlohmann::json schemas = nlohmann::json::array();
        for (const auto& [name, entry] : tools_) {
            schemas.push_back(build_schema_json(entry));
        }
        return schemas;
    }

    /**
     * @brief Returns the names of every registered tool.
     *
     * @return Vector containing one entry per registered tool.
     */
    std::vector<std::string> get_tool_names() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> names;
        names.reserve(tools_.size());
        for (const auto& [name, _] : tools_) {
            names.push_back(name);
        }
        return names;
    }

    /// Returns the number of registered tools.
    size_t size() const {
        std::shared_lock lock(mutex_);
        return tools_.size();
    }

  private:
    /// Converts a registry entry into the schema shape consumed by prompts and grammar generation.
    static nlohmann::json build_schema_json(const ToolEntry& entry) {
        return nlohmann::json{{"type", "function"},
                              {"function",
                               {{"name", entry.name},
                                {"description", entry.description},
                                {"parameters", entry.parameters_schema}}}};
    }

    std::unordered_map<std::string, ToolEntry> tools_;
    mutable std::shared_mutex mutex_;
};

} // namespace zoo::tools
