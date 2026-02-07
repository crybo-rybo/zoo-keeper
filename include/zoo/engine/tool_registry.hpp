#pragma once

#include "../types.hpp"
#include <nlohmann/json.hpp>
#include <functional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <type_traits>

namespace zoo {
namespace engine {

// ============================================================================
// Tool Handler Type
// ============================================================================

/** @brief Callable type for tool execution; takes JSON arguments and returns JSON result or Error. */
using ToolHandler = std::function<Expected<nlohmann::json>(const nlohmann::json&)>;

// ============================================================================
// Type Traits for JSON Schema Generation
// ============================================================================

namespace detail {

template<typename T>
struct json_type_name;

template<> struct json_type_name<int> {
    static constexpr const char* type = "integer";
};

template<> struct json_type_name<float> {
    static constexpr const char* type = "number";
};

template<> struct json_type_name<double> {
    static constexpr const char* type = "number";
};

template<> struct json_type_name<bool> {
    static constexpr const char* type = "boolean";
};

template<> struct json_type_name<std::string> {
    static constexpr const char* type = "string";
};

// Function traits to extract parameter types from callables
template<typename T>
struct function_traits;

// Function pointer
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

// Member function pointer (const — non-mutable lambdas)
template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

// Member function pointer (non-const — mutable lambdas)
template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

// Lambda / functor: delegate to operator()
template<typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};

// std::function
template<typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

// Build JSON schema properties from a tuple of parameter types
template<typename Tuple, size_t... Is>
nlohmann::json build_properties_impl(const std::vector<std::string>& param_names, std::index_sequence<Is...>) {
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json required = nlohmann::json::array();

    // Use fold expression to add each parameter
    ((properties[param_names[Is]] = nlohmann::json{
        {"type", json_type_name<std::tuple_element_t<Is, Tuple>>::type}
    }, required.push_back(param_names[Is])), ...);

    return nlohmann::json{
        {"type", "object"},
        {"properties", properties},
        {"required", required}
    };
}

template<typename Tuple>
nlohmann::json build_properties(const std::vector<std::string>& param_names) {
    constexpr size_t N = std::tuple_size_v<Tuple>;
    return build_properties_impl<Tuple>(param_names, std::make_index_sequence<N>{});
}

// Extract a single argument from JSON
template<typename T>
T extract_arg(const nlohmann::json& args, const std::string& name) {
    return args.at(name).get<T>();
}

// Invoke a callable with arguments extracted from JSON
template<typename Func, typename Tuple, size_t... Is>
auto invoke_with_json_impl(const Func& func, const nlohmann::json& args,
                           const std::vector<std::string>& param_names,
                           std::index_sequence<Is...>) {
    return func(extract_arg<std::tuple_element_t<Is, Tuple>>(args, param_names[Is])...);
}

template<typename Func, typename Tuple>
auto invoke_with_json(const Func& func, const nlohmann::json& args,
                      const std::vector<std::string>& param_names) {
    constexpr size_t N = std::tuple_size_v<Tuple>;
    return invoke_with_json_impl<Func, Tuple>(
        func, args, param_names,
        std::make_index_sequence<N>{});
}

// Wrap a return value into JSON
template<typename T>
nlohmann::json wrap_result(T&& value) {
    return nlohmann::json{{"result", std::forward<T>(value)}};
}

} // namespace detail

// ============================================================================
// Tool Registration Entry
// ============================================================================

/** @brief Holds metadata and handler for a single registered tool. */
struct ToolEntry {
    std::string name;                    ///< Unique tool name used for invocation
    std::string description;             ///< Human-readable description of what the tool does
    nlohmann::json parameters_schema;    ///< JSON Schema describing expected parameters
    ToolHandler handler;                 ///< Callable that executes the tool logic
};

// ============================================================================
// ToolRegistry
// ============================================================================

/**
 * @brief Registry for tool definitions, schema generation, and invocation.
 *
 * Supports both template-based registration (automatic schema generation from
 * function signatures) and manual registration with explicit JSON schemas.
 *
 * @threadsafety All public methods are thread-safe. Read operations use shared
 * locks; write operations (register_tool) use exclusive locks.
 */
class ToolRegistry {
public:
    /**
     * Template-based registration: extracts parameter types and generates schema.
     *
     * @param name Tool name
     * @param description Tool description
     * @param param_names Parameter names (must match function arity)
     * @param func Callable to invoke
     */
    template<typename Func>
    void register_tool(const std::string& name, const std::string& description,
                       const std::vector<std::string>& param_names, Func func) {
        using traits = detail::function_traits<Func>;
        using args_tuple = typename traits::args_tuple;

        if (param_names.size() != traits::arity) {
            throw std::invalid_argument(
                "Parameter name count (" + std::to_string(param_names.size()) +
                ") does not match function arity (" + std::to_string(traits::arity) + ")");
        }

        // Generate schema from function signature
        nlohmann::json schema;
        if constexpr (traits::arity == 0) {
            schema = nlohmann::json{
                {"type", "object"},
                {"properties", nlohmann::json::object()},
                {"required", nlohmann::json::array()}
            };
        } else {
            schema = detail::build_properties<args_tuple>(param_names);
        }

        // Create handler that extracts args from JSON and calls the function
        auto captured_names = param_names;
        ToolHandler handler = [f = std::move(func), names = std::move(captured_names)](
            const nlohmann::json& args) -> Expected<nlohmann::json> {
            try {
                if constexpr (traits::arity == 0) {
                    auto result = f();
                    return detail::wrap_result(std::move(result));
                } else {
                    auto result = detail::invoke_with_json<decltype(f), args_tuple>(
                        f, args, names);
                    return detail::wrap_result(std::move(result));
                }
            } catch (const nlohmann::json::exception& e) {
                return tl::unexpected(Error{
                    ErrorCode::ToolExecutionFailed,
                    std::string("JSON argument error: ") + e.what()
                });
            } catch (const std::exception& e) {
                return tl::unexpected(Error{
                    ErrorCode::ToolExecutionFailed,
                    std::string("Tool execution failed: ") + e.what()
                });
            }
        };

        register_tool(name, description, std::move(schema), std::move(handler));
    }

    /**
     * @brief Manual registration with an explicit JSON schema and handler.
     *
     * @param name Tool name (must be unique)
     * @param description Human-readable tool description
     * @param schema JSON Schema describing the tool's parameters
     * @param handler Callable that executes the tool logic
     */
    void register_tool(const std::string& name, const std::string& description,
                       nlohmann::json schema, ToolHandler handler) {
        std::unique_lock lock(mutex_);
        tools_.insert_or_assign(name, ToolEntry{name, description, std::move(schema), std::move(handler)});
    }

    /** @brief Check whether a tool with the given name is registered. */
    bool has_tool(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return tools_.find(name) != tools_.end();
    }

    /** @brief Invoke a registered tool by name with the given JSON arguments. */
    Expected<nlohmann::json> invoke(const std::string& name, const nlohmann::json& args) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return tl::unexpected(Error{
                ErrorCode::ToolNotFound,
                "Tool not found: " + name
            });
        }
        return it->second.handler(args);
    }

    /** @brief Get the JSON function-calling schema for a single tool, or empty JSON if not found. */
    nlohmann::json get_tool_schema(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return nlohmann::json{};
        }
        return build_schema_json(it->second);
    }

    /**
     * @brief Direct access to the parameters schema without the wrapper envelope.
     *
     * Avoids constructing the full {"type":"function","function":{...}} wrapper
     * when only the parameters are needed (e.g., argument validation).
     *
     * @return Pointer to the parameters_schema, or nullptr if tool not found.
     */
    const nlohmann::json* get_parameters_schema(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return nullptr;
        }
        return &it->second.parameters_schema;
    }

    /** @brief Get an array of JSON function-calling schemas for all registered tools. */
    nlohmann::json get_all_schemas() const {
        std::shared_lock lock(mutex_);
        nlohmann::json schemas = nlohmann::json::array();
        for (const auto& [name, entry] : tools_) {
            schemas.push_back(build_schema_json(entry));
        }
        return schemas;
    }

    /** @brief Return a list of all registered tool names. */
    std::vector<std::string> get_tool_names() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> names;
        names.reserve(tools_.size());
        for (const auto& [name, _] : tools_) {
            names.push_back(name);
        }
        return names;
    }

    /** @brief Return the number of registered tools. */
    size_t size() const {
        std::shared_lock lock(mutex_);
        return tools_.size();
    }

private:
    static nlohmann::json build_schema_json(const ToolEntry& entry) {
        return nlohmann::json{
            {"type", "function"},
            {"function", {
                {"name", entry.name},
                {"description", entry.description},
                {"parameters", entry.parameters_schema}
            }}
        };
    }

    std::unordered_map<std::string, ToolEntry> tools_;
    mutable std::shared_mutex mutex_;
};

} // namespace engine
} // namespace zoo
