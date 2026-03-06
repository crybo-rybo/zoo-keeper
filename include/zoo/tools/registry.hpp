#pragma once

#include "types.hpp"
#include <nlohmann/json.hpp>
#include <functional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <type_traits>

namespace zoo::tools {

namespace detail {

template<typename T>
struct json_type_name;

template<> struct json_type_name<int> { static constexpr const char* type = "integer"; };
template<> struct json_type_name<float> { static constexpr const char* type = "number"; };
template<> struct json_type_name<double> { static constexpr const char* type = "number"; };
template<> struct json_type_name<bool> { static constexpr const char* type = "boolean"; };
template<> struct json_type_name<std::string> { static constexpr const char* type = "string"; };

template<typename T>
struct function_traits;

template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

template<typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};

template<typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> {
    using return_type = R;
    using args_tuple = std::tuple<std::decay_t<Args>...>;
    static constexpr size_t arity = sizeof...(Args);
};

template<typename Tuple, size_t... Is>
nlohmann::json build_properties_impl(const std::vector<std::string>& param_names, std::index_sequence<Is...>) {
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json required = nlohmann::json::array();
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

template<typename T>
T extract_arg(const nlohmann::json& args, const std::string& name) {
    return args.at(name).get<T>();
}

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
        func, args, param_names, std::make_index_sequence<N>{});
}

template<typename T>
nlohmann::json wrap_result(T&& value) {
    return nlohmann::json{{"result", std::forward<T>(value)}};
}

} // namespace detail

class ToolRegistry {
public:
    template<typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                       const std::vector<std::string>& param_names, Func func) {
        using traits = detail::function_traits<Func>;
        using args_tuple = typename traits::args_tuple;

        if (param_names.size() != traits::arity) {
            return std::unexpected(Error{
                ErrorCode::InvalidToolSignature,
                "Parameter name count (" + std::to_string(param_names.size()) +
                ") does not match function arity (" + std::to_string(traits::arity) + ")"
            });
        }

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
                return std::unexpected(Error{
                    ErrorCode::ToolExecutionFailed,
                    std::string("JSON argument error: ") + e.what()
                });
            } catch (const std::exception& e) {
                return std::unexpected(Error{
                    ErrorCode::ToolExecutionFailed,
                    std::string("Tool execution failed: ") + e.what()
                });
            }
        };

        register_tool(name, description, std::move(schema), std::move(handler));
        return {};
    }

    void register_tool(const std::string& name, const std::string& description,
                       nlohmann::json schema, ToolHandler handler) {
        std::unique_lock lock(mutex_);
        tools_.insert_or_assign(name, ToolEntry{name, description, std::move(schema), std::move(handler)});
    }

    bool has_tool(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return tools_.find(name) != tools_.end();
    }

    Expected<nlohmann::json> invoke(const std::string& name, const nlohmann::json& args) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return std::unexpected(Error{
                ErrorCode::ToolNotFound,
                "Tool not found: " + name
            });
        }
        return it->second.handler(args);
    }

    nlohmann::json get_tool_schema(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return nlohmann::json{};
        }
        return build_schema_json(it->second);
    }

    std::optional<nlohmann::json> get_parameters_schema(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) {
            return std::nullopt;
        }
        return it->second.parameters_schema;
    }

    nlohmann::json get_all_schemas() const {
        std::shared_lock lock(mutex_);
        nlohmann::json schemas = nlohmann::json::array();
        for (const auto& [name, entry] : tools_) {
            schemas.push_back(build_schema_json(entry));
        }
        return schemas;
    }

    std::vector<std::string> get_tool_names() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> names;
        names.reserve(tools_.size());
        for (const auto& [name, _] : tools_) {
            names.push_back(name);
        }
        return names;
    }

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

} // namespace zoo::tools
