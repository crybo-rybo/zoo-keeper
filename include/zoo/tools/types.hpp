#pragma once

#include <zoo/core/types.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <string>
#include <optional>
#include <functional>

namespace zoo::tools {

// Tool call extracted from model output
struct ToolCall {
    std::string id;
    std::string name;
    nlohmann::json arguments;

    bool operator==(const ToolCall& other) const = default;
};

// Callable type for tool execution
using ToolHandler = std::function<Expected<nlohmann::json>(const nlohmann::json&)>;

// Metadata and handler for a registered tool
struct ToolEntry {
    std::string name;
    std::string description;
    nlohmann::json parameters_schema;
    ToolHandler handler;
};

} // namespace zoo::tools
