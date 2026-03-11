# Migration to 1.0

This guide summarizes the main consumer-facing changes from the `0.2.x` line to the upcoming `1.0` release.

## CMake target

Use `ZooKeeper::zoo` as the primary target for all new integrations:

```cmake
find_package(ZooKeeper CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ZooKeeper::zoo)
```

`ZooKeeper::zoo_core` may still exist as a compatibility shim, but it is no longer the primary documented target.

## Tool registration

Register tools directly on `zoo::Agent` in both supported modes:

- typed registration for native C++ callables
- manual-schema registration for JSON-backed handlers

If older code drops to `zoo::tools::ToolRegistry` just to register manual-schema tools, move that registration to `Agent::register_tool(...)` unless you intentionally need the lower-level registry.

## Tool results on responses

Tool-loop activity is now surfaced as explicit `Response::tool_invocations` records instead of requiring consumers to infer behavior from chat history:

- `Succeeded`
- `ValidationFailed`
- `ExecutionFailed`

Update any response-handling code that previously inspected injected tool messages to instead read `response->tool_invocations`.

## JSON configuration loading

Zoo-Keeper now ships opt-in JSON helpers for public config types:

```cpp
#include <zoo/core/json.hpp>

std::ifstream file("config.json");
auto json = nlohmann::json::parse(file);
zoo::Config config = json.get<zoo::Config>();
```

This is the recommended path when loading `zoo::Config` from files.

## Internal headers

Treat `include/zoo/` as the supported public boundary. Internal headers under `include/zoo/internal/` are implementation details and are not part of the intended installed API surface.
