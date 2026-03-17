---
paths:
  - "include/zoo/tools/**"
  - "include/zoo/internal/tools/**"
---

# Tools Layer Rules (zoo::tools)

- **Header-only layer with zero dependency on llama.cpp or Layer 1 (zoo::core).**
- Operates entirely on strings and `nlohmann::json` — no model types.
- `ToolRegistry` is the public registration and invocation surface.
- `ToolCallParser` detects tool calls in model output text.
- `GrammarBuilder` and `ToolCallInterceptor` are internal (`include/zoo/internal/tools/`).
- JSON schema is generated automatically from C++ function signatures via template metaprogramming.
- Supported argument types: `int`, `float`, `double`, `bool`, `std::string`.
- Test all tools-layer logic via unit tests only — no model required.
