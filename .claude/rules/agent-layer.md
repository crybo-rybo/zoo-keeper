---
paths:
  - "src/agent/**"
  - "src/agent.cpp"
  - "include/zoo/agent.hpp"
  - "include/zoo/internal/agent/**"
---

# Agent Layer Rules (zoo::Agent)

- Agent owns the inference thread. Callers submit via `chat()` / `complete()` and receive `std::future<Response>`.
- **All callbacks and tool handlers execute on the inference thread.**
- Model access is protected by `model_mutex_` — never access Model from the calling thread while Agent is running.
- `runtime.cpp` is the core execution engine. If it exceeds ~500 lines, decompose by responsibility (request processing, tool execution, streaming, lifecycle).
- `backend_model.cpp` adapts `zoo::core::Model` to the internal `Backend` interface.
- The agentic tool loop: generate → detect tool calls → execute → append result → re-generate.
- Agent composes Layer 1 (Model) + Layer 2 (ToolRegistry). Never bypass the layer boundary.
