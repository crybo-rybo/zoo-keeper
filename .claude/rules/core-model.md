---
paths:
  - "src/core/**"
  - "include/zoo/core/**"
  - "include/zoo/internal/core/**"
---

# Core Layer Rules (zoo::core)

- **All llama.cpp calls live in `src/core/model*.cpp` — nowhere else in the codebase.**
- `model.hpp` uses forward declarations for llama types. Never `#include "llama.h"` in any public header.
- `role_to_string()` returns `const char*` (static storage duration) — safe for `llama_chat_message`.
- `validate_role_sequence()` is a free function in `types.hpp` — pure logic, no llama dependency.
- Model is synchronous and single-threaded. It is NOT thread-safe on its own; Agent wraps it with `model_mutex_`.
- Error handling uses `std::expected` (C++23). No exceptions.
- Model directly owns `llama_model*`, `llama_context*`, `llama_sampler*`, `llama_vocab*` — no abstraction layer.
- Keep individual `model_*.cpp` files under 300 lines. If a file grows, split by responsibility.
