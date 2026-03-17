---
paths:
  - "tests/**"
---

# Testing Rules

- **Never `using namespace zoo;`** — `zoo::testing` clashes with `::testing` (GoogleTest).
- Unit tests cover pure logic only: types, tools, validation, parsing, grammar, interceptor, batch.
- Model and Agent behavior requires integration tests with a real GGUF model.
- Prefer deterministic tests. Use fixtures from `tests/fixtures/` for sample data.
- Test file naming: `tests/unit/test_<component>.cpp`.
- Run focused tests while iterating: `scripts/test -R TestSuiteName`.
- All tests must pass before any PR: `scripts/test`.
