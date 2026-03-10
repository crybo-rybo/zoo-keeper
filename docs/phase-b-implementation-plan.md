# Phase B Implementation Plan

## Purpose

Phase B turns Epic 2 from [cleanup-roadmap.md](cleanup-roadmap.md) into an execution plan that matches the codebase as it exists after Phase A.

This phase should be treated as a real subsystem redesign, not a set of cosmetic follow-ups. The tool system is the library's most distinctive feature, and it is still carrying too much contract ambiguity.

## What Phase A Already Changed

Phase A already removed some of the structural blockers that the original roadmap called out:

- `zoo` is now the primary public target.
- `Agent` runtime implementation lives in [`src/agent.cpp`](../src/agent.cpp), not in the installed header.
- internal headers are no longer installed.
- `Agent` already exposes a manual-schema registration overload.

That means Phase B should not spend time rediscovering those changes. The remaining work is almost entirely about making the tool subsystem coherent.

## Current State Audit

The current tooling code is functional, but the contract is still fragmented in a few important ways:

1. `Agent` and `ToolRegistry` still feel like two different public stories.
   - `Agent` has both registration overloads.
   - docs and examples still teach advanced usage through `ToolRegistry` directly.

2. The registration pipeline is duplicated.
   - typed registration logic exists in both [`include/zoo/agent.hpp`](../include/zoo/agent.hpp) and [`include/zoo/tools/registry.hpp`](../include/zoo/tools/registry.hpp).
   - schema building, handler wrapping, and signature validation are not centralized.

3. Schema support is still "best effort" instead of explicit.
   - validation only checks required fields plus primitive types.
   - grammar generation only understands flat required arguments.
   - unsupported schema keywords are effectively ignored.
   - the manual-schema example should use the supported subset and the main `Agent` API, not a nested schema that the runtime cannot guarantee.

4. Tool order is nondeterministic.
   - `ToolRegistry` stores tools in an `unordered_map`.
   - schema dumps, grammar alternatives, and tool name lists can vary by hash iteration order.
   - the tests currently tolerate that nondeterminism.

5. `Response::tool_calls` does not describe what happened.
   - it stores injected `Message` objects, not tool invocation records.
   - it loses tool name, parsed arguments, outcome type, and whether a failure happened during validation or execution.

## Decisions

These decisions should be locked before implementation starts.

### 1. `Agent` becomes the canonical tool API

`Agent::register_tool(...)` is the public path we teach and optimize for. `ToolRegistry` remains public, but it becomes a lower-level utility, not the "advanced mode" that users are pushed toward.

### 2. We do not implement general JSON Schema in Phase B

Phase B should implement a deliberately small supported subset and fail fast outside it. The current runtime is nowhere near full JSON Schema, and pretending otherwise is the source of most of the current ambiguity.

### 3. Registration order becomes the canonical tool order

Do not sort tools alphabetically and do not rely on hash iteration order. Preserve registration order everywhere:

- prompt schema dumps,
- grammar generation,
- `get_tool_names()`,
- response observability,
- tests.

If an existing tool name is re-registered, it should keep its original slot and replace its definition in place.

### 4. We should accept one intentional breaking API change

`Response::tool_calls` should be replaced, not kept as the long-term public shape. Carrying both the old semantic lie and the new explicit type will make the cleanup weaker.

## Supported Schema Contract for Phase B

The public manual registration API can continue to accept `nlohmann::json`, but registration must normalize it into one supported internal model.

Phase B schema support should be:

- top-level `type: "object"`,
- `properties` object required,
- property types limited to `string`, `integer`, `number`, `boolean`,
- `required` array optional,
- property `description` allowed,
- property `enum` allowed when every value matches the declared primitive type,
- `additionalProperties` allowed only when absent or `false`.

Phase B must reject at registration time:

- nested `object` properties,
- arrays and `items`,
- `oneOf`, `anyOf`, `allOf`, `not`,
- `$ref`,
- numeric and string bounds (`minimum`, `maximum`, `pattern`, `minLength`, etc.),
- any unknown property keyword that would change validation semantics.

This gives us a contract that is:

- strong enough to justify manual schemas,
- small enough to validate and generate grammar for correctly,
- identical across validation, prompt generation, and grammar mode.

## Target Internal Shape

The key implementation move is to stop treating raw JSON as the internal schema model.

Phase B should introduce a normalized internal definition roughly shaped like this:

```cpp
enum class ToolValueType { String, Integer, Number, Boolean };

struct ToolParameter {
    std::string name;
    ToolValueType type;
    bool required = false;
    std::string description;
    std::vector<std::string> enum_values;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::vector<ToolParameter> parameters; // registration order
    nlohmann::json source_schema;          // normalized public-facing schema
    ToolHandler handler;
};
```

The exact type names can change, but the architecture should not:

- registration validates once,
- registry stores normalized definitions,
- validator works from normalized definitions,
- grammar builder works from normalized definitions,
- prompt/schema export works from normalized definitions.

## Public Response Model

Replace the current response field with an explicit tool invocation record.

Recommended shape:

```cpp
enum class ToolInvocationStatus {
    Succeeded,
    ValidationFailed,
    ExecutionFailed
};

struct ToolInvocation {
    std::string id;
    std::string name;
    std::string arguments_json;
    ToolInvocationStatus status;
    std::optional<std::string> result_json;
    std::optional<Error> error;
};
```

And then:

```cpp
struct Response {
    std::string text;
    TokenUsage usage;
    Metrics metrics;
    std::vector<ToolInvocation> tool_invocations;
};
```

Why this shape:

- it is explicit without dragging `nlohmann::json` into `Response`,
- it preserves exact serialized arguments and results,
- it distinguishes validation failures from execution failures,
- it lets docs and examples explain tool behavior directly instead of via `Message`.

## Work Packages

### Work Package 1 - Centralize tool registration

Create one shared registration pipeline used by both:

- `Agent::register_tool(name, description, param_names, func)`,
- `Agent::register_tool(name, description, schema, handler)`,
- `ToolRegistry::register_tool(...)`.

Concrete changes:

- extract typed callable introspection and handler wrapping out of `Agent` and `ToolRegistry`,
- normalize both typed and manual schemas into the same `ToolDefinition`,
- make schema validation a required registration step.

Exit criteria:

- no duplicate registration pipeline across `Agent` and `ToolRegistry`,
- typed and manual registration produce identical downstream behavior,
- unsupported manual schemas fail during registration.

### Work Package 2 - Replace `ToolRegistry` storage with deterministic ordering

Move from `unordered_map<std::string, ToolEntry>` to an ordered representation.

Recommended structure:

- `std::vector<ToolDefinition> tools_`,
- `std::unordered_map<std::string, size_t> index_by_name_`.

Concrete behavior:

- first registration appends,
- re-registration replaces the stored definition in the existing slot,
- exported schema order and grammar order match registration order exactly.

Exit criteria:

- `get_all_schemas()` is stable across runs,
- `get_tool_names()` is stable across runs,
- grammar alternatives are emitted in stable order,
- tests assert exact order.

### Work Package 3 - Split validation from retry bookkeeping

The current `ErrorRecovery` class does two unrelated jobs.

Phase B should split it into:

- a pure tool-argument validator,
- simple retry bookkeeping local to the agent loop.

Concrete changes:

- rename or replace `ErrorRecovery` with a validator that returns structured validation failures,
- move retry counts into `Agent::Impl::process_request()`,
- make validation use normalized definitions rather than raw JSON lookups.

Exit criteria:

- validation logic has no retry state,
- retry policy is visibly an agent concern,
- validation failures are structured enough to populate `ToolInvocation`.

### Work Package 4 - Rebuild grammar generation on top of the supported subset

Once schema normalization exists, grammar generation should use it directly.

Concrete changes:

- generate grammar from ordered normalized parameters, not from arbitrary schema JSON,
- support optional primitive parameters in canonical parameter order,
- emit enum constraints when present,
- reject unsupported tools from grammar generation instead of silently degrading.

Important rule:

- do not allow grammar mode to advertise a capability that unconstrained validation cannot also honor.

Exit criteria:

- grammar support matches the documented schema subset,
- unsupported schemas fail at registration instead of disappearing into fallback behavior,
- tests cover optional fields, enum fields, and deterministic tool ordering.

### Work Package 5 - Replace `Response::tool_calls` with explicit invocation records

Update the agent loop so it records one `ToolInvocation` per attempted tool call.

Concrete changes:

- capture parsed tool name, id, and serialized arguments as soon as a tool call is detected,
- record `ValidationFailed` when schema validation rejects arguments,
- record `ExecutionFailed` when the handler returns an error or throws,
- record `Succeeded` with the serialized JSON result when execution succeeds,
- stop populating `Response` from tool-result `Message` objects.

Implementation note:

- the conversation history should still use `Message::tool(...)` internally because the model consumes messages,
- the public `Response` should stop exposing those internal message objects.

Exit criteria:

- examples can inspect tool behavior without reading conversation history,
- response field names match actual contents,
- docs no longer need to explain away the old mismatch.

### Work Package 6 - Rewrite docs and examples around the new contract

This is part of Phase B, not cleanup afterward.

Concrete changes:

- rewrite [`docs/tools.md`](./tools.md) around `Agent` as the primary entry point,
- update [`docs/getting-started.md`](./getting-started.md) and [`docs/examples.md`](./examples.md) for `tool_invocations`,
- replace the manual-schema example with one that uses `Agent::register_tool(..., schema, handler)` and stays inside the supported subset,
- update [`docs/building.md`](./building.md) to reference the replacement example,
- remove any wording that implies nested schemas are supported.

Exit criteria:

- no official example routes users through raw registry access unless that is the point of the example,
- no doc claims a schema feature the runtime rejects,
- the response examples use explicit tool invocation records.

## Recommended Execution Order

Land Phase B in this sequence:

1. Add normalized schema types plus the shared registration pipeline.
2. Replace registry storage with deterministic ordering.
3. Replace validation with a pure validator based on normalized schemas.
4. Rebuild grammar generation on top of the normalized model.
5. Introduce `ToolInvocation` and update the agent response path.
6. Rewrite examples, tests, and docs to the new contract.

This order matters. Do not redesign the response model first while the registry and schema contract are still ambiguous underneath it.

## Test Plan

Add or rewrite unit coverage in these areas:

- schema normalization success cases for typed and manual registration,
- registration rejection for unsupported keywords and nested schemas,
- deterministic ordering in `get_tool_names()`, `get_all_schemas()`, and grammar output,
- enum validation and optional-parameter validation,
- tool invocation response records for success, validation failure, and execution failure,
- overwrite-in-place behavior preserving registration order.

Specific test file changes:

- keep extending [`tests/unit/test_tool_registry.cpp`](../tests/unit/test_tool_registry.cpp),
- replace or heavily rewrite [`tests/unit/test_error_recovery.cpp`](../tests/unit/test_error_recovery.cpp),
- extend [`tests/unit/test_grammar_builder.cpp`](../tests/unit/test_grammar_builder.cpp) to assert exact ordering,
- update [`tests/unit/test_types.cpp`](../tests/unit/test_types.cpp) for the new response field.

If a small helper seam is needed inside `Agent` to unit-test `ToolInvocation` recording without a live model, add it now. That is a Phase B-friendly seam and not a premature Phase C refactor.

## Breaking Changes We Should Accept

These are worth doing in one deliberate pass:

- rename `Response::tool_calls` to `Response::tool_invocations`,
- stop teaching `ToolRegistry` as the primary advanced registration API,
- reject manual schemas that previously "worked" only because the runtime ignored unsupported parts,
- replace the nested manual-schema example.

If a temporary compatibility shim is useful while landing the branch, keep it short-lived and remove it before merge.

## Done Means

Phase B is done when:

- typed and manual registration follow one pipeline,
- supported schema features are explicit and enforced,
- tool ordering is deterministic,
- response tool data is explicit and trustworthy,
- examples stop teaching unsupported schema patterns,
- the docs describe exactly one obvious tool story.
