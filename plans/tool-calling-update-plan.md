# Plan: Replace `<tool_call>` Sentinel with llama.cpp Template-Driven Tool Calling

## Context

Zoo-keeper currently hardcodes `<tool_call>` / `</tool_call>` sentinel tags for tool call detection and grammar constraining. This only works with Hermes/Qwen-style models. Different GGUF models use different tool calling formats (Llama 3.x uses `<|python_tag|>`, Mistral uses `[TOOL_CALLS]`, DeepSeek has its own format, etc. — 29+ formats exist).

llama.cpp already has a mature, battle-tested system in its `common/chat.*` layer that:
1. Reads the Jinja2 chat template from GGUF metadata
2. Auto-detects the model's tool calling format (29+ formats)
3. Generates format-specific GBNF grammars with lazy trigger support
4. Parses model output back into structured tool calls

We should leverage this instead of maintaining our own format-specific logic.

## Key llama.cpp APIs to Use

| API | File | Purpose |
|-----|------|---------|
| `common_chat_templates_init(model, override)` | `common/chat.h` | Init template system from loaded model |
| `common_chat_templates_apply(tmpls, inputs)` | `common/chat.h` | Render prompt + get format-specific grammar, triggers, stops |
| `common_chat_parse(text, is_partial, syntax)` | `common/chat.h` | Parse model output into `common_chat_msg` with tool calls |
| `common_chat_tool` | `common/chat.h` | Tool metadata struct (name, description, parameters JSON) |
| `common_chat_params` | `common/chat.h` | Output: prompt, grammar, grammar_lazy, grammar_triggers, additional_stops |
| `common_chat_format` | `common/chat.h` | Enum of 29+ detected formats |

## Architecture: What Changes Where

### Layer 1 (zoo::core) — Primary Changes

**New: `ToolCallingContext` in Model** — initialized at model load time via `common_chat_templates_init()`. Stores the parsed templates and detected format. This is the single source of truth for "how does this model do tool calls."

**Files to modify:**

1. **`include/zoo/core/model.hpp`** — Add:
   - `common_chat_templates_ptr chat_templates_` member (forward-declared, pimpl'd)
   - `struct ToolCallingConfig` — exposes: detected format name, whether tools are supported, grammar string, triggers, stops
   - `prepare_tool_calling(tools_metadata) -> ToolCallingConfig` method — calls `common_chat_templates_apply()` with the registered tools, returns everything needed for constrained generation
   - `parse_tool_response(text, config) -> ParsedToolResponse` method — calls `common_chat_parse()` and returns structured result
   - Remove `GrammarMode::ToolCall` concept (replaced by lazy grammar from llama.cpp)

2. **`src/core/model.cpp`** (or new `src/core/model_chat.cpp`)  — Implement:
   - Template initialization in constructor: `chat_templates_ = common_chat_templates_init(model_, "")`
   - `prepare_tool_calling()`: converts `zoo::tools::ToolMetadata` → `common_chat_tool`, calls `common_chat_templates_apply()`, extracts grammar + triggers + stops
   - `parse_tool_response()`: calls `common_chat_parse()`, maps `common_chat_tool_call` → `zoo::tools::ToolCall`

3. **`src/core/model_inference.cpp`** — Replace sentinel detection:
   - Remove hardcoded `<tool_call>` / `</tool_call>` sentinel logic (lines 125-134, 165)
   - Replace with trigger-based grammar activation using `common_chat_params::grammar_triggers`
   - Use `common_chat_params::additional_stops` for stop sequences
   - Remove `SentinelStreamFilter` usage — replace with format-aware streaming

4. **`src/core/model_prompt.cpp`** — Replace `llama_chat_apply_template()`:
   - Use `common_chat_templates_apply()` instead for prompt rendering
   - This handles tool-aware prompt formatting natively (tools go in the template, not the system prompt)

5. **`src/core/model_sampling.cpp`** — Update grammar handling:
   - Support lazy grammar mode from `common_chat_params::grammar_lazy`
   - Support `common_grammar_trigger` types (TOKEN, WORD, PATTERN, PATTERN_FULL)

### Layer 2 (zoo::tools) — Minimal Changes

Layer 2 stays model-agnostic. The key changes are removing format-specific logic:

6. **`include/zoo/internal/tools/grammar.hpp`** — **Deprecate/Remove `GrammarBuilder`**
   - Grammar generation moves to Layer 1 via `common_chat_templates_apply()`
   - `build_schema()` for structured extraction can remain (it's format-independent)

7. **`include/zoo/tools/parser.hpp`** — **Simplify `ToolCallParser`**
   - Remove `parse_sentinel()` method (sentinel tags gone)
   - Keep `parse()` heuristic as a fallback for models without tool support
   - Add a new method that accepts pre-parsed `common_chat_msg` and maps to `ToolCall`

8. **`include/zoo/internal/tools/interceptor.hpp`** — **Simplify or remove `ToolCallInterceptor`**
   - With lazy grammar + triggers, the model's output is cleanly structured
   - The interceptor's job of hiding JSON from streaming becomes unnecessary if grammar triggers handle it
   - May keep a simplified version for the "no grammar" fallback path

9. **`include/zoo/internal/tools/sentinel_stream_filter.hpp`** — **Remove entirely**
   - No more sentinel tags to filter

### Layer 3 (zoo::Agent) — Orchestration Changes

10. **`src/agent.cpp` / `src/agent/runtime_tool_loop.cpp`** — Update the agentic loop:
    - At tool registration time: call `Model::prepare_tool_calling()` to get `ToolCallingConfig`
    - Pass grammar, triggers, and stops to `Model::generate_from_history()`
    - After generation: call `Model::parse_tool_response()` instead of `ToolCallParser::parse_sentinel()`
    - Map `common_chat_tool_call` → `zoo::tools::ToolCall` for registry invocation
    - Remove `build_tool_system_prompt()` — tools are now formatted by the template itself

11. **`src/agent/runtime.cpp`** — Remove system prompt injection of tool instructions
    - The Jinja template handles tool formatting natively

### CMake Changes

12. **`CMakeLists.txt`** — Link against llama.cpp `common` library:
    - Currently `zoo` links `llama` privately. Need to also link `common` (which provides `chat.h` implementation)
    - `common` depends on: nlohmann/json (already have), minja (bundled in llama.cpp)
    - **Ask first**: This changes the CMake build structure

## Data Flow (New)

```
Model Load:
  GGUF → llama_model → common_chat_templates_init() → chat_templates_ stored in Model

Tool Registration:
  zoo::tools::ToolMetadata → convert to common_chat_tool[]
  → common_chat_templates_apply(chat_templates_, {tools, messages})
  → common_chat_params {grammar, grammar_lazy, grammar_triggers, additional_stops, format}
  → Store as ToolCallingConfig

Generation:
  1. Apply prompt via common_chat_templates_apply() (tools baked into template)
  2. Start generation with lazy grammar + triggers from ToolCallingConfig
  3. Model generates freely until trigger token appears
  4. Grammar activates → constrains to valid tool call JSON
  5. Stop on additional_stops or EOG

Parsing:
  generated_text → common_chat_parse(text, is_partial, syntax)
  → common_chat_msg {tool_calls: [{name, arguments, id}]}
  → map to zoo::tools::ToolCall
  → ToolRegistry::invoke()
```

## Migration Strategy (Incremental)

### Phase 1: Add llama.cpp `common` integration to Layer 1
- Link `common` in CMake
- Initialize `chat_templates_` in Model constructor
- Add `prepare_tool_calling()` and `parse_tool_response()` methods
- Keep existing sentinel path working as fallback
- **Test**: Load different models, verify format detection works

### Phase 2: Wire up the Agent layer
- Agent calls `prepare_tool_calling()` on tool registration
- Use returned grammar/triggers/stops in generation
- Parse output with `common_chat_parse()`
- Remove `build_tool_system_prompt()` — template handles it
- **Test**: Run tool calling with Hermes, Llama 3.x, and Mistral models

### Phase 3: Remove legacy code
- Remove `GrammarBuilder` (or keep only `build_schema()`)
- Remove `SentinelStreamFilter`
- Remove `parse_sentinel()` from `ToolCallParser`
- Simplify or remove `ToolCallInterceptor`
- Remove hardcoded `<tool_call>` detection in `model_inference.cpp`
- **Test**: Full regression with multiple model families

### Phase 4: Streaming support
- Use `common_chat_parse(text, is_partial=true, syntax)` for incremental parsing
- Replace `ToolCallInterceptor` with format-aware streaming based on `common_chat_msg_diff`
- **Test**: Streaming callbacks correctly hide tool JSON across all formats

## Key Design Decisions

1. **`common/` is C++ (not stable C API)**: It uses exceptions and nlohmann/json. Since zoo-keeper already uses nlohmann/json and the `common` layer is bundled in the submodule, this is acceptable. Wrap calls at the Layer 1 boundary to convert exceptions → `std::expected`.

2. **Template-driven vs system prompt injection**: Currently tools are described in the system prompt via `build_tool_system_prompt()`. With template-driven tool calling, the Jinja template itself formats tool descriptions. This is more correct since each model was trained with a specific tool prompt format.

3. **Grammar `build_schema()` stays**: The structured extraction grammar (for `extract()` requests) is independent of tool calling format and stays in Layer 2.

4. **Fallback for models without tool support**: If `common_chat_templates_apply()` returns `COMMON_CHAT_FORMAT_CONTENT_ONLY` (no tool support detected), fall back to the Generic format which uses a wrapper JSON grammar.

## Files to Modify (Summary)

| File | Action |
|------|--------|
| `CMakeLists.txt` | Link `common` library |
| `include/zoo/core/model.hpp` | Add template/tool calling members |
| `src/core/model.cpp` | Template init, prepare/parse methods |
| `src/core/model_inference.cpp` | Replace sentinel with trigger-based detection |
| `src/core/model_prompt.cpp` | Use `common_chat_templates_apply()` for prompts |
| `src/core/model_sampling.cpp` | Support lazy grammar + triggers |
| `include/zoo/internal/tools/grammar.hpp` | Remove `build()`, keep `build_schema()` |
| `include/zoo/tools/parser.hpp` | Remove `parse_sentinel()`, add mapping method |
| `include/zoo/internal/tools/sentinel_stream_filter.hpp` | **Delete** |
| `include/zoo/internal/tools/interceptor.hpp` | Simplify or remove |
| `src/agent.cpp` / `src/agent/runtime_tool_loop.cpp` | Use new Model APIs |
| `src/agent/runtime.cpp` | Remove `build_tool_system_prompt()` |

## Verification

1. **Build**: `scripts/build` — verify clean compilation with `common` linked
2. **Unit tests**: `scripts/test` — existing pure logic tests should still pass; sentinel-related tests will need updating
3. **Format detection test**: Load GGUF models from different families, assert correct `common_chat_format` detected
4. **Integration tests** (requires GGUF models):
   - Hermes-style model (Qwen 2.5): tool call works
   - Llama 3.x model: tool call works with `<|python_tag|>` trigger
   - Mistral model: tool call works with `[TOOL_CALLS]` trigger
   - Model without tool support: graceful fallback to Generic format
5. **Streaming**: Verify streaming callbacks don't leak tool JSON to user
6. **Formatting**: `scripts/format` passes

## Risks & Mitigations

- **`common` API instability**: llama.cpp's common layer changes frequently. Mitigate by wrapping at the Model boundary and keeping the interface minimal.
- **Exception boundary**: `common` uses exceptions. Wrap all calls in try/catch at the Layer 1 boundary, converting to `std::expected`.
- **Build size**: `common` pulls in minja (Jinja engine). This is ~2,500 lines header-only — acceptable.
