# Tool Calling Implementation Plan

## Goal

Implement reliable, grammar-constrained tool calling with chain-of-thought support
and proper streaming control. The model can think freely, then emit a tool call in a
guaranteed-valid format — or just respond normally.

## Architecture: Sentinel + Lazy Grammar

**Core idea:** Use `llama_sampler_init_grammar_lazy_patterns` to let the model generate
freely (streamable to the user) until a trigger pattern is detected. Once triggered,
a GBNF grammar constrains the remainder to valid tool call JSON.

```
Normal response:      "The answer is 42."              → streamed to user
CoT + tool call:      "Let me calculate.\n<tool_call>  → streamed | buffered
                       {"name":"add","arguments":{"a":2,"b":2}}\n</tool_call>"
```

The sentinel `<tool_call>` is:
- Instructed via system prompt (with few-shot example)
- Enforced by the GBNF grammar once triggered (model can't produce malformed JSON after it)
- Unambiguous (won't appear in normal text)
- Enables CoT (model can reason before deciding to call a tool)

## Key llama.cpp APIs

| API | Purpose |
|-----|---------|
| `llama_sampler_init_grammar_lazy_patterns` | Lazy grammar — free generation until trigger, then constrained |
| `llama_model_chat_template(model, nullptr)` | Get chat template to detect native tool support |
| `llama_vocab_get_text` / `llama_vocab_is_control` | Inspect tokens if needed |
| `llama_sampler_chain_add` / `llama_sampler_chain_remove` | Manage sampler chain dynamically |

### Lazy Grammar Sampler Signature

```c
llama_sampler * llama_sampler_init_grammar_lazy_patterns(
    const llama_vocab * vocab,
    const char * grammar_str,        // GBNF grammar (applied after trigger)
    const char * grammar_root,       // root rule name
    const char ** trigger_patterns,  // regex patterns that activate the grammar
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens, // token IDs that activate the grammar
    size_t num_trigger_tokens);
```

**Behavior:** The model generates freely. The sampler scans output for trigger patterns
(from the start of generation). When a match is found, the grammar activates and constrains
all subsequent tokens. Content is fed to the grammar starting from the first capture group.

## Implementation Phases

### Phase 1: Tool Support Detection

**File:** `src/core/model.cpp`, `include/zoo/core/model.hpp`

Detect whether the loaded model has tool-calling capability by inspecting the chat template.

```cpp
// In Model, after loading:
bool Model::detect_tool_support() const {
    if (!tmpl_) return false;
    std::string_view t(tmpl_);
    // Check for tool-related constructs in the Jinja template
    return t.find("tool_calls") != std::string_view::npos
        || t.find("tools") != std::string_view::npos;
}
```

Also check for a dedicated tool_use template:
```cpp
const char* tool_tmpl = llama_model_chat_template(llama_model_, "tool_use");
if (tool_tmpl) { /* model has explicit tool support */ }
```

**Expose:** `bool supports_tools() const` on Model.

**Agent behavior:** If `!model_->supports_tools()`, skip tool registration and log a warning.
This prevents users from registering tools on models that can't use them.

### Phase 2: Dynamic GBNF Grammar Generation

**File:** `include/zoo/tools/grammar.hpp` (new, header-only)

Generate a GBNF grammar dynamically from the registered tools in the `ToolRegistry`.
The grammar constrains tool calls to known tool names with correctly-typed arguments.

**Example:** Given tools `add(a: int, b: int)` and `multiply(a: double, b: double)`:

```gbnf
root      ::= tool-call
tool-call ::= add-call | multiply-call

add-call      ::= "{" ws "\"name\"" ws ":" ws "\"add\"" ws ","
                   ws "\"arguments\"" ws ":" ws add-args ws "}"
multiply-call ::= "{" ws "\"name\"" ws ":" ws "\"multiply\"" ws ","
                   ws "\"arguments\"" ws ":" ws multiply-args ws "}"

add-args      ::= "{" ws "\"a\"" ws ":" ws integer ws ","
                   ws "\"b\"" ws ":" ws integer ws "}"
multiply-args ::= "{" ws "\"a\"" ws ":" ws number ws ","
                   ws "\"b\"" ws ":" ws number ws "}"

# Primitives
integer ::= ("-"? ([0-9] | [1-9] [0-9]{0,15}))
number  ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]{1,15})?
string  ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
boolean ::= "true" | "false"
ws      ::= | " " | "\n" [ \t]{0,20}
```

**Key design decisions:**
- Each tool becomes a separate alternative in the grammar (context-free but exhaustive)
- Argument types map from the existing `ToolRegistry` JSON schema types:
  - `"integer"` → `integer` rule
  - `"number"` → `number` rule
  - `"string"` → `string` rule
  - `"boolean"` → `boolean` rule
- Tool names are literal strings (the grammar prevents the model from inventing tools)
- Argument names are literal strings (prevents misspelled parameter names)

**Implementation:**

```cpp
namespace zoo::tools {

class GrammarBuilder {
public:
    // Build GBNF grammar string from registered tools
    static std::string build(const ToolRegistry& registry);

private:
    static std::string type_to_rule(const std::string& json_type);
    static std::string build_tool_call_rule(const std::string& name,
                                             const nlohmann::json& params_schema);
};

} // namespace zoo::tools
```

The `build()` method iterates over `registry.get_tool_names()`, gets each tool's
parameter schema via `registry.get_parameters_schema(name)`, and generates the
corresponding GBNF rules.

### Phase 3: Sampler Chain Integration

**File:** `src/core/model.cpp`, `include/zoo/core/model.hpp`

Add the lazy grammar sampler to the Model's sampler chain when tools are configured.

**New Model methods:**

```cpp
// Set the tool call grammar (called by Agent after tools are registered)
void Model::set_tool_grammar(const std::string& grammar_str);

// Clear the tool grammar (if tools are unregistered)
void Model::clear_tool_grammar();
```

**Sampler chain management:**

The sampler chain currently is:
```
[penalties] → [top_k] → [top_p] → [temperature] → [dist/greedy]
```

With tool grammar, it becomes:
```
[penalties] → [top_k] → [top_p] → [temperature] → [lazy_grammar] → [dist/greedy]
```

The lazy grammar sampler is inserted before the final distribution sampler. It has no
effect until a trigger pattern is matched, so normal generation is unaffected.

**Trigger configuration:**

```cpp
// Trigger pattern: the sentinel string
const char* trigger = "<tool_call>";

// Build sampler
auto* grammar_sampler = llama_sampler_init_grammar_lazy_patterns(
    vocab_,
    grammar_str.c_str(),
    "root",
    &trigger, 1,     // trigger patterns
    nullptr, 0        // no trigger tokens (could add <|python_tag|> here)
);
```

**Optional native token trigger:** If the model has `<|python_tag|>` in its vocabulary
(detected during Phase 1), add it as a trigger token alongside the sentinel pattern.
This provides dual-path triggering — native token for models that emit it, sentinel
for models that follow the system prompt instruction.

### Phase 4: Streaming Control

**File:** `src/core/model.cpp`

Modify `run_inference()` to detect when the grammar has triggered and switch from
streaming to buffering.

**Detection strategy:**

Since the trigger pattern is a known string (`<tool_call>`), detection is simple:
scan the accumulated `generated_text` for the sentinel.

```cpp
// In the token generation loop within run_inference():
generated_text.append(buff, n);

// Check if we've entered tool call mode
if (!in_tool_call && generated_text.find("<tool_call>") != std::string::npos) {
    in_tool_call = true;
    // Don't stream from here on — the rest is tool call JSON
}

if (!in_tool_call && on_token) {
    (*on_token)(std::string_view(buff, n));
}
```

**New return type for run_inference (or generate_from_history):**

Extend `GenerationResult` to indicate whether a tool call was detected:

```cpp
struct GenerationResult {
    std::string text;
    int prompt_tokens = 0;
    bool contains_tool_call = false;  // true if <tool_call> sentinel was detected
};
```

The Agent can then use `contains_tool_call` to decide whether to parse for tool calls,
rather than speculatively parsing every response.

### Phase 5: Agent Integration

**File:** `include/zoo/agent.hpp`

Update the Agent's `process_request()` to leverage grammar-constrained tool calling.

**Changes to the tool loop:**

```cpp
// In process_request(), after generate_from_history():
if (generated->contains_tool_call) {
    // Extract tool call JSON from between <tool_call> and </tool_call>
    // Grammar guarantees this is valid JSON with correct structure
    auto tool_call = extract_tool_call(generated->text);

    // No need for ToolCallParser heuristic — grammar enforced validity
    // No need for ErrorRecovery argument validation — grammar enforced types

    // Execute tool directly
    auto result = tool_registry_.invoke(tool_call.name, tool_call.arguments);
    // ... inject result, continue loop
} else {
    // Normal response — already streamed to user
    // ... build Response, return
}
```

**Simplifications enabled by grammar:**
- `ToolCallParser` becomes a simple sentinel extractor (no heuristic JSON scanning)
- `ErrorRecovery` argument validation is largely unnecessary (grammar prevents bad args)
- Streaming works correctly — free text is streamed, tool calls are buffered
- No tool call JSON ever leaks to the user

### Phase 6: System Prompt Construction

**File:** `include/zoo/agent.hpp`

Update `build_tool_system_prompt()` to instruct the model on the sentinel format
with a few-shot example.

```cpp
std::string build_tool_system_prompt(const std::string& base_prompt) const {
    auto schemas = tool_registry_.get_all_schemas();
    if (schemas.empty()) return base_prompt;

    return base_prompt +
        "\n\nYou have access to tools. When you want to call a tool, "
        "wrap the call in <tool_call> tags like this:\n\n"
        "<tool_call>\n"
        "{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n"
        "</tool_call>\n\n"
        "Rules:\n"
        "- You may think and reason before calling a tool.\n"
        "- Place the tool call JSON inside <tool_call></tool_call> tags.\n"
        "- Only call one tool at a time.\n"
        "- After receiving a tool result, incorporate it into a natural response.\n"
        "- If no tool is needed, respond normally without tags.\n\n"
        "Available tools:\n" + schemas.dump(2);
}
```

## File Changes Summary

| File | Change |
|------|--------|
| `include/zoo/core/model.hpp` | Add `supports_tools()`, `set_tool_grammar()`, extend `GenerationResult` |
| `src/core/model.cpp` | Tool detection, lazy grammar sampler setup, streaming control in `run_inference()` |
| `include/zoo/tools/grammar.hpp` | **New** — `GrammarBuilder` for dynamic GBNF generation |
| `include/zoo/agent.hpp` | Updated tool loop, simplified parsing, new system prompt |
| `include/zoo/tools/parser.hpp` | Simplified to sentinel extraction (or kept as fallback) |
| `include/zoo/tools/validation.hpp` | Reduced role (grammar handles most validation) |

## Testing Strategy

### Unit Tests (pure logic, no model)

| Test | Validates |
|------|-----------|
| `test_grammar_builder.cpp` | GBNF generation from ToolRegistry schemas |
| `test_tool_support_detection.cpp` | Template string parsing for tool support |
| `test_sentinel_extraction.cpp` | Extracting tool call JSON from sentinel-wrapped text |

### Integration Tests (requires GGUF model)

| Test | Validates |
|------|-----------|
| Tool call with CoT | Model thinks, then calls tool correctly |
| Normal response streaming | Tokens stream to callback without delay |
| Tool call buffering | Tool call JSON is not streamed to user |
| Grammar constraint | Invalid tool names / argument types are rejected |
| Multi-turn tool loop | Tool result injection → re-generation works |

## Migration Path

The implementation can be done incrementally:

1. **Phase 1-2** can be built and tested independently (pure logic)
2. **Phase 3** requires integration testing with a real model
3. **Phase 4-5** build on Phase 3 and can be tested together
4. **Phase 6** is a system prompt change, testable immediately

The existing `ToolCallParser` and `ErrorRecovery` remain as fallbacks during migration.
Once grammar-based tool calling is stable, the heuristic parser can be simplified and
the error recovery reduced.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Model ignores sentinel instruction | Grammar trigger also supports native tokens (`<\|python_tag\|>`) as fallback |
| Grammar too restrictive for edge cases | Keep `ToolCallParser` as fallback path (config toggle) |
| Performance overhead of grammar sampler | Lazy grammar has zero overhead until triggered |
| Dynamic grammar generation bugs | Extensive unit tests on `GrammarBuilder` output |
| `llama_chat_apply_template` doesn't support sentinel tags | Use raw prompt injection or switch to Jinja-capable template rendering |

## Future Extensions

- **Parallel tool calls:** Grammar could be extended to allow arrays of tool calls
- **Streaming tool arguments:** For tools with large string args, stream the argument value
- **Grammar caching:** Cache compiled grammars when tool set doesn't change
- **Native template integration:** Use model's native tool calling format when available,
  falling back to sentinel for models without native support
