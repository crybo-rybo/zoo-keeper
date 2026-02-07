# Technical Requirements Document: Zoo-Keeper

| Field | Value |
|-------|-------|
| **Version** | 2.0 |
| **Status** | Phase 2 Complete |
| **Technical Lead** | C.Rybacki |
| **Last Updated** | 2.7.2026 |
| **Related PRD** | Zoo-Keeper PRD v1.0 |

---

## 1. Introduction

### 1.1 Purpose

This Technical Requirements Document (TRD) specifies the software architecture, component design, state machines, and test plan for Zoo-Keeper—a header-only C++17 library that wraps llama.cpp to provide an ergonomic Agent Engine for local LLM inference.

### 1.2 Scope

This document covers:

- High-level architecture and component responsibilities
- State machines for async inference and agentic loop
- Threading model and synchronization strategy
- Test-Driven Development (TDD) methodology
- Comprehensive test plan with requirement traceability
- llama.cpp abstraction strategy for mocking

### 1.3 Design Principles

| Principle | Rationale |
|-----------|-----------|
| Header-only distribution | Simplifies integration; no linking complexity |
| Single `Agent` class entry point | Minimal API surface; easy to learn |
| `std::expected` for error handling | Modern, composable error handling without exceptions |
| Dependency injection for llama.cpp | Enables comprehensive unit testing via mocks |
| Value semantics where practical | Predictable ownership; fewer lifetime bugs |

### 1.4 Reference Documents

- Zoo-Keeper PRD v1.0
- llama.cpp API documentation
- C++17 Standard Library reference

---

## 2. System Architecture

### 2.1 Architectural Overview

Zoo-Keeper follows a layered architecture with clear separation between the public API, core engine logic, and inference backend. This separation enables unit testing of engine components without requiring actual model inference.

#### Layer Structure

| Layer | Components | Depends On |
|-------|------------|------------|
| **Consumer Code** | Application using Zoo-Keeper | Public API Layer |
| **Public API Layer** | `zoo::Agent` (single entry point for all library functionality) | Engine Layer |
| **Engine Layer** | Request Queue, History Manager, Tool Registry, Agentic Loop, Template Engine, Error Recovery | Backend Layer |
| **Backend Layer** | Inference Backend Interface, Llama Backend (production), Mock Backend (testing) | llama.cpp (production only) |

#### Data Flow

Consumer Code → `zoo::Agent` → Engine Components → Backend Interface → llama.cpp (or Mock)

### 2.2 Layer Responsibilities

| Layer | Responsibility | Dependencies |
|-------|----------------|--------------|
| **Public API** | Exposes `Agent` class; defines public types (`Config`, `Message`, `Response`); manages inference thread lifecycle | Engine Layer |
| **Engine** | Implements core logic: request queuing, history management, tool orchestration, template formatting, error recovery | Backend Layer |
| **Backend** | Abstracts llama.cpp behind an interface; provides mock implementation for testing | llama.cpp (production only) |

---

## 3. Component Specifications

### 3.1 Public API Layer

#### 3.1.1 Agent

| Aspect | Description |
|--------|-------------|
| **Purpose** | Single entry point for all library functionality; owns and coordinates all internal components |
| **Responsibilities** | Construct and configure the inference pipeline; spawn and manage the inference thread; accept chat requests and return futures; provide tool registration interface; handle stop/cancellation requests |
| **Thread Ownership** | Owns the inference thread; all public methods are safe to call from the calling thread |
| **Lifetime** | Created via factory method; move-only semantics; destructor joins inference thread |
| **Error Strategy** | Factory returns `std::expected<Agent, Error>`; methods return `std::expected` or `std::future<std::expected>` |

#### 3.1.2 Config

| Aspect | Description |
|--------|-------------|
| **Purpose** | Immutable configuration bundle for Agent construction |
| **Responsibilities** | Store model path, context size, sampling parameters, template selection, threading options, and hardware acceleration settings |
| **Design** | Plain struct with sensible defaults; builder-pattern methods for ergonomic construction |
| **Validation** | Validated during Agent construction; invalid configs produce `Error` |

#### 3.1.3 Message

| Aspect | Description |
|--------|-------------|
| **Purpose** | Value type representing a single conversation message |
| **Responsibilities** | Store role (System/User/Assistant/Tool), content, and optional tool metadata |
| **Design** | Immutable value type; factory methods for each role type |

#### 3.1.4 Response

| Aspect | Description |
|--------|-------------|
| **Purpose** | Result of a completed chat request |
| **Responsibilities** | Contain generated text, tool call/result history, token usage statistics, latency metrics, and any error recovery information |
| **Design** | Immutable value type populated by the agentic loop |

### 3.2 Engine Layer

#### 3.2.1 Request Queue

| Aspect | Description |
|--------|-------------|
| **Purpose** | Thread-safe queue for chat requests from calling thread to inference thread |
| **Responsibilities** | Accept new requests atomically; provide blocking dequeue for inference thread; support cancellation signaling; expose queue depth for monitoring |
| **Thread Safety** | Multiple producers (calling threads), single consumer (inference thread); lock-free or mutex-based MPSC design |
| **Backpressure** | Exposes queue depth; does not enforce limits (consumer responsibility) |

#### 3.2.2 History Manager

| Aspect | Description |
|--------|-------------|
| **Purpose** | Maintains conversation history and manages context window |
| **Responsibilities** | Store ordered sequence of Messages; track token count per message and total; enforce valid role sequences (e.g., no consecutive User messages); implement FIFO pruning when context window is exceeded; preserve System prompt during pruning; notify via callback when messages are pruned |
| **KV Cache Coordination** | Track which messages are "committed" to KV cache; identify shared prefix for cache reuse |
| **RAG Handling** | Accept ephemeral context that is used for current turn but not persisted |

#### 3.2.3 Tool Registry

| Aspect | Description |
|--------|-------------|
| **Purpose** | Stores registered tools and handles invocation |
| **Responsibilities** | Register tools with name, description, and handler; generate JSON schema from C++ function signatures (for supported types: `int`, `float`, `double`, `bool`, `std::string`); store explicit JSON schemas for manually registered tools; look up tools by name; invoke handlers with parsed JSON arguments; return JSON results |
| **Supported Types** | Primitive types via template reflection; complex types via explicit schema registration |
| **Uniqueness** | Tool names must be unique; re-registration overwrites |

#### 3.2.4 Agentic Loop

| Aspect | Description |
|--------|-------------|
| **Purpose** | Orchestrates the inference → tool detection → execution → continuation cycle |
| **Responsibilities** | Coordinate with Template Engine to format prompts; invoke Backend for inference; parse model output for tool calls; delegate to Tool Registry for execution; inject tool results back into context; loop until model produces final response (no tool calls); delegate to Error Recovery on validation failures |
| **Termination** | Loop exits when model output contains no tool calls, or max iterations reached, or cancellation requested |

#### 3.2.5 Template Engine

| Aspect | Description |
|--------|-------------|
| **Purpose** | Converts internal Message buffer to model-specific prompt format |
| **Responsibilities** | Support multiple prompt templates (Llama3, Llama2, Mistral, ChatML, Phi3, Gemma, Raw); auto-detect template from GGUF metadata when configured; format tool schemas into system prompt; format tool results into appropriate message format |
| **Extensibility** | Template implementations are internal; no public extension point in v1.0 |

#### 3.2.6 Error Recovery

| Aspect | Description |
|--------|-------------|
| **Purpose** | Handles tool call validation failures and manages retry logic |
| **Responsibilities** | Validate tool call arguments against registered schema; on validation failure, construct error message as System message; track retry count per tool call; enforce maximum retry limit (default: 2); collect all errors for inclusion in final Response |
| **Failure Modes** | Tool not found, argument type mismatch, missing required argument, handler exception |

### 3.3 Backend Layer

#### 3.3.1 Inference Backend Interface

| Aspect | Description |
|--------|-------------|
| **Purpose** | Abstract interface decoupling engine from llama.cpp |
| **Responsibilities** | Define operations: load model, tokenize, evaluate prompt, sample tokens, manage KV cache, get model metadata |
| **Design** | Pure virtual interface; implementations injected into Agent |

#### 3.3.2 Llama Backend

| Aspect | Description |
|--------|-------------|
| **Purpose** | Production implementation wrapping llama.cpp |
| **Responsibilities** | Implement all interface methods using llama.cpp API; manage `llama_model` and `llama_context` lifetime; handle GPU/Metal acceleration; implement KV cache shifting for context pruning |
| **Thread Safety** | All methods called exclusively from inference thread |

#### 3.3.3 Mock Backend

| Aspect | Description |
|--------|-------------|
| **Purpose** | Test implementation for unit testing without real models |
| **Responsibilities** | Implement interface with configurable responses; support scripted token sequences; simulate tool call outputs; simulate errors and edge cases; track method calls for verification |
| **Test Support** | Configurable latency, token-by-token streaming simulation, error injection |

---

## 4. State Machines

### 4.1 Agent Lifecycle State Machine

#### States

| State | Description |
|-------|-------------|
| **Uninitialized** | Initial state before `create()` is called |
| **Loading** | Model is being loaded; inference thread starting |
| **Failed** | Model load failed; Agent cannot be used (terminal) |
| **Idle** | Ready to accept chat requests; inference thread waiting |
| **Processing** | Actively processing a chat request |
| **Cancelling** | Stop requested; waiting for current operation to abort |
| **Terminated** | Destructor called; inference thread joined; resources released (terminal) |

#### State Transitions

| From State | Event/Trigger | To State | Notes |
|------------|---------------|----------|-------|
| Uninitialized | `create(config)` called | Loading | Factory method invoked |
| Loading | Model load succeeds | Idle | Inference thread started |
| Loading | Model load fails | Failed | Error returned to caller |
| Idle | `chat()` called | Processing | Request enqueued |
| Processing | Request completes | Idle | Future resolved with Response |
| Processing | `stop()` called | Cancelling | Cancellation flag set |
| Cancelling | Cancellation completes | Idle | Future resolved with error |
| Any state | Destructor called | Terminated | Inference thread joined |

### 4.2 Agentic Loop State Machine

#### States

| State | Description |
|-------|-------------|
| **Await Request** | Inference thread blocked on queue; waiting for work |
| **Format Prompt** | Template Engine converting messages to model-specific format |
| **Inference** | Backend generating tokens; streaming callbacks firing |
| **Cancelled** | Stop signal received; aborting current generation |
| **Parse Output** | Analyzing generated text for tool calls |
| **Validate Args** | Checking tool call arguments against schema |
| **Execute Tool** | Invoking registered tool handler |
| **Error Recovery** | Constructing error message for self-correction |
| **Inject Result** | Adding tool result to context; preparing for next inference loop |
| **Build Response** | Assembling final Response object with all metadata |

#### State Transitions

| From State | Event/Trigger | To State | Notes |
|------------|---------------|----------|-------|
| Await Request | Request dequeued | Format Prompt | New chat request received |
| Format Prompt | Prompt ready | Inference | Template Engine completed formatting |
| Inference | `stop()` called | Cancelled | Cancellation requested |
| Inference | Generation complete | Parse Output | All tokens generated |
| Cancelled | — | Build Response | Future resolved with cancellation error |
| Parse Output | No tool calls found | Build Response | Direct response path |
| Parse Output | Tool call detected | Validate Args | Tool call JSON parsed |
| Parse Output | Parse error | Build Response | Malformed output handling |
| Validate Args | Arguments valid | Execute Tool | Schema validation passed |
| Validate Args | Arguments invalid | Error Recovery | Schema validation failed |
| Execute Tool | Tool returns result | Inject Result | Handler completed |
| Error Recovery | Retries remaining | Inject Result | Error message constructed for retry |
| Error Recovery | Retries exhausted | Build Response | Max retries exceeded |
| Inject Result | — | Inference | Loop back for continued generation |
| Build Response | — | Await Request | Future resolved; ready for next request |

### 4.3 History Manager State Machine

#### States

| State | Description |
|-------|-------------|
| **Empty** | No messages in history; initial state |
| **System Only** | Only system prompt present; awaiting first user message |
| **Accumulating** | Normal operation; messages being added within context window |
| **Pruning** | Context window exceeded; removing oldest User/Assistant pairs |

#### State Transitions

| From State | Event/Trigger | To State | Notes |
|------------|---------------|----------|-------|
| Empty | `set_system_prompt()` called | System Only | System prompt stored |
| System Only | User message added | Accumulating | Conversation started |
| System Only | System prompt updated | System Only | Prompt replaced |
| Accumulating | Message added (within window) | Accumulating | Normal message accumulation |
| Accumulating | Context window exceeded | Pruning | Token limit reached |
| Accumulating | `clear_history()` called | System Only | History cleared; system prompt preserved |
| Accumulating | System prompt updated | System Only | Resets to system-only state |
| Pruning | Oldest pairs removed | Accumulating | System prompt preserved; prune callback fired |
| Pruning | System prompt updated | System Only | Resets to system-only state |

---

## 5. Threading Model

### 5.1 Thread Responsibilities

| Thread | Identity | Responsibilities |
|--------|----------|------------------|
| **Calling Thread** | Any thread invoking Agent methods | Submit chat requests; register tools; set system prompt; request cancellation; receive futures |
| **Inference Thread** | Dedicated thread owned by Agent | Process request queue; execute inference; run tool handlers; manage history; fire callbacks |

### 5.2 Thread Interaction Sequence

The following table describes the sequence of interactions between the Calling Thread and Inference Thread during a typical chat request:

| Step | Calling Thread | Direction | Inference Thread | Notes |
|------|----------------|-----------|------------------|-------|
| 1 | `chat(message)` | → | | Request initiated |
| 2 | [enqueue request] | → | | Request added to queue |
| 3 | receives `std::future` | ← | | Returns immediately |
| 4 | | | [dequeue request] | Worker picks up request |
| 5 | | | [format prompt] | Template Engine runs |
| 6 | | | [run inference] | Token generation begins |
| 7 | [on_token callback] | ← | | Streaming callback (on inference thread) |
| 8 | | | [detect tool call] | Parse output for tools |
| 9 | | | [execute tool] | Tool handler invoked |
| 10 | | | [continue inference] | Resume if more generation needed |
| 11 | | | [build response] | Assemble final Response |
| 12 | [future resolved] | ← | | Response available |

**Cancellation Sequence:**

| Step | Calling Thread | Direction | Inference Thread | Notes |
|------|----------------|-----------|------------------|-------|
| 1 | `stop()` | → | | Cancellation requested |
| 2 | [set atomic flag] | → | | Flag visible to inference thread |
| 3 | | | [check flag] | Detected at next check point |
| 4 | | | [abort generation] | Cleanup and exit loop |
| 5 | [future resolved with error] | ← | | Cancellation error returned |

### 5.3 Synchronization Points

| Resource | Access Pattern | Synchronization Mechanism |
|----------|----------------|---------------------------|
| Request Queue | Multi-producer (calling), single-consumer (inference) | Lock-free queue or mutex + condition variable |
| Cancellation Flag | Write (calling), read (inference) | `std::atomic<bool>` |
| System Prompt | Write (calling), read (inference) | Mutex-protected; copied on read |
| Tool Registry | Write (calling during setup), read (inference) | Mutex-protected; registration typically done before first chat |
| History Buffer | Read/write (inference only) | No synchronization needed; single-threaded access |

### 5.4 Callback Execution Context

All callbacks execute on the **inference thread**. This design:

- Avoids context-switching overhead during streaming
- Provides predictable execution order
- Places responsibility on consumer for cross-thread dispatch

Consumers requiring updates on other threads must implement their own dispatch mechanism (e.g., posting to an event loop, signaling a condition variable, or using a thread-safe queue).

---

## 6. Error Handling Strategy

### 6.1 Error Type Design

The library uses `std::expected<T, Error>` for all fallible operations:

| Category | Error Codes | Recovery |
|----------|-------------|----------|
| **Initialization** | `ModelLoadFailed`, `InvalidConfig` | Caller must handle; Agent not usable |
| **Runtime** | `InferenceAborted`, `ContextOverflow` | Request fails; Agent remains usable for subsequent requests |
| **Tool** | `ToolNotFound`, `ToolValidationFailed`, `ToolRetriesExhausted` | Self-correction attempted; ultimately reported in Response |
| **State** | `InvalidMessageSequence`, `AgentNotRunning` | Caller error; request rejected |

### 6.2 Error Propagation

```
Backend Error
     │
     ▼
Engine Layer (catches, wraps in Error)
     │
     ▼
Agent (packages into std::expected)
     │
     ▼
Future resolved with error
     │
     ▼
Caller handles via .value_or(), .error(), or .has_value()
```

### 6.3 Self-Correction Flow

When tool validation fails:

1. Error Recovery component constructs descriptive error message
2. Error injected as System message into context
3. Inference continues (retry attempt)
4. Retry count incremented
5. If retries exhausted, error included in final Response
6. Consumer can inspect `Response::tool_errors` for details

---

## 7. Test-Driven Development Strategy

### 7.1 TDD Workflow

The project follows the standard Red-Green-Refactor cycle:

| Phase | Action | Exit Criteria |
|-------|--------|---------------|
| **Red** | Write a failing test for the new requirement | Test compiles but fails |
| **Green** | Write minimal code to pass the test | Test passes |
| **Refactor** | Improve code structure while keeping tests green | All tests still pass; code meets quality standards |

After Refactor, the cycle repeats for the next requirement. This ensures that every feature has test coverage from inception and that the codebase remains maintainable.

### 7.2 Test Framework

| Tool | Purpose |
|------|---------|
| **GoogleTest** | Primary test framework; test fixtures, assertions, parameterized tests |
| **GoogleMock** | Mock objects for Backend interface; expectation setting; call verification |

### 7.3 Mock Backend Capabilities

The Mock Backend must support:

| Capability | Purpose |
|------------|---------|
| **Scripted Responses** | Configure exact token sequences to return |
| **Tool Call Simulation** | Return outputs that parse as tool calls |
| **Error Injection** | Simulate backend failures |
| **Latency Simulation** | Test timeout and cancellation behavior |
| **Call Verification** | Assert methods called with expected arguments |
| **Token Streaming** | Fire token callbacks with controlled timing |

### 7.4 Test Organization

| Directory | File | Purpose |
|-----------|------|---------|
| `tests/unit/` | `test_request_queue.cpp` | Request Queue unit tests |
| | `test_history_manager.cpp` | History Manager unit tests |
| | `test_tool_registry.cpp` | Tool Registry unit tests |
| | `test_agentic_loop.cpp` | Agentic Loop unit tests |
| | `test_error_recovery.cpp` | Error Recovery unit tests |
| | `test_agent.cpp` | Agent integration tests (with mock backend) |
| `tests/mocks/` | `mock_backend.hpp` | Mock Backend interface declaration |
| | `mock_backend.cpp` | Mock Backend implementation |
| `tests/fixtures/` | `sample_responses.hpp` | Scripted model responses and tool call JSON |
| | `tool_definitions.hpp` | Sample tool definitions for testing |
| `tests/` | `CMakeLists.txt` | Test build configuration |

---

## 8. Test Plan

### 8.1 Test Categories

| Category | Scope | Mock Usage |
|----------|-------|------------|
| **Unit Tests** | Individual components in isolation | Full mocking of dependencies |
| **Component Tests** | Multiple components working together | Mock Backend only |
| **Contract Tests** | Verify Mock Backend matches real Backend | Side-by-side comparison (future) |
| **Integration Tests** | Full system with real models | No mocks (out of scope for TRD) |

### 8.2 Requirement Traceability Matrix

#### 8.2.1 Request Queue Tests

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| TQ-001 | FR-101 | Enqueue from multiple threads simultaneously | All requests queued; no data loss or corruption |
| TQ-002 | FR-101 | Dequeue from single consumer thread | Requests retrieved in FIFO order |
| TQ-003 | FR-101 | Concurrent enqueue and dequeue | No deadlock; no data races (TSan clean) |
| TQ-004 | FR-103 | Query queue depth during operations | Depth accurately reflects pending requests |
| TQ-005 | FR-103 | Queue depth zero when empty | Returns 0; does not block |

#### 8.2.2 History Manager Tests

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| TH-001 | FR-201 | Add messages of all role types | Messages stored in order; roles preserved |
| TH-002 | FR-202 | Add consecutive User messages | Second message rejected with error |
| TH-003 | FR-202 | Add Assistant after User | Accepted; valid sequence |
| TH-004 | FR-202 | Add Tool message without preceding Assistant tool call | Rejected with error |
| TH-005 | FR-203 | Token count tracked per message | Counts available via API; sum matches total |
| TH-006 | FR-206 | Exceed context window | Oldest User/Assistant pair pruned |
| TH-007 | FR-206 | Exceed context window by large amount | Multiple pairs pruned until within limit |
| TH-008 | FR-207 | Prune with system prompt | System prompt preserved; only User/Assistant pruned |
| TH-009 | FR-208 | Prune callback | Callback invoked with pruned messages |
| TH-010 | FR-403 | Add RAG context | Context used in current turn; not in history after |

#### 8.2.3 Tool Registry Tests

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| TR-001 | FR-301 | Register tool with primitive parameters | Schema generated correctly |
| TR-002 | FR-302 | Register tool with int parameter | Schema shows integer type |
| TR-003 | FR-302 | Register tool with float parameter | Schema shows number type |
| TR-004 | FR-302 | Register tool with bool parameter | Schema shows boolean type |
| TR-005 | FR-302 | Register tool with string parameter | Schema shows string type |
| TR-006 | FR-302 | Register tool with multiple parameters | Schema shows all parameters |
| TR-007 | FR-303 | Generate schema for registered tool | Valid JSON schema produced |
| TR-008 | FR-303 | Schema includes description | Description field populated |
| TR-009 | — | Invoke registered tool with valid args | Handler called; result returned |
| TR-010 | — | Invoke tool with wrong argument types | Error returned; handler not called |
| TR-011 | — | Invoke unregistered tool | ToolNotFound error returned |
| TR-012 | — | Re-register tool with same name | Previous registration overwritten |

#### 8.2.4 Agentic Loop Tests

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| TA-001 | FR-105 | Submit chat request | Future returned immediately |
| TA-002 | FR-105 | Await future resolution | Response received with content |
| TA-003 | FR-305 | Model output contains tool call JSON | Tool call detected and parsed |
| TA-004 | FR-305 | Model output contains no tool call | Response returned directly |
| TA-005 | FR-306 | Tool call detected | Registered handler invoked |
| TA-006 | FR-307 | Tool executed successfully | Result injected; inference continues |
| TA-007 | FR-308 | Model requests multiple tools sequentially | All tools executed in order |
| TA-008 | FR-107 | Stop called during inference | Inference aborted within 100ms |
| TA-009 | FR-108 | Request after cancellation | New request processes normally |
| TA-010 | FR-106 | Streaming callback registered | Callback invoked for each token |

#### 8.2.5 Template Engine Tests

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| TE-001 | FR-503 | Format messages for Llama3 | Correct Llama3 format produced |
| TE-002 | FR-503 | Format messages for ChatML | Correct ChatML format produced |
| TE-003 | FR-503 | Format messages for Mistral | Correct Mistral format produced |
| TE-004 | FR-502 | Override template in config | Specified template used regardless of metadata |
| TE-005 | FR-501 | Auto-detect from GGUF metadata | Correct template selected |
| TE-006 | — | Format tool schemas in system prompt | Tools described in system prompt |
| TE-007 | — | Format tool results | Tool results formatted per template spec |

#### 8.2.6 Error Recovery Tests

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| ER-001 | FR-601 | Tool call with valid arguments | Validation passes; tool executed |
| ER-002 | FR-601 | Tool call missing required argument | Validation fails; error returned |
| ER-003 | FR-601 | Tool call with wrong argument type | Validation fails; error returned |
| ER-004 | FR-602 | Validation failure | Error message injected as System message |
| ER-005 | FR-602 | Self-correction succeeds on retry | Tool executes on second attempt |
| ER-006 | FR-603 | Two retries fail | ToolRetriesExhausted error in Response |
| ER-007 | FR-604 | Multiple tool errors | All errors included in Response |

#### 8.2.7 Agent Tests (Component Level with Mock Backend)

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| AG-001 | G-01 | Measure library overhead | Overhead < 10ms p99 |
| AG-002 | FR-104 | Verify inference on dedicated thread | Backend methods called only from inference thread |
| AG-003 | FR-204 | Send multi-turn conversation | KV cache reuse signaled to backend |
| AG-004 | FR-205 | Modify history mid-conversation | Cache invalidation signaled |
| AG-005 | — | Create Agent with invalid config | Error returned; no crash |
| AG-006 | — | Create Agent with valid config | Agent ready; Idle state |
| AG-007 | — | Destroy Agent during processing | Clean shutdown; no resource leaks |
| AG-008 | — | Register tool after first chat | Tool available for subsequent chats |

### 8.3 Test Data Requirements

| Data Type | Description | Location |
|-----------|-------------|----------|
| **Scripted Responses** | Pre-defined model outputs for various scenarios | `tests/fixtures/sample_responses.hpp` |
| **Tool Definitions** | Sample tools with various parameter combinations | `tests/fixtures/tool_definitions.hpp` |
| **Tool Call JSON** | Valid and invalid tool call formats | `tests/fixtures/sample_responses.hpp` |
| **Template Outputs** | Expected formatted prompts for each template | `tests/fixtures/template_expectations.hpp` |

### 8.4 Test Coverage Goals

| Component | Line Coverage Target | Branch Coverage Target |
|-----------|---------------------|------------------------|
| Request Queue | 95% | 90% |
| History Manager | 95% | 90% |
| Tool Registry | 95% | 90% |
| Agentic Loop | 90% | 85% |
| Template Engine | 95% | 90% |
| Error Recovery | 95% | 90% |
| Agent | 85% | 80% |

---

## 9. Build and Development Environment

### 9.1 Build Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `ZOO_ENABLE_METAL` | Enable Metal acceleration (macOS) | ON (macOS) |
| `ZOO_ENABLE_CUDA` | Enable CUDA acceleration | OFF |
| `ZOO_BUILD_TESTS` | Build unit test suite | OFF |
| `ZOO_BUILD_EXAMPLES` | Build example applications | OFF |
| `ZOO_ENABLE_COVERAGE` | Enable code coverage instrumentation | OFF |
| `ZOO_ENABLE_SANITIZERS` | Enable ASan/TSan/UBSan | OFF |

### 9.2 Compiler Requirements

| Platform | Compiler | Minimum Version |
|----------|----------|-----------------|
| Windows | MSVC | 2019 (19.20) |
| macOS | Clang | 13.0 |
| Linux | GCC | 11.0 |
| Linux | Clang | 13.0 |

### 9.3 Dependencies

| Dependency | Version | Integration |
|------------|---------|-------------|
| llama.cpp | [pinned commit] | Git submodule |
| nlohmann/json | 3.11+ | CMake FetchContent |
| GoogleTest | 1.14+ | CMake FetchContent (tests only) |

### 9.4 CI Pipeline Stages

| Stage | Actions |
|-------|---------|
| **Build** | Compile on all supported platforms and compilers |
| **Test** | Run unit tests; fail on any test failure |
| **Sanitize** | Run tests under ASan, TSan, UBSan |
| **Coverage** | Generate coverage report; fail if below threshold |
| **Lint** | Run clang-tidy and clang-format checks |

---

## 10. Open Technical Questions

| ID | Question | Impact | Owner | Due Date |
|----|----------|--------|-------|----------|
| TQ-01 | Lock-free queue vs mutex-based queue? | Performance, complexity | [TBD] | [TBD] |
| TQ-02 | Should Template Engine be extensible for custom templates? | API surface, maintenance | [TBD] | [TBD] |
| TQ-03 | How to handle GGUF files without template metadata? | User experience | [TBD] | [TBD] |
| TQ-04 | Should tool handlers be allowed to be async? | API complexity, use cases | [TBD] | [TBD] |
| TQ-05 | Maximum number of agentic loop iterations before forced termination? | Safety, resource usage | [TBD] | [TBD] |

---

## 11. Glossary

| Term | Definition |
|------|------------|
| **Calling Thread** | Any thread that invokes public Agent methods |
| **Inference Thread** | Dedicated thread owned by Agent that processes requests |
| **MPSC** | Multi-Producer Single-Consumer queue pattern |
| **KV Cache** | Key-Value cache storing attention states for token reuse |
| **TDD** | Test-Driven Development; write tests before implementation |
| **TSan** | ThreadSanitizer; detects data races |
| **ASan** | AddressSanitizer; detects memory errors |
| **UBSan** | UndefinedBehaviorSanitizer; detects undefined behavior |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 1.17.2025 | C.Rybacki | Initial TRD created |