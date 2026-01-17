# Product Requirements Document: Zoo-Keeper

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Status** | Ready for Technical Prototype |
| **Product Owner** | [Product Owner Name] |
| **Last Updated** | [Date] |
| **Target Release** | [Q_ 20__] |

---

## 1. Executive Summary

### 1.1 Problem Statement

C++ application developers building local AI-powered software face significant challenges integrating Large Language Models (LLMs) into their applications. Existing inference libraries like llama.cpp provide excellent raw performance but require developers to manually implement conversation management, tool orchestration, and asynchronous processing. This creates substantial development overhead and introduces potential for threading bugs and inconsistent agent behavior.

### 1.2 Proposed Solution

Zoo-Keeper is a modern C++ wrapper library built on top of llama.cpp that functions as a complete **Agent Engine**. It abstracts the complexity of agentic AI systems by providing automated conversation history management, type-safe tool registration, asynchronous inference with cancellation support, and intelligent context window management—all through an ergonomic C++ API.

### 1.3 Value Proposition

- Reduce integration time for AI agents in C++ applications from weeks to days
- Eliminate common threading and context management bugs
- Enable non-ML engineers to build sophisticated AI agents
- Maximize inference efficiency through intelligent KV cache reuse

---

## 2. Goals and Success Metrics

### 2.1 Product Goals

| Goal ID | Goal | Priority |
|---------|------|----------|
| G-01 | Provide thread-safe async inference with <10ms library overhead | P0 |
| G-02 | Automate tool call detection, execution, and result injection | P0 |
| G-03 | Implement automatic context window management with FIFO pruning | P1 |
| G-04 | Support cross-platform deployment (Windows, macOS, Linux) | P1 |
| G-05 | Enable RAG context injection without history pollution | P2 |

### 2.2 Success Metrics

| Metric ID | Metric | Target | Measurement Method |
|-----------|--------|--------|-------------------|
| M-01 | Library call overhead | <10ms p99 | Benchmark suite |
| M-02 | Multi-turn latency reduction (KV reuse) | >40% vs cold start | Benchmark suite |
| M-03 | Tool call success rate | >95% first attempt | Integration tests |
| M-04 | Self-correction recovery rate | >80% within 2 retries | Integration tests |
| M-05 | API adoption friction | <30 min to first agent | User testing |

### 2.3 Non-Goals (Out of Scope)

- Cloud/remote inference support
- Model training or fine-tuning capabilities
- Built-in UI components
- Automatic model downloading or management

---

## 3. Target Users and Use Cases

### 3.1 Primary Persona: C++ Application Developer

**Profile:** Mid-to-senior level C++ developer building desktop applications, games, or embedded systems who wants to add intelligent AI capabilities without becoming an ML expert.

**Environment:** Local execution on edge devices, consumer desktops, or Apple Silicon Macs.

**Pain Points:**
- Unfamiliar with LLM prompt formats and tokenization
- Frustrated by manual conversation history management
- Concerned about thread safety with UI integration
- Needs deterministic tool calling behavior

### 3.2 Use Cases

| Use Case ID | Description | User Story |
|-------------|-------------|------------|
| UC-01 | Game NPC Intelligence | As a game developer, I want to give NPCs conversational AI so players can have natural dialogue that affects gameplay. |
| UC-02 | Desktop Assistant | As a productivity app developer, I want to embed a local AI assistant that can execute system commands based on user requests. |
| UC-03 | Code Analysis Tool | As a dev tools creator, I want to build an IDE plugin that analyzes code locally without sending data to external servers. |
| UC-04 | Document Processor | As an enterprise developer, I want to build a document analysis tool that processes sensitive files entirely on-device. |

---

## 4. Functional Requirements

### 4.1 Asynchronous Inference Manager

#### 4.1.1 Request Queue

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-101 | Thread-safe request queue | P0 | Queue operations are atomic; no data races under concurrent access |
| FR-102 | Request prioritization | P2 | Urgent requests can preempt queued requests |
| FR-103 | Queue depth monitoring | P1 | API exposes current queue depth for backpressure handling |

#### 4.1.2 Background Worker

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-104 | Dedicated inference thread | P0 | Single worker thread owns llama_context; main thread never touches it |
| FR-105 | Future-based results | P0 | `chat()` returns `std::future<Response>` immediately |
| FR-106 | Streaming callbacks | P1 | `on_token` callback invoked on inference thread for each generated token |

#### 4.1.3 Cancellation

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-107 | Immediate stop command | P0 | `stop()` aborts generation within 100ms |
| FR-108 | Graceful state recovery | P0 | After cancellation, next request processes normally |

### 4.2 Automated Context Management

#### 4.2.1 State Tracking

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-201 | Message buffer | P0 | Maintains ordered buffer of User, Assistant, System, and Tool messages |
| FR-202 | Role enforcement | P0 | Rejects invalid role sequences (e.g., consecutive User messages) |
| FR-203 | Token counting | P1 | Tracks token count per message for accurate window management |

#### 4.2.2 KV Cache Management

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-204 | Prefix reuse | P0 | Detects shared prefix and reuses KV cache entries |
| FR-205 | Cache invalidation | P1 | Correctly invalidates cache when history is modified |

#### 4.2.3 Context Pruning

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-206 | FIFO pruning strategy | P0 | Drops oldest User/Assistant pairs when window is full |
| FR-207 | System prompt preservation | P0 | System prompt is never pruned regardless of window state |
| FR-208 | Pruning notification | P1 | Callback notifies consumer when messages are pruned |

### 4.3 Tool Registry and Agentic Loop

#### 4.3.1 Tool Registration

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-301 | Template-based registration | P0 | `register_tool<Func>(name, description, func)` extracts parameter types automatically |
| FR-302 | Supported types | P0 | Supports `int`, `float`, `double`, `bool`, `std::string` parameters |
| FR-303 | Schema generation | P0 | Generates JSON schema from function signature |
| FR-304 | Optional parameters | P2 | Supports `std::optional<T>` for optional tool parameters |

#### 4.3.2 Orchestration Loop

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-305 | Tool call detection | P0 | Parses model output for tool call JSON |
| FR-306 | Automatic execution | P0 | Invokes registered function with parsed arguments |
| FR-307 | Result injection | P0 | Injects tool result as Tool message and continues generation |
| FR-308 | Multi-tool support | P1 | Handles multiple sequential tool calls in single turn |

### 4.4 RAG and Ephemeral Context

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-401 | Context injection | P1 | Accepts `RAG_CONTEXT` payload with user prompt |
| FR-402 | Priority positioning | P1 | Injected context appears immediately before user message |
| FR-403 | Ephemeral lifecycle | P0 | RAG context is not stored in long-term history buffer |

### 4.5 Model Template Mapping

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-501 | Auto-detection | P1 | Reads template from GGUF metadata when available |
| FR-502 | Manual override | P1 | Config accepts template enum (Llama3, Mistral, ChatML, etc.) |
| FR-503 | Runtime translation | P0 | Converts internal Message buffer to model-specific format |

### 4.6 Error Recovery

| Req ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| FR-601 | Schema validation | P0 | Validates tool call arguments against registered schema |
| FR-602 | Self-correction loop | P0 | On validation failure, injects error as System message |
| FR-603 | Retry limit | P0 | Maximum 2 retry attempts before returning error to caller |
| FR-604 | Error reporting | P1 | Final Response includes all attempted tool calls and errors |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| NFR ID | Requirement | Target | Priority |
|--------|-------------|--------|----------|
| NFR-01 | Library overhead per request | <10ms p99 | P0 |
| NFR-02 | Memory overhead beyond llama.cpp | <50MB | P1 |
| NFR-03 | Queue throughput | >1000 requests/sec enqueue | P2 |

### 5.2 Reliability

| NFR ID | Requirement | Target | Priority |
|--------|-------------|--------|----------|
| NFR-04 | Thread safety | Zero data races (verified by TSan) | P0 |
| NFR-05 | Crash recovery | No memory leaks on abnormal termination | P1 |

### 5.3 Compatibility

| NFR ID | Requirement | Target | Priority |
|--------|-------------|--------|----------|
| NFR-06 | macOS support | Metal acceleration default on Apple Silicon | P0 |
| NFR-07 | Windows support | MSVC 2019+ with CUDA optional | P0 |
| NFR-08 | Linux support | GCC 11+ with CUDA optional | P0 |
| NFR-09 | C++ standard | C++17 minimum | P0 |

---

## 6. Technical Architecture

### 6.1 Component Overview

Zoo-Keeper follows a layered architecture separating public API, core engine, and inference backend:

- **Public API Layer:** Type-safe C++ interface exposed to consumers
- **Engine Layer:** Manages conversation state, tool registry, and orchestration logic
- **Backend Layer:** Thin wrapper over llama.cpp handling model loading and inference

### 6.2 Threading Model

| Thread | Responsibilities |
|--------|------------------|
| **Calling Thread** | Submits `chat()` requests, receives `std::future<Response>`. |
| **Inference Thread** | Owns `llama_context`, processes queue, executes tools, manages history pruning |

**Callback Execution:** All callbacks (`on_token`, `on_tool_call`) execute on the inference thread. The consumer is responsible for any cross-thread synchronization required by their application (e.g., dispatching to an event loop, posting to a game engine's job system, or signaling a condition variable).

### 6.3 Dependencies

| Dependency | Purpose | Integration Method |
|------------|---------|-------------------|
| llama.cpp | LLM inference engine | Git submodule |
| nlohmann/json | JSON parsing and serialization | Header-only / CMake FetchContent |

### 6.4 Build System

CMake 3.20+ with the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `ZOO_ENABLE_METAL` | Enable Metal acceleration | ON (macOS) |
| `ZOO_ENABLE_CUDA` | Enable CUDA acceleration | OFF |
| `ZOO_BUILD_TESTS` | Build test suite | OFF |
| `ZOO_BUILD_EXAMPLES` | Build example applications | OFF |

---

## 7. Risks and Mitigations

| Risk ID | Risk | Impact | Likelihood | Mitigation |
|---------|------|--------|------------|------------|
| R-01 | llama.cpp API breaking changes | High | Medium | Pin to stable release; abstract internal usage |
| R-02 | Model-specific prompt format incompatibility | Medium | High | Extensive template library; user override option |
| R-03 | Tool call hallucinations exceed retry budget | Medium | Medium | Tune prompts; expose retry config; provide fallback |
| R-04 | Thread contention under heavy load | Medium | Low | Lock-free queue; benchmark-driven optimization |

---

## 8. Development Phases

### Phase 1: MVP (Weeks 1–4)

**Objective:** Functional async inference with basic conversation management

**Deliverables:**
- GGUF model loading via llama.cpp
- Thread-safe request queue and inference worker
- Basic message buffer (User, Assistant, System)
- `std::future`-based response delivery
- Manual template selection (Llama3, ChatML)

**Exit Criteria:** Demo application can load model, send prompts, and receive responses asynchronously.

### Phase 2: Tool System (Weeks 5–8)

**Objective:** Complete agentic loop with type-safe tools

**Deliverables:**
- Template-based tool registration with schema generation
- Tool call detection and parsing
- Automatic execution and result injection
- Schema validation and self-correction loop
- Multi-tool orchestration

**Exit Criteria:** Agent can register native functions and invoke them based on model output with >90% accuracy.

### Phase 3: Optimization (Weeks 9–12)

**Objective:** Production-ready performance and features

**Deliverables:**
- KV cache reuse for prefix matching
- FIFO context pruning with system prompt preservation
- RAG context injection (ephemeral)
- Auto-detection of GGUF templates
- Streaming token callbacks
- Comprehensive benchmark suite

**Exit Criteria:** Benchmark suite passes all NFR targets; documentation complete.

---

## 9. Open Questions

| ID | Question | Owner | Due Date |
|----|----------|-------|----------|
| OQ-01 | Should we support parallel tool execution for independent calls? | [TBD] | [TBD] |
| OQ-02 | What is the maximum supported context window size? | [TBD] | [TBD] |
| OQ-03 | Should we provide built-in common tools (file read, HTTP fetch)? | [TBD] | [TBD] |
| OQ-04 | How should we handle models without native tool call support? | [TBD] | [TBD] |

---

## 10. Appendix

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **Agentic Loop** | The cycle of inference, tool detection, tool execution, and result injection |
| **KV Cache** | Key-Value cache storing computed attention states for token reuse |
| **GGUF** | GPT-Generated Unified Format; binary format for LLM weights |
| **RAG** | Retrieval-Augmented Generation; injecting retrieved context into prompts |
| **Context Window** | Maximum number of tokens a model can process in one inference |

### 10.2 References

- llama.cpp repository: https://github.com/ggerganov/llama.cpp
- GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- nlohmann/json: https://github.com/nlohmann/json

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | [Date] | [Author] | Initial industry-standard PRD created from v1.0 semi-final draft |