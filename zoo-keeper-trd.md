# Technical Requirements Document: Zoo-Keeper

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Status** | Draft |
| **Technical Lead** | [Technical Lead Name] |
| **Last Updated** | [Date] |
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

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Consumer Code                              │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Public API Layer                               │
│                                                                         │
│                            zoo::Agent                                   │
│         (Single entry point for all library functionality)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Engine Layer                                  │
│                                                                         │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│   │   Request   │  │   History   │  │    Tool     │  │   Agentic   │   │
│   │    Queue    │  │   Manager   │  │  Registry   │  │    Loop     │   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
│   ┌─────────────┐  ┌─────────────┐                                     │
│   │  Template   │  │    Error    │                                     │
│   │   Engine    │  │  Recovery   │                                     │
│   └─────────────┘  └─────────────┘                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Backend Layer                                  │
│                                                                         │
│                  ┌─────────────────────────────┐                        │
│                  │  Inference Backend Interface │                        │
│                  └─────────────────────────────┘                        │
│                         ▲               ▲                               │
│                         │               │                               │
│              ┌──────────┴───┐     ┌─────┴──────────┐                    │
│              │ Llama Backend │     │  Mock Backend  │                    │
│              │  (Production) │     │   (Testing)    │                    │
│              └──────────────┘     └────────────────┘                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

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

<svg viewBox="0 0 600 650" xmlns="http://www.w3.org/2000/svg" style="max-width: 600px; font-family: Arial, sans-serif;">
  <defs>
    <marker id="arrowhead1" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  
  <!-- Uninitialized -->
  <rect x="220" y="20" width="140" height="50" rx="8" fill="#e8f4f8" stroke="#333" stroke-width="2"/>
  <text x="290" y="50" text-anchor="middle" font-size="14" font-weight="bold">Uninitialized</text>
  
  <!-- Arrow: Uninitialized -> Loading -->
  <line x1="290" y1="70" x2="290" y2="110" stroke="#333" stroke-width="2" marker-end="url(#arrowhead1)"/>
  <text x="300" y="95" font-size="11" fill="#666">create(config)</text>
  
  <!-- Loading -->
  <rect x="220" y="115" width="140" height="50" rx="8" fill="#fff3cd" stroke="#333" stroke-width="2"/>
  <text x="290" y="145" text-anchor="middle" font-size="14" font-weight="bold">Loading</text>
  
  <!-- Arrow: Loading -> Failed -->
  <line x1="220" y1="140" x2="130" y2="140" stroke="#333" stroke-width="2"/>
  <line x1="130" y1="140" x2="130" y2="210" stroke="#333" stroke-width="2" marker-end="url(#arrowhead1)"/>
  <text x="140" y="180" font-size="11" fill="#666">load failed</text>
  
  <!-- Arrow: Loading -> Idle -->
  <line x1="290" y1="165" x2="290" y2="210" stroke="#333" stroke-width="2" marker-end="url(#arrowhead1)"/>
  <text x="300" y="192" font-size="11" fill="#666">load success</text>
  
  <!-- Failed -->
  <rect x="60" y="215" width="140" height="50" rx="8" fill="#f8d7da" stroke="#333" stroke-width="2"/>
  <text x="130" y="245" text-anchor="middle" font-size="14" font-weight="bold">Failed</text>
  
  <!-- Idle -->
  <rect x="220" y="215" width="140" height="50" rx="8" fill="#d4edda" stroke="#333" stroke-width="2"/>
  <text x="290" y="245" text-anchor="middle" font-size="14" font-weight="bold">Idle</text>
  
  <!-- Arrow: Idle -> Processing -->
  <line x1="290" y1="265" x2="290" y2="310" stroke="#333" stroke-width="2" marker-end="url(#arrowhead1)"/>
  <text x="300" y="292" font-size="11" fill="#666">chat() called</text>
  
  <!-- Processing -->
  <rect x="220" y="315" width="140" height="50" rx="8" fill="#cce5ff" stroke="#333" stroke-width="2"/>
  <text x="290" y="345" text-anchor="middle" font-size="14" font-weight="bold">Processing</text>
  
  <!-- Arrow: Processing -> Idle (request complete) -->
  <line x1="360" y1="340" x2="480" y2="340" stroke="#333" stroke-width="2"/>
  <line x1="480" y1="340" x2="480" y2="240" stroke="#333" stroke-width="2"/>
  <line x1="480" y1="240" x2="360" y2="240" stroke="#333" stroke-width="2" marker-end="url(#arrowhead1)"/>
  <text x="490" y="295" font-size="11" fill="#666">request</text>
  <text x="490" y="308" font-size="11" fill="#666">complete</text>
  
  <!-- Arrow: Processing -> Cancelling -->
  <line x1="290" y1="365" x2="290" y2="410" stroke="#333" stroke-width="2" marker-end="url(#arrowhead1)"/>
  <text x="300" y="392" font-size="11" fill="#666">stop() called</text>
  
  <!-- Cancelling -->
  <rect x="220" y="415" width="140" height="50" rx="8" fill="#fff3cd" stroke="#333" stroke-width="2"/>
  <text x="290" y="445" text-anchor="middle" font-size="14" font-weight="bold">Cancelling</text>
  
  <!-- Arrow: Cancelling -> Idle -->
  <line x1="360" y1="440" x2="520" y2="440" stroke="#333" stroke-width="2"/>
  <line x1="520" y1="440" x2="520" y2="240" stroke="#333" stroke-width="2"/>
  <line x1="520" y1="240" x2="480" y2="240" stroke="#333" stroke-width="2"/>
  <text x="530" y="350" font-size="11" fill="#666">cancellation</text>
  <text x="530" y="363" font-size="11" fill="#666">complete</text>
  
  <!-- Terminated -->
  <rect x="220" y="530" width="140" height="50" rx="8" fill="#e2e3e5" stroke="#333" stroke-width="2"/>
  <text x="290" y="560" text-anchor="middle" font-size="14" font-weight="bold">Terminated</text>
  
  <!-- Arrow: destructor -> Terminated -->
  <line x1="130" y1="555" x2="215" y2="555" stroke="#333" stroke-width="2" marker-end="url(#arrowhead1)"/>
  <text x="60" y="550" font-size="11" fill="#666">destructor</text>
  <text x="60" y="563" font-size="11" fill="#666">(from any)</text>
</svg>

**State Descriptions:**

| State | Description |
|-------|-------------|
| **Uninitialized** | Initial state before `create()` is called |
| **Loading** | Model is being loaded; inference thread starting |
| **Failed** | Model load failed; Agent cannot be used |
| **Idle** | Ready to accept chat requests; inference thread waiting |
| **Processing** | Actively processing a chat request |
| **Cancelling** | Stop requested; waiting for current operation to abort |
| **Terminated** | Destructor called; inference thread joined; resources released |

### 4.2 Agentic Loop State Machine

<svg viewBox="0 0 750 850" xmlns="http://www.w3.org/2000/svg" style="max-width: 750px; font-family: Arial, sans-serif;">
  <defs>
    <marker id="arrowhead2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  
  <!-- Await Request -->
  <rect x="280" y="20" width="150" height="50" rx="8" fill="#d4edda" stroke="#333" stroke-width="2"/>
  <text x="355" y="50" text-anchor="middle" font-size="13" font-weight="bold">Await Request</text>
  
  <!-- Arrow: Await Request -> Format Prompt -->
  <line x1="355" y1="70" x2="355" y2="110" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <text x="365" y="95" font-size="10" fill="#666">request dequeued</text>
  
  <!-- Format Prompt -->
  <rect x="280" y="115" width="150" height="50" rx="8" fill="#e8f4f8" stroke="#333" stroke-width="2"/>
  <text x="355" y="145" text-anchor="middle" font-size="13" font-weight="bold">Format Prompt</text>
  
  <!-- Arrow: Format Prompt -> Inference -->
  <line x1="355" y1="165" x2="355" y2="205" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <text x="365" y="190" font-size="10" fill="#666">prompt ready</text>
  
  <!-- Inference -->
  <rect x="280" y="210" width="150" height="50" rx="8" fill="#cce5ff" stroke="#333" stroke-width="2"/>
  <text x="355" y="240" text-anchor="middle" font-size="13" font-weight="bold">Inference</text>
  
  <!-- Arrow: Inference -> Cancelled -->
  <line x1="430" y1="235" x2="530" y2="235" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <text x="460" y="225" font-size="10" fill="#666">stop()</text>
  
  <!-- Cancelled -->
  <rect x="535" y="210" width="120" height="50" rx="8" fill="#f8d7da" stroke="#333" stroke-width="2"/>
  <text x="595" y="240" text-anchor="middle" font-size="13" font-weight="bold">Cancelled</text>
  
  <!-- Arrow: Cancelled -> Build Response -->
  <line x1="595" y1="260" x2="595" y2="750" stroke="#333" stroke-width="2"/>
  <line x1="595" y1="750" x2="505" y2="750" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <text x="605" y="500" font-size="10" fill="#666">resolve future</text>
  
  <!-- Arrow: Inference -> Parse Output -->
  <line x1="355" y1="260" x2="355" y2="300" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <text x="365" y="285" font-size="10" fill="#666">generation complete</text>
  
  <!-- Parse Output -->
  <rect x="280" y="305" width="150" height="50" rx="8" fill="#e8f4f8" stroke="#333" stroke-width="2"/>
  <text x="355" y="335" text-anchor="middle" font-size="13" font-weight="bold">Parse Output</text>
  
  <!-- Three branches from Parse Output -->
  <!-- No Tool Calls -->
  <line x1="280" y1="330" x2="100" y2="330" stroke="#333" stroke-width="2"/>
  <line x1="100" y1="330" x2="100" y2="380" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <rect x="30" y="385" width="140" height="45" rx="8" fill="#d4edda" stroke="#333" stroke-width="2"/>
  <text x="100" y="412" text-anchor="middle" font-size="12" font-weight="bold">No Tool Calls</text>
  
  <!-- Tool Call Detected -->
  <line x1="355" y1="355" x2="355" y2="395" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <rect x="280" y="400" width="150" height="45" rx="8" fill="#fff3cd" stroke="#333" stroke-width="2"/>
  <text x="355" y="427" text-anchor="middle" font-size="12" font-weight="bold">Tool Call Detected</text>
  
  <!-- Parse Error -->
  <line x1="430" y1="330" x2="530" y2="330" stroke="#333" stroke-width="2"/>
  <line x1="530" y1="330" x2="530" y2="380" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <rect x="460" y="385" width="140" height="45" rx="8" fill="#f8d7da" stroke="#333" stroke-width="2"/>
  <text x="530" y="412" text-anchor="middle" font-size="12" font-weight="bold">Parse Error</text>
  
  <!-- Arrow: Tool Call Detected -> Validate Args -->
  <line x1="355" y1="445" x2="355" y2="485" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  
  <!-- Validate Args -->
  <rect x="280" y="490" width="150" height="45" rx="8" fill="#e8f4f8" stroke="#333" stroke-width="2"/>
  <text x="355" y="517" text-anchor="middle" font-size="12" font-weight="bold">Validate Args</text>
  
  <!-- Valid branch -->
  <line x1="280" y1="512" x2="200" y2="512" stroke="#333" stroke-width="2"/>
  <line x1="200" y1="512" x2="200" y2="560" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <rect x="130" y="565" width="140" height="45" rx="8" fill="#d4edda" stroke="#333" stroke-width="2"/>
  <text x="200" y="592" text-anchor="middle" font-size="12" font-weight="bold">Execute Tool</text>
  <text x="235" y="540" font-size="10" fill="#666">valid</text>
  
  <!-- Invalid branch -->
  <line x1="430" y1="512" x2="510" y2="512" stroke="#333" stroke-width="2"/>
  <line x1="510" y1="512" x2="510" y2="560" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <rect x="440" y="565" width="140" height="45" rx="8" fill="#fff3cd" stroke="#333" stroke-width="2"/>
  <text x="510" y="592" text-anchor="middle" font-size="12" font-weight="bold">Error Recovery</text>
  <text x="475" y="540" font-size="10" fill="#666">invalid</text>
  
  <!-- Arrow: Execute Tool -> Inject Result -->
  <line x1="200" y1="610" x2="200" y2="660" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  
  <!-- Inject Result -->
  <rect x="130" y="665" width="140" height="45" rx="8" fill="#cce5ff" stroke="#333" stroke-width="2"/>
  <text x="200" y="692" text-anchor="middle" font-size="12" font-weight="bold">Inject Result</text>
  
  <!-- Error Recovery branches -->
  <!-- Retry OK -->
  <line x1="480" y1="610" x2="480" y2="650" stroke="#333" stroke-width="2"/>
  <line x1="480" y1="650" x2="270" y2="650" stroke="#333" stroke-width="2"/>
  <line x1="270" y1="650" x2="270" y2="687" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <text x="380" y="640" font-size="10" fill="#666">retry OK</text>
  
  <!-- Retries Exhausted -->
  <line x1="540" y1="610" x2="540" y2="660" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <rect x="470" y="665" width="140" height="45" rx="8" fill="#f8d7da" stroke="#333" stroke-width="2"/>
  <text x="540" y="692" text-anchor="middle" font-size="12" font-weight="bold">Retries Exhausted</text>
  
  <!-- Loop back: Inject Result -> Inference -->
  <line x1="130" y1="687" x2="40" y2="687" stroke="#333" stroke-width="2"/>
  <line x1="40" y1="687" x2="40" y2="235" stroke="#333" stroke-width="2"/>
  <line x1="40" y1="235" x2="275" y2="235" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <text x="50" y="460" font-size="10" fill="#666">loop back</text>
  
  <!-- Build Response -->
  <rect x="205" y="725" width="300" height="50" rx="8" fill="#e2e3e5" stroke="#333" stroke-width="2"/>
  <text x="355" y="755" text-anchor="middle" font-size="13" font-weight="bold">Build Response</text>
  
  <!-- Arrows to Build Response -->
  <line x1="100" y1="430" x2="100" y2="750" stroke="#333" stroke-width="2"/>
  <line x1="100" y1="750" x2="200" y2="750" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  
  <line x1="530" y1="430" x2="530" y2="720" stroke="#333" stroke-width="2"/>
  <line x1="530" y1="720" x2="510" y2="720" stroke="#333" stroke-width="2"/>
  <line x1="510" y1="720" x2="510" y2="750" stroke="#333" stroke-width="2"/>
  <line x1="510" y1="750" x2="510" y2="750" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  
  <line x1="540" y1="710" x2="540" y2="720" stroke="#333" stroke-width="2"/>
  
  <!-- Arrow: Build Response -> Await Request -->
  <line x1="355" y1="775" x2="355" y2="810" stroke="#333" stroke-width="2"/>
  <line x1="355" y1="810" x2="700" y2="810" stroke="#333" stroke-width="2"/>
  <line x1="700" y1="810" x2="700" y2="45" stroke="#333" stroke-width="2"/>
  <line x1="700" y1="45" x2="435" y2="45" stroke="#333" stroke-width="2" marker-end="url(#arrowhead2)"/>
  <text x="550" y="825" font-size="10" fill="#666">resolve future</text>
</svg>

**State Descriptions:**

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

### 4.3 History Manager State Machine

<svg viewBox="0 0 550 500" xmlns="http://www.w3.org/2000/svg" style="max-width: 550px; font-family: Arial, sans-serif;">
  <defs>
    <marker id="arrowhead3" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
  
  <!-- Empty -->
  <rect x="180" y="20" width="160" height="50" rx="8" fill="#e8f4f8" stroke="#333" stroke-width="2"/>
  <text x="260" y="42" text-anchor="middle" font-size="14" font-weight="bold">Empty</text>
  <text x="260" y="58" text-anchor="middle" font-size="11" fill="#666">(no messages)</text>
  
  <!-- Arrow: Empty -> System Only -->
  <line x1="260" y1="70" x2="260" y2="115" stroke="#333" stroke-width="2" marker-end="url(#arrowhead3)"/>
  <text x="270" y="98" font-size="10" fill="#666">set_system_prompt()</text>
  
  <!-- System Only -->
  <rect x="180" y="120" width="160" height="50" rx="8" fill="#fff3cd" stroke="#333" stroke-width="2"/>
  <text x="260" y="150" text-anchor="middle" font-size="14" font-weight="bold">System Only</text>
  
  <!-- Arrow: System Only -> Accumulating -->
  <line x1="260" y1="170" x2="260" y2="215" stroke="#333" stroke-width="2" marker-end="url(#arrowhead3)"/>
  <text x="270" y="198" font-size="10" fill="#666">add user message</text>
  
  <!-- Accumulating -->
  <rect x="180" y="220" width="160" height="50" rx="8" fill="#d4edda" stroke="#333" stroke-width="2"/>
  <text x="260" y="250" text-anchor="middle" font-size="14" font-weight="bold">Accumulating</text>
  
  <!-- Arrow: Accumulating -> Pruning -->
  <line x1="260" y1="270" x2="260" y2="315" stroke="#333" stroke-width="2" marker-end="url(#arrowhead3)"/>
  <text x="270" y="298" font-size="10" fill="#666">context window exceeded</text>
  
  <!-- Pruning -->
  <rect x="180" y="320" width="160" height="50" rx="8" fill="#cce5ff" stroke="#333" stroke-width="2"/>
  <text x="260" y="350" text-anchor="middle" font-size="14" font-weight="bold">Pruning</text>
  
  <!-- Arrow: Pruning -> Accumulating (loop back) -->
  <line x1="340" y1="345" x2="420" y2="345" stroke="#333" stroke-width="2"/>
  <line x1="420" y1="345" x2="420" y2="245" stroke="#333" stroke-width="2"/>
  <line x1="420" y1="245" x2="345" y2="245" stroke="#333" stroke-width="2" marker-end="url(#arrowhead3)"/>
  <text x="430" y="280" font-size="10" fill="#666">oldest pairs removed</text>
  <text x="430" y="293" font-size="10" fill="#666">(system prompt preserved)</text>
  <text x="430" y="306" font-size="10" fill="#666">prune callback fired</text>
  
  <!-- Arrow: Accumulating -> System Only (clear_history) -->
  <line x1="340" y1="245" x2="480" y2="245" stroke="#333" stroke-width="2"/>
  <line x1="480" y1="245" x2="480" y2="145" stroke="#333" stroke-width="2"/>
  <line x1="480" y1="145" x2="345" y2="145" stroke="#333" stroke-width="2" marker-end="url(#arrowhead3)"/>
  <text x="490" y="185" font-size="10" fill="#666">clear_history()</text>
  <text x="490" y="198" font-size="10" fill="#666">(preserves system prompt)</text>
  
  <!-- Arrow: update system prompt (Pruning/Accumulating -> System Only) -->
  <line x1="180" y1="345" x2="80" y2="345" stroke="#333" stroke-width="2"/>
  <line x1="80" y1="345" x2="80" y2="145" stroke="#333" stroke-width="2"/>
  <line x1="80" y1="145" x2="175" y2="145" stroke="#333" stroke-width="2" marker-end="url(#arrowhead3)"/>
  <text x="20" y="240" font-size="10" fill="#666">update</text>
  <text x="20" y="253" font-size="10" fill="#666">system</text>
  <text x="20" y="266" font-size="10" fill="#666">prompt</text>
</svg>

**State Descriptions:**

| State | Description |
|-------|-------------|
| **Empty** | No messages in history; initial state |
| **System Only** | Only system prompt present; awaiting first user message |
| **Accumulating** | Normal operation; messages being added within context window |
| **Pruning** | Context window exceeded; removing oldest User/Assistant pairs |

---

## 5. Threading Model

### 5.1 Thread Responsibilities

| Thread | Identity | Responsibilities |
|--------|----------|------------------|
| **Calling Thread** | Any thread invoking Agent methods | Submit chat requests; register tools; set system prompt; request cancellation; receive futures |
| **Inference Thread** | Dedicated thread owned by Agent | Process request queue; execute inference; run tool handlers; manage history; fire callbacks |

### 5.2 Thread Interaction Diagram

```
    Calling Thread                              Inference Thread
    ──────────────                              ────────────────
          │                                            │
          │  chat(message)                             │
          ├───────────────────────────────────────────►│
          │  [enqueue request]                         │
          │                                            │
          │◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
          │  returns std::future<Result<Response>>     │
          │                                            │
          │                                            │ [dequeue request]
          │                                            │
          │                                            │ [format prompt]
          │                                            │
          │                                            │ [run inference]
          │                         on_token(token)    │
          │◄ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
          │  [callback on inference thread]            │
          │                                            │
          │                                            │ [detect tool call]
          │                                            │
          │                                            │ [execute tool]
          │                                            │
          │                                            │ [continue inference]
          │                                            │
          │                                            │ [build response]
          │                                            │
          │◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
          │  [future resolved]                         │
          │                                            │
          │  stop()                                    │
          ├───────────────────────────────────────────►│
          │  [set cancellation flag]                   │
          │                                            │ [check flag, abort]
          │                                            │
```

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

```
┌─────────────────────────────────────────────────────────────────┐
│                         TDD Cycle                               │
│                                                                 │
│    ┌─────────┐      ┌─────────┐      ┌─────────┐               │
│    │         │      │         │      │         │               │
│    │  RED    │─────►│  GREEN  │─────►│ REFACTOR│───┐           │
│    │         │      │         │      │         │   │           │
│    └─────────┘      └─────────┘      └─────────┘   │           │
│         ▲                                          │           │
│         │                                          │           │
│         └──────────────────────────────────────────┘           │
│                                                                 │
│    RED:      Write failing test for new requirement            │
│    GREEN:    Write minimal code to pass test                   │
│    REFACTOR: Improve code while keeping tests green            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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

```
tests/
├── unit/
│   ├── test_request_queue.cpp
│   ├── test_history_manager.cpp
│   ├── test_tool_registry.cpp
│   ├── test_agentic_loop.cpp
│   ├── test_template_engine.cpp
│   ├── test_error_recovery.cpp
│   └── test_agent.cpp
├── mocks/
│   ├── mock_backend.hpp
│   └── mock_backend.cpp
├── fixtures/
│   ├── sample_responses.hpp
│   └── tool_definitions.hpp
└── CMakeLists.txt
```

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
| 1.0 | [Date] | [Author] | Initial TRD created |