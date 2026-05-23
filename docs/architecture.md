# Architecture

Zoo-Keeper exposes three core public layers plus an optional hub layer. Higher
layers build on lower layers, and consumers can stop at the lowest layer that
fits their needs.

```mermaid
flowchart TB
    subgraph L4["Layer 4 — Hub (optional, ZOO_BUILD_HUB=ON)"]
        H["zoo::hub<br/>HuggingFaceClient · ModelStore"]
    end

    subgraph L3["Layer 3 — Agent"]
        A["zoo::Agent<br/>RequestHandle · async orchestration"]
    end

    subgraph L2["Layer 2 — Tools (llama.cpp-free)"]
        T["zoo::tools<br/>ToolRegistry · Parser · Validator"]
    end

    subgraph L1["Layer 1 — Core"]
        C["zoo::core<br/>Model · GgufInspector · SystemProbe"]
    end

    subgraph Engine["Inference engine"]
        LL["llama.cpp + llama-common"]
    end

    H --> A
    A --> T
    A --> C
    C --> LL

    style L4 fill:#f0f4ff,stroke:#4a6fa5
    style L3 fill:#eef8f0,stroke:#3d8b5a
    style L2 fill:#fff8ee,stroke:#c49a3c
    style L1 fill:#f5f0ff,stroke:#7b5ea7
    style Engine fill:#f4f4f4,stroke:#666
```

## Public Layers

| Layer | Primary Types | Responsibility |
|-------|---------------|----------------|
| Hub *(optional)* | `zoo::hub::HuggingFaceClient`, `zoo::hub::ModelStore` | HuggingFace downloads, local model cataloging |
| Agent | `zoo::Agent`, `zoo::RequestHandle<Result>` | Async request submission, background inference, native tool orchestration |
| Tools | `zoo::tools::ToolRegistry`, `zoo::tools::ToolCallParser`, `zoo::tools::ToolArgumentsValidator` | Tool registration, native tool-call parsing, schema validation |
| Core | `zoo::core::Model`, `zoo::core::GgufInspector`, `zoo::core::SystemProbe` | Direct synchronous llama.cpp wrapper, GGUF metadata read, hardware probe, hardware-aware auto-configuration |

## Usage Model

### `zoo::core::Model`

Use `Model` when you want direct, single-threaded inference without the agent
runtime. It owns model loading, prompt rendering, history, KV-cache
interaction, sampling, and generation.

### `zoo::Agent`

Use `Agent` when you want queued asynchronous requests, streaming callbacks,
cancellation, native tool execution, and a choice between stateful `chat(...)`
requests and stateless request-scoped `complete(...)` requests. `Agent` is the
primary high-level runtime surface for most consumers.

`RequestHandle<Result>` is the public async return type. It carries the request
ID and exposes `cancel()`, `ready()`, and `await_result()` for cancellation,
polling, and retrieving the completed response or error.

### Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant App as Calling thread
    participant Facade as zoo::Agent
    participant Slots as RequestSlots
    participant Mailbox as RuntimeMailbox
    participant Inf as Inference thread
    participant Model as zoo::core::Model
    participant CB as CallbackDispatcher

    App->>Facade: chat(message, callback)
    Facade->>Slots: allocate slot + RequestHandle
    Facade->>Mailbox: push_request(QueuedRequest)
    Facade-->>App: RequestHandle<TextResponse>

    loop until request complete
        Inf->>Mailbox: pop (commands first)
        Inf->>Model: render prompt + generate
        Model-->>Inf: token(s)
        opt streaming callback registered
            Inf->>CB: dispatch token
            CB->>App: on_token(token)
            App-->>CB: Continue / Stop
        end
    end

    Inf->>Slots: complete slot with TextResponse
    App->>Facade: await_result()
    Facade->>Slots: wait + release
    Facade-->>App: Expected<TextResponse>
```

1. The calling thread submits via `chat()`, `complete()`, or `extract()` and
   receives a `RequestHandle<Result>` immediately.
2. The runtime enqueues work on the inference thread through
   `RuntimeMailbox` (commands are prioritized over queued requests).
3. Generation runs on `zoo::core::Model`; optional streaming callbacks are
   dispatched on `CallbackDispatcher`.
4. The caller observes completion through `await_result()`.

## Public Threading Guarantees

- `zoo::Agent` owns a background inference thread.
- Requests are submitted from the calling thread through `chat(...)`,
  `complete(...)`, or `extract(...)`.
- Request completion is observed through `RequestHandle<Result>::await_result()`.
- Model state is owned by the inference thread while the agent is running.
- Streaming token callbacks execute on the CallbackDispatcher thread. Tool
  handlers execute on a dedicated ToolExecutor worker while the tool loop waits
  for their result.
- Direct `ToolRegistry` use is single-threaded unless callers externally
  synchronize overlapping operations. `Agent` serializes registry mutation on
  its inference thread.

```mermaid
flowchart LR
    subgraph Caller["Calling thread(s)"]
        APP["Application code"]
        SUB["chat() · complete() · extract()"]
        AWAIT["RequestHandle::await_result()"]
        APP --> SUB --> AWAIT
    end

    subgraph Runtime["Agent runtime"]
        MB["RuntimeMailbox<br/>requests + commands"]
        INF["Inference thread<br/>AgentRuntime"]
        BE["AgentBackend → Model"]
        MB --> INF --> BE
    end

    subgraph Workers["Dedicated workers"]
        CB["CallbackDispatcher<br/>streaming token callbacks"]
        TE["ToolExecutor<br/>user tool handlers"]
    end

    SUB -->|"enqueue request"| MB
    SUB -->|"return handle"| AWAIT
    INF -->|"dispatch tokens"| CB
    CB -->|"TokenAction::Continue / Stop"| APP
    INF -->|"invoke handler"| TE
    TE -->|"result"| INF
    AWAIT -->|"poll / block until done"| INF

    style Caller fill:#eef4ff,stroke:#4a6fa5
    style Runtime fill:#eef8f0,stroke:#3d8b5a
    style Workers fill:#fff8ee,stroke:#c49a3c
```

These guarantees are part of the public behavioral contract. Private runtime
mechanisms that implement them are documented separately for maintainers.

## Tool Calling Model

Tool calling is native-only. Zoo-Keeper only executes model-emitted native tool
calls when the active model/template supports them. If the selected model does
not expose native tool calling, the runtime remains on the text path.

When `GenerationOptions::record_tool_trace` is enabled, the request can retain
a `tool_trace` describing the attempts made during the tool loop.

```mermaid
flowchart TD
    START(["User request enters tool loop"])
    GEN["Model generates tokens<br/>(grammar-constrained when tools active)"]
    PARSE["parse_tool_response()<br/>native format detection"]
    TEXT{"Tool calls<br/>detected?"}
    DONE(["Return TextResponse<br/>+ optional tool_trace"])
    VAL["Validate arguments<br/>against registered schema"]
    OK{"Valid?"}
    EXEC["ToolExecutor runs<br/>registered handler"]
    INJ["Inject tool result message<br/>into conversation"]
    RETRY{"Retries<br/>remaining?"}
    FAIL(["Fail: ToolRetriesExhausted"])
    LIMIT{"Within<br/>iteration budget?"}
    LIMITFAIL(["Fail: ToolLoopLimitReached"])

    START --> GEN --> PARSE --> TEXT
    TEXT -->|no| DONE
    TEXT -->|yes| LIMIT
    LIMIT -->|no| LIMITFAIL
    LIMIT -->|yes| VAL --> OK
    OK -->|yes| EXEC --> INJ --> GEN
    OK -->|no| RETRY
    RETRY -->|yes| INJ
    RETRY -->|no| FAIL

    style START fill:#eef4ff,stroke:#4a6fa5
    style DONE fill:#eef8f0,stroke:#3d8b5a
    style FAIL fill:#ffecec,stroke:#c44
    style LIMITFAIL fill:#ffecec,stroke:#c44
```

See [tools.md](tools.md) for registration, schema rules, and error codes.

## CMake Targets

| Target | Status | Notes |
|--------|--------|-------|
| `ZooKeeper::zoo` | Primary | Recommended target for new consumers |
| `ZooKeeper::zoo_core` | Compatibility only | Forwarding target retained for existing consumers |

## Design Goals

- One obvious public runtime story centered on `ZooKeeper::zoo`
- Small installed API surface under `include/zoo/`
- Explicit native tool execution data and deterministic tool metadata behavior
- Docs that describe supported behavior without exposing private implementation
  details as API

## For Maintainers

Internal runtime ownership, private module boundaries, and contributor-facing
invariants live in [maintainer-architecture.md](maintainer-architecture.md).
