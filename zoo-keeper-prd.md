# Product Requirements Document: Zoo-Keeper
**Version:** 1.0 (Semi-Final)
**Status:** Ready for Technical Prototype
**Product:** Zoo-Keeper (C++ LLM Agent Library)

## 1. Product Overview
Zoo-Keeper is a modern C++ wrapper for the `llama.cpp` inference library, designed to function as an "Agent Engine." Unlike raw inference libraries that focus primarily on token generation, Zoo-Keeper focuses on the **Agentic Loop**: managing conversation history, orchestrating tool execution, and handling asynchronous inference.

## 2. Target Audience & Use Cases
* **Primary User:** C++ Application Developers (Game Development, Systems Programming, Desktop UI).
* **Environment:** Local execution (Edge devices, Consumer desktops, Apple Silicon Macs).
* **Key Use Case:** Embedding "Smart Agents" into local software that can interact with the host system via native code without blocking the user interface.

---

## 3. Functional Specifications

### 3.1. Asynchronous Inference Manager
* **Request Queue:** Thread-safe queue for incoming chat requests.
* **Background Worker:** A dedicated worker thread manages the model context, ensuring thread-safe access to the underlying `llama_context`.
* **Cancellation:** Support for immediate "Stop" commands to abort active generation.

### 3.2. Automated Context Management
* **State Tracking:** Automatically maintains a buffer of User, Assistant, System, and Tool messages.
* **KV Cache Management:** Reuses the KV cache for shared prefixes to minimize latency on multi-turn conversations.
* **Context Pruning (Sliding Window):** When the window is full, the library uses a **FIFO (First-In, First-Out) Pruning** strategy, dropping the oldest User/Assistant pairs while strictly preserving the System Prompt.

### 3.3. RAG & Ephemeral Context
* **Injection:** Accepts a `RAG_CONTEXT` payload alongside user prompts.
* **Lifecycle:** Injected context is prioritized for the current inference turn but is **not** stored in the long-term history buffer to prevent window saturation.

### 3.4. The Tool Registry & Agentic Loop
* **Automatic Schema Generation:** Template-based registration for functions using primitive types (`int`, `float`, `double`, `bool`, `std::string`).
* **Orchestration:** The library detects tool calls, pauses inference, executes the C++ function, and feeds the result back into the model automatically.

---

## 4. Advanced Technical Specifications (New)

### 4.1. Model Template Mapping
To ensure cross-model compatibility, Zoo-Keeper must handle the diversity of prompt formats.
* **Auto-Detection:** The library will attempt to read the model template from GGUF metadata.
* **Manual Override:** Users can specify the template (e.g., `Llama3`, `Mistral`, `ChatML`) in the `Config` object.
* **Abstraction:** The library translates the internal `Message` buffer into the specific string format required by the model at runtime.

### 4.2. Error Recovery & Validation
To handle model "hallucinations" or malformed outputs:
* **Schema Validation:** If the model generates a tool call with incorrect types or missing parameters, the library will not crash.
* **Self-Correction Loop:** Instead of returning a C++ error to the user, the library will inject a hidden "System Message" describing the error back into the model, prompting it to try the tool call again with the correct parameters (limit: 2 retries).

### 4.3. Threading & Callback Model
* **Callback Context:** All streaming callbacks (`on_token`) and tool executions are performed on the **Inference Thread** to avoid context-switching overhead.
* **User Responsibility:** The library consumer is responsible for dispatching UI updates to the main thread (e.g., using a signal/slot system or a thread-safe queue).

---

## 5. Technical Architecture

### 5.1. Threading Model
* **Main Thread:** Submits `chat()` requests and receives a `std::future<Response>`.
* **Inference Thread:** Handles evaluation, tool execution, and history pruning.

### 5.2. Dependency Management
* **Inference:** `llama.cpp` (git submodule).
* **JSON:** `jsoncpp` or `nlohmann/json` (configurable).
* **Build System:** CMake 3.20+.

---

## 6. Non-Functional Requirements (NFR)

* **P-01 Latency:** Internal library overhead < 10ms.
* **P-02 Safety:** Thread-safe access to the internal message buffer.
* **C-01 Hardware:** Must default to Metal on macOS and support CUDA for NVIDIA GPUs.
* **C-02 Cross-Platform:** Support for Windows (MSVC), macOS (Clang), and Linux (GCC).

---

## 7. Development Phases

1.  **Phase 1 (MVP):** Async inference and basic GGUF loading.
2.  **Phase 2 (Tools):** Template reflection for tool registration and self-correction logic.
3.  **Phase 3 (Optimization):** RAG injection, KV cache shifting, and automated template detection.