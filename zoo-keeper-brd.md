# Business Requirements Document: Zoo-Keeper Library
**Version:** 1.1
**Scope:** C++ Asynchronous Inference & Tool Use Wrapper for llama.cpp

## 1. Executive Summary
Zoo-Keeper is a high-level C++ library designed to democratize access to local LLM inference and agentic workflows. It abstracts the complexity of `llama.cpp` into a developer-friendly, asynchronous API. It manages the entire lifecycle of a chat session, including context window management, tool execution loops, and the injection of retrieval-augmented generation (RAG) data, allowing developers to focus on building features rather than managing tensors.

---

## 2. Functional Requirements (FR)

### FR-01: Asynchronous Inference Engine
* **Description:** The inference process must not block the calling thread. The library must provide a non-blocking mechanism to request generation and receive results.
* **User Story:** As a developer building a GUI application, I need to send a prompt to the model and receive tokens as they are generated without freezing my user interface.
* **Acceptance Criteria:**
    * The primary `chat()` function returns immediately (e.g., returning a `std::future` or handle).
    * Support for callback functions to handle streaming tokens in real-time.
    * Support for a "completion" callback when the full generation (including potential tool loops) is finished.

### FR-02: Primitive-Typed Tool Registration
* **Description:** The library must allow the registration of C++ functions as tools using standard primitive types (`int`, `float`, `double`, `bool`, `std::string`).
* **User Story:** As a developer, I want to bind a C++ function `calculate_tax(float amount, string region)` to the agent without manually writing a JSON schema.
* **Acceptance Criteria:**
    * The API automatically inspects the function signature (via templates/meta-programming) to generate the compatible JSON tool schema required by the model.
    * The library handles the deserialization of the model's JSON output into the specific C++ primitive types required by the function.
    * Type-checking: If the model hallucinates the wrong data type, the library handles the error gracefully.

### FR-03: Automated "Agentic Loop"
* **Description:** The library acts as the orchestrator for tool usage. It must handle the "Stop -> Execute -> Resume" cycle autonomously.
* **User Story:** As a developer, I want to ask "What is 5 * 5?" and get the answer "25" without having to manually intercept the tool call, run the math function, and feed the result back to the model.
* **Acceptance Criteria:**
    * The library detects when a model output indicates a tool call.
    * The library pauses generation, routes the execution to the registered C++ function, captures the return value, feeds it back into the context, and resumes generation.
    * This loop occurs within the asynchronous context, eventually returning the final natural language answer to the user.

### FR-04: Managed Context History
* **Description:** The library maintains the state of the conversation (the "Context Window") automatically.
* **User Story:** As a developer, I want to call `agent.chat("Hello")` followed by `agent.chat("My name is John")` and have the model remember the previous turn without me re-sending the whole history.
* **Acceptance Criteria:**
    * The library stores the conversation history (System, User, Assistant, Tool messages).
    * The library automatically handles the prompt formatting (templating) required by the specific model loaded (e.g., ChatML, Llama-3 format).
    * **Context Shifting:** If the conversation exceeds the context window, the library automatically truncates the oldest messages (preserving the System Prompt) to make room.

### FR-05: RAG Context Injection (Ephemeral Context)
* **Description:** The API must allow users to inject external data (RAG context) into the prompt for a specific turn without permanently polluting the long-term conversation history.
* **User Story:** As a developer, I have retrieved 3 documents from a vector database. I want to pass these to the model to answer the user's specific question, but I don't need these documents to stay in the context window for the rest of the session.
* **Acceptance Criteria:**
    * API provides a method (e.g., `chat(user_input, additional_context_vector)`) to insert data.
    * This injected context is prioritized in the prompt structure but marked as ephemeral (or effectively managed so it doesn't displace the core conversation history unnecessarily).

---

## 3. Non-Functional Requirements (NFR)

### NFR-01: Thread Safety
* **Description:** Since the API is asynchronous, the library must be thread-safe.
* **Requirement:** Internal state (context history, KV cache) must be protected against race conditions if the user attempts to send a new message while the previous one is still processing.

### NFR-02: Zero-Copy String Handling (Target)
* **Description:** Minimize memory overhead when passing strings between the C++ application and the inference engine.
* **Requirement:** Where possible, use `std::string_view` or references to avoid copying large text blocks (like RAG context) multiple times in memory.

### NFR-03: Hardware Optimization (Apple Silicon & CUDA)
* **Description:** The library must inherit the hardware acceleration capabilities of `llama.cpp`.
* **Requirement:** Configuration options must allow the user to specify GPU layer counts. On macOS, the library must default to Metal acceleration.

---
