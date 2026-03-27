# Concurrency Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve concurrency in three areas: release the registry lock before executing tool handlers, add a batch tool registration API to avoid per-tool round-trips, and offload streaming callbacks off the inference thread.

**Architecture:** Three independent improvements, each touching different files with no cross-dependencies. (1) Registry `invoke()` copies the handler and releases the shared lock before executing. (2) A new `register_tools()` method on both `Agent` and `AgentRuntime` accepts a vector of definitions and sends a single `update_tool_calling()` command. (3) Streaming callbacks are dispatched to a dedicated callback thread via a lock-free queue, keeping the inference thread unblocked during user callback execution.

**Tech Stack:** C++23, std::shared_mutex, std::thread, std::mutex, std::condition_variable, GoogleTest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `include/zoo/tools/registry.hpp` | Modify | Copy handler out of lock before invoking |
| `tests/unit/test_tool_registry.cpp` | Modify | Add test for lock-release-before-invoke behavior |
| `include/zoo/agent.hpp` | Modify | Add `register_tools()` batch API |
| `src/agent.cpp` | Modify | Implement `Agent::register_tools()` delegation |
| `include/zoo/internal/agent/runtime.hpp` | Modify | Add `AgentRuntime::register_tools()` |
| `src/agent/runtime.cpp` | Modify | Implement `AgentRuntime::register_tools()` |
| `tests/unit/test_agent_runtime.cpp` | Modify | Test batch registration sends single update |
| `tests/unit/test_tool_registry.cpp` | Modify | Test batch registration on ToolRegistry |
| `include/zoo/internal/agent/callback_dispatcher.hpp` | Create | Lock-free callback queue + dispatcher thread |
| `include/zoo/internal/agent/request.hpp` | Modify | Check for callback dispatcher reference |
| `src/agent/runtime_tool_loop.cpp` | Modify | Dispatch callbacks through dispatcher |
| `src/agent/runtime_extraction.cpp` | Modify | Dispatch callbacks through dispatcher |
| `tests/unit/test_callback_dispatcher.cpp` | Create | Unit tests for callback dispatcher |
| `tests/unit/test_agent_runtime.cpp` | Modify | Test callbacks arrive on non-inference thread |

---

### Task 1: Registry invoke() — release lock before handler execution

**Files:**
- Modify: `include/zoo/tools/registry.hpp:595-602`
- Modify: `tests/unit/test_tool_registry.cpp`

- [ ] **Step 1: Write the failing test**

Add a test that proves handler execution does not block concurrent registry reads. The test registers a slow tool handler, invokes it on one thread, and simultaneously calls `has_tool()` on another. If the lock is held during execution, `has_tool()` will block for the full handler duration.

In `tests/unit/test_tool_registry.cpp`:

```cpp
TEST_F(ToolRegistryTest, InvokeDoesNotBlockConcurrentReads) {
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    zoo::tools::ToolHandler slow_handler =
        [entered, release_future](const json&) -> zoo::Expected<json> {
        entered->set_value();
        release_future.wait();
        return json{{"result", "done"}};
    };

    ASSERT_TRUE(
        registry
            .register_tool("slow", "A slow tool",
                           json{{"type", "object"}, {"properties", json::object()}}, std::move(slow_handler))
            .has_value());

    std::thread invoker([this] { registry.invoke("slow", json::object()); });

    ASSERT_EQ(entered_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);

    auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(registry.has_tool("slow"));
    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_LT(elapsed, std::chrono::milliseconds(100));

    release->set_value();
    invoker.join();
}
```

Required includes at top of test file:

```cpp
#include <chrono>
#include <future>
#include <thread>
```

- [ ] **Step 2: Run test to verify it fails**

Run: `scripts/build && scripts/test -R ToolRegistryTest.InvokeDoesNotBlockConcurrentReads`
Expected: FAIL — `has_tool()` blocks until the slow handler completes because invoke holds the shared_lock.

- [ ] **Step 3: Implement the fix**

In `include/zoo/tools/registry.hpp`, replace the `invoke` method (lines 595-602):

```cpp
Expected<nlohmann::json> invoke(const std::string& name, const nlohmann::json& args) const {
    ToolHandler handler;
    {
        std::shared_lock lock(mutex_);
        auto it = index_by_name_.find(name);
        if (it == index_by_name_.end()) {
            return std::unexpected(Error{ErrorCode::ToolNotFound, "Tool not found: " + name});
        }
        handler = tools_[it->second].handler;
    }
    return handler(args);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `scripts/test -R ToolRegistryTest.InvokeDoesNotBlockConcurrentReads`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `scripts/test`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add include/zoo/tools/registry.hpp tests/unit/test_tool_registry.cpp
git commit -m "perf: release registry lock before executing tool handler

invoke() now copies the handler and releases the shared_lock before
calling it, so concurrent reads are not blocked by slow handlers."
```

---

### Task 2: Batch tool registration on ToolRegistry

**Files:**
- Modify: `include/zoo/tools/registry.hpp`
- Modify: `tests/unit/test_tool_registry.cpp`

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_tool_registry.cpp`:

```cpp
TEST_F(ToolRegistryTest, RegisterToolsBatchAddsAllTools) {
    auto def1 = zoo::tools::detail::make_tool_definition("add", "Add", std::vector<std::string>{"a", "b"}, add);
    auto def2 = zoo::tools::detail::make_tool_definition("greet", "Greet", std::vector<std::string>{"name"}, greet);
    ASSERT_TRUE(def1.has_value());
    ASSERT_TRUE(def2.has_value());

    std::vector<zoo::tools::ToolDefinition> definitions;
    definitions.push_back(std::move(*def1));
    definitions.push_back(std::move(*def2));

    auto result = registry.register_tools(std::move(definitions));
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(registry.size(), 2u);
    EXPECT_TRUE(registry.has_tool("add"));
    EXPECT_TRUE(registry.has_tool("greet"));

    auto names = registry.get_tool_names();
    EXPECT_EQ(names, std::vector<std::string>({"add", "greet"}));
}

TEST_F(ToolRegistryTest, RegisterToolsBatchEmptyIsNoOp) {
    auto result = registry.register_tools({});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(registry.size(), 0u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `scripts/build && scripts/test -R ToolRegistryTest.RegisterToolsBatch`
Expected: FAIL — `register_tools` does not exist.

- [ ] **Step 3: Implement register_tools on ToolRegistry**

In `include/zoo/tools/registry.hpp`, add after the existing `register_tool(ToolDefinition)` method (after line 582):

```cpp
Expected<void> register_tools(std::vector<ToolDefinition> definitions) {
    std::unique_lock lock(mutex_);
    for (auto& definition : definitions) {
        auto it = index_by_name_.find(definition.metadata.name);
        if (it != index_by_name_.end()) {
            tools_[it->second] = std::move(definition);
        } else {
            const size_t index = tools_.size();
            index_by_name_.emplace(definition.metadata.name, index);
            tools_.push_back(std::move(definition));
        }
    }
    return {};
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `scripts/test -R ToolRegistryTest.RegisterToolsBatch`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add include/zoo/tools/registry.hpp tests/unit/test_tool_registry.cpp
git commit -m "feat: add ToolRegistry::register_tools() batch API

Accepts a vector of ToolDefinitions and registers them under a single
lock acquisition, avoiding repeated lock/unlock overhead."
```

---

### Task 3: Batch tool registration on AgentRuntime and Agent

**Files:**
- Modify: `include/zoo/internal/agent/runtime.hpp`
- Modify: `src/agent/runtime.cpp`
- Modify: `include/zoo/agent.hpp`
- Modify: `src/agent.cpp`
- Modify: `tests/unit/test_agent_runtime.cpp`

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_agent_runtime.cpp`, add a test that batch-registers multiple tools and verifies they all work:

```cpp
TEST(AgentRuntimeTest, RegisterToolsBatchRegistersAllToolsWithSingleUpdate) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    auto def1 = zoo::tools::detail::make_tool_definition(
        "add", "Add two numbers", std::vector<std::string>{"a", "b"},
        [](int a, int b) { return a + b; });
    auto def2 = zoo::tools::detail::make_tool_definition(
        "greet", "Greet someone", std::vector<std::string>{"name"},
        [](std::string name) { return "Hello, " + name + "!"; });
    ASSERT_TRUE(def1.has_value());
    ASSERT_TRUE(def2.has_value());

    std::vector<zoo::tools::ToolDefinition> definitions;
    definitions.push_back(std::move(*def1));
    definitions.push_back(std::move(*def2));

    auto result = runtime.register_tools(std::move(definitions));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(runtime.tool_count(), 2u);

    // Verify tools are usable via a tool-calling flow
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(tool_call_generation("add", {{"a", 3}, {"b", 4}}));
    });
    backend_ptr->push_generation([](TokenCallback, const CancellationCallback&) {
        return Expected<GenerationResult>(GenerationResult{"7", 0, false, "", {}});
    });

    GenerationOptions options;
    options.record_tool_trace = true;
    auto handle = runtime.chat("add 3 and 4", options);
    auto chat_result = handle.await_result();
    ASSERT_TRUE(chat_result.has_value()) << chat_result.error().to_string();
    EXPECT_EQ(chat_result->text, "7");
    ASSERT_TRUE(chat_result->tool_trace.has_value());
    EXPECT_EQ(chat_result->tool_trace->invocations[0].status, ToolInvocationStatus::Succeeded);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `scripts/build && scripts/test -R AgentRuntimeTest.RegisterToolsBatch`
Expected: FAIL — `register_tools` does not exist on AgentRuntime.

- [ ] **Step 3: Add register_tools to AgentRuntime**

In `include/zoo/internal/agent/runtime.hpp`, add after `register_tool` declaration (line 66):

```cpp
Expected<void> register_tools(std::vector<tools::ToolDefinition> definitions);
```

In `src/agent/runtime.cpp`, add after the existing `register_tool` method (after line 230):

```cpp
Expected<void> AgentRuntime::register_tools(std::vector<tools::ToolDefinition> definitions) {
    assert(!inference_thread_.joinable() ||
           std::this_thread::get_id() != inference_thread_.get_id());

    if (definitions.empty()) {
        return {};
    }

    if (auto result = tool_registry_.register_tools(std::move(definitions)); !result) {
        return std::unexpected(result.error());
    }

    update_tool_calling();
    return {};
}
```

- [ ] **Step 4: Add register_tools to Agent public API**

In `include/zoo/agent.hpp`, add after the existing `register_tool` overloads (after line 256):

```cpp
Expected<void> register_tools(std::vector<tools::ToolDefinition> definitions);
```

In `src/agent.cpp`, add after the existing `register_tool` methods (after line 130):

```cpp
Expected<void> Agent::register_tools(std::vector<tools::ToolDefinition> definitions) {
    return impl_->runtime.register_tools(std::move(definitions));
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `scripts/test -R AgentRuntimeTest.RegisterToolsBatch`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `scripts/test`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add include/zoo/internal/agent/runtime.hpp src/agent/runtime.cpp include/zoo/agent.hpp src/agent.cpp tests/unit/test_agent_runtime.cpp
git commit -m "feat: add Agent::register_tools() batch API

Registers multiple tools in one call with a single update_tool_calling()
round-trip to the inference thread, avoiding N round-trips for N tools."
```

---

### Task 4: Callback dispatcher — core implementation

**Files:**
- Create: `include/zoo/internal/agent/callback_dispatcher.hpp`
- Create: `tests/unit/test_callback_dispatcher.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_callback_dispatcher.cpp`:

```cpp
/**
 * @file test_callback_dispatcher.cpp
 * @brief Unit tests for the async callback dispatcher.
 */

#include "zoo/internal/agent/callback_dispatcher.hpp"
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;
using zoo::internal::agent::CallbackDispatcher;

TEST(CallbackDispatcherTest, DispatchedCallbacksArriveInOrder) {
    CallbackDispatcher dispatcher;

    std::mutex mutex;
    std::vector<std::string> received;

    auto callback = [&](std::string_view token) {
        std::lock_guard<std::mutex> lock(mutex);
        received.emplace_back(token);
    };

    dispatcher.dispatch(callback, "hello ");
    dispatcher.dispatch(callback, "world");
    dispatcher.drain();

    std::lock_guard<std::mutex> lock(mutex);
    ASSERT_EQ(received.size(), 2u);
    EXPECT_EQ(received[0], "hello ");
    EXPECT_EQ(received[1], "world");
}

TEST(CallbackDispatcherTest, CallbacksRunOnDispatcherThread) {
    CallbackDispatcher dispatcher;

    std::thread::id callback_thread_id;
    std::promise<void> done;
    auto future = done.get_future();

    auto callback = [&](std::string_view) {
        callback_thread_id = std::this_thread::get_id();
        done.set_value();
    };

    dispatcher.dispatch(callback, "test");
    ASSERT_EQ(future.wait_for(2s), std::future_status::ready);

    EXPECT_NE(callback_thread_id, std::thread::id{});
    EXPECT_NE(callback_thread_id, std::this_thread::get_id());
}

TEST(CallbackDispatcherTest, DrainBlocksUntilAllCallbacksComplete) {
    CallbackDispatcher dispatcher;

    std::atomic<int> count{0};
    auto callback = [&](std::string_view) {
        std::this_thread::sleep_for(10ms);
        count.fetch_add(1, std::memory_order_relaxed);
    };

    for (int i = 0; i < 5; ++i) {
        dispatcher.dispatch(callback, "x");
    }
    dispatcher.drain();

    EXPECT_EQ(count.load(std::memory_order_relaxed), 5);
}

TEST(CallbackDispatcherTest, DestructorDrainsRemainingCallbacks) {
    std::atomic<int> count{0};
    auto callback = [&](std::string_view) {
        count.fetch_add(1, std::memory_order_relaxed);
    };

    {
        CallbackDispatcher dispatcher;
        for (int i = 0; i < 3; ++i) {
            dispatcher.dispatch(callback, "x");
        }
    } // destructor should drain

    EXPECT_EQ(count.load(std::memory_order_relaxed), 3);
}

TEST(CallbackDispatcherTest, NoCallbackIsNoOp) {
    CallbackDispatcher dispatcher;
    dispatcher.drain(); // should not hang
}

} // namespace
```

Add the test file to `tests/CMakeLists.txt` in the `add_executable` list:

```
unit/test_callback_dispatcher.cpp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `scripts/build && scripts/test -R CallbackDispatcherTest`
Expected: FAIL — file `callback_dispatcher.hpp` does not exist.

- [ ] **Step 3: Implement CallbackDispatcher**

Create `include/zoo/internal/agent/callback_dispatcher.hpp`:

```cpp
/**
 * @file callback_dispatcher.hpp
 * @brief Offloads streaming callbacks to a dedicated thread.
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>

namespace zoo::internal::agent {

/**
 * @brief Dispatches streaming callbacks on a dedicated thread.
 *
 * The inference thread calls `dispatch()` to enqueue a callback invocation
 * without blocking on the user's callback. The dispatcher thread executes
 * callbacks in FIFO order. `drain()` blocks until all queued callbacks have
 * been executed, providing a synchronization point between generation passes.
 */
class CallbackDispatcher {
  public:
    CallbackDispatcher() : thread_([this] { run(); }) {}

    ~CallbackDispatcher() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    CallbackDispatcher(const CallbackDispatcher&) = delete;
    CallbackDispatcher& operator=(const CallbackDispatcher&) = delete;
    CallbackDispatcher(CallbackDispatcher&&) = delete;
    CallbackDispatcher& operator=(CallbackDispatcher&&) = delete;

    /**
     * @brief Enqueues a callback invocation for async execution.
     *
     * The token string is copied into the queue. The callback reference must
     * remain valid until `drain()` returns.
     */
    void dispatch(std::function<void(std::string_view)>& callback, std::string_view token) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(Entry{&callback, std::string(token)});
        }
        cv_.notify_one();
    }

    /**
     * @brief Blocks until all previously dispatched callbacks have executed.
     */
    void drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        drain_cv_.wait(lock, [this] { return queue_.empty() && !executing_; });
    }

  private:
    struct Entry {
        std::function<void(std::string_view)>* callback;
        std::string token;
    };

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            cv_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });

            while (!queue_.empty()) {
                auto entry = std::move(queue_.front());
                queue_.pop();
                executing_ = true;
                lock.unlock();

                (*entry.callback)(entry.token);

                lock.lock();
                executing_ = false;
                drain_cv_.notify_all();
            }

            if (shutdown_) {
                return;
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable drain_cv_;
    std::queue<Entry> queue_;
    bool shutdown_ = false;
    bool executing_ = false;
    std::thread thread_;
};

} // namespace zoo::internal::agent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `scripts/build && scripts/test -R CallbackDispatcherTest`
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add include/zoo/internal/agent/callback_dispatcher.hpp tests/unit/test_callback_dispatcher.cpp tests/CMakeLists.txt
git commit -m "feat: add CallbackDispatcher for async callback offloading

Provides a dedicated thread that executes streaming callbacks in FIFO
order, so the inference thread is not blocked by user callback logic."
```

---

### Task 5: Integrate CallbackDispatcher into AgentRuntime

**Files:**
- Modify: `include/zoo/internal/agent/runtime.hpp`
- Modify: `src/agent/runtime_tool_loop.cpp`
- Modify: `src/agent/runtime_extraction.cpp`
- Modify: `tests/unit/test_agent_runtime.cpp`

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_agent_runtime.cpp`, add a test proving callbacks run on a different thread than inference:

```cpp
TEST(AgentRuntimeTest, StreamingCallbackRunsOffInferenceThread) {
    auto backend = std::make_unique<FakeBackend>();
    auto* backend_ptr = backend.get();
    AgentRuntime runtime(make_model_config(), make_agent_config(), GenerationOptions{},
                         std::move(backend));

    std::thread::id inference_thread_id;
    std::thread::id callback_thread_id;

    backend_ptr->push_generation(
        [&inference_thread_id](TokenCallback on_token, const CancellationCallback&) {
            inference_thread_id = std::this_thread::get_id();
            if (on_token) {
                on_token("hello");
            }
            return Expected<GenerationResult>(GenerationResult{"hello", 5, false, "", {}});
        });

    auto handle = runtime.chat("test", GenerationOptions{}, [&](std::string_view) {
        callback_thread_id = std::this_thread::get_id();
    });

    auto result = handle.await_result();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    EXPECT_NE(inference_thread_id, std::thread::id{});
    EXPECT_NE(callback_thread_id, std::thread::id{});
    EXPECT_NE(callback_thread_id, inference_thread_id);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `scripts/build && scripts/test -R AgentRuntimeTest.StreamingCallbackRunsOffInferenceThread`
Expected: FAIL — callback currently runs on the inference thread so `callback_thread_id == inference_thread_id`.

- [ ] **Step 3: Add CallbackDispatcher to AgentRuntime**

In `include/zoo/internal/agent/runtime.hpp`, add include and member:

Add after the `#include "mailbox.hpp"` line:

```cpp
#include "callback_dispatcher.hpp"
```

Add a new member after `std::atomic<bool> tool_grammar_active_{false};` (line 93):

```cpp
CallbackDispatcher callback_dispatcher_;
```

- [ ] **Step 4: Update runtime_tool_loop.cpp to dispatch callbacks**

In `src/agent/runtime_tool_loop.cpp`, modify the token callback lambda (lines 69-79). Replace:

```cpp
    auto callback = [&](std::string_view token) -> TokenAction {
        if (request.streaming_callback && *request.streaming_callback) {
            (*request.streaming_callback)(token);
        }
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        return TokenAction::Continue;
    };
```

With:

```cpp
    auto callback = [&](std::string_view token) -> TokenAction {
        if (request.streaming_callback && *request.streaming_callback) {
            callback_dispatcher_.dispatch(*request.streaming_callback, token);
        }
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        return TokenAction::Continue;
    };
```

Then add a drain call after generation completes. After line 88 (`return std::unexpected(generated.error());` / after the `generated` error check), add a drain. Specifically, insert after line 88 and before line 90:

```cpp
    callback_dispatcher_.drain();
```

Also add a drain before the final return (before line 245, inside the `response_text` non-empty path). After `backend_->finalize_response();` (line 219) and before building the response:

```cpp
    callback_dispatcher_.drain();
```

Also add a drain before `continue` after tool execution (before line 207):

```cpp
    callback_dispatcher_.drain();
```

And before `continue` after validation failure (before line 181):

```cpp
    callback_dispatcher_.drain();
```

And before `continue` after the empty response nudge (before line 213):

```cpp
    callback_dispatcher_.drain();
```

- [ ] **Step 5: Update runtime_extraction.cpp to dispatch callbacks**

In `src/agent/runtime_extraction.cpp`, modify the callback lambda (lines 82-92). Replace:

```cpp
    auto callback = [&](std::string_view token) -> TokenAction {
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        if (request.streaming_callback && *request.streaming_callback) {
            (*request.streaming_callback)(token);
        }
        return TokenAction::Continue;
    };
```

With:

```cpp
    auto callback = [&](std::string_view token) -> TokenAction {
        if (!first_token_received) {
            first_token_time = std::chrono::steady_clock::now();
            first_token_received = true;
        }
        ++completion_tokens;
        if (request.streaming_callback && *request.streaming_callback) {
            callback_dispatcher_.dispatch(*request.streaming_callback, token);
        }
        return TokenAction::Continue;
    };
```

Add a drain after generation completes. After the `generated` error check (after line 101), add:

```cpp
    callback_dispatcher_.drain();
```

- [ ] **Step 6: Run test to verify it passes**

Run: `scripts/test -R AgentRuntimeTest.StreamingCallbackRunsOffInferenceThread`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `scripts/test`
Expected: All tests pass. The existing `ChatStreamingCallbackSurvivesTokenStreaming` test should still pass — callbacks are delivered in order, just on a different thread.

- [ ] **Step 8: Commit**

```bash
git add include/zoo/internal/agent/runtime.hpp src/agent/runtime_tool_loop.cpp src/agent/runtime_extraction.cpp tests/unit/test_agent_runtime.cpp
git commit -m "perf: offload streaming callbacks to dedicated dispatcher thread

Streaming callbacks now execute on the CallbackDispatcher thread instead
of the inference thread. drain() is called at synchronization points
(after generation, before tool execution, before response assembly) to
ensure all callbacks complete before the next operation."
```

---

### Task 6: Format and final validation

- [ ] **Step 1: Format all changed files**

Run: `scripts/format`

- [ ] **Step 2: Build clean**

Run: `scripts/build`
Expected: Clean build, no warnings.

- [ ] **Step 3: Run full test suite**

Run: `scripts/test`
Expected: All tests pass.

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -A
git commit -m "style: format concurrency improvement changes"
```
