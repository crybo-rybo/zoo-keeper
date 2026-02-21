# Project Evaluation: Potential Improvements
Based on a comprehensive review of the `zoo-keeper` project's core components (Engine, Backend, and MCP), here is a ranked list of potential improvements ordered from most to least impact.

> [!IMPORTANT]
> The findings are strictly based on the existing codebase implementation (excluding the `extern` folder as requested).

## 1. [Performance] SQLite Prepared Statements Uncached in `ContextDatabase`
**Location**: [context_database.hpp](file:///Users/conorrybacki/Programs/zoo-keeper/include/zoo/engine/context_database.hpp)
**Impact**: **Very High**
Currently, `sqlite3_prepare_v2` is called to compile SQL statements on *every single invocation* of `add_message` and `retrieve`. This completely defeats the performance benefits of prepared statements and introduces massive overhead on database operations. 
**Improvement**: Cache the prepared `sqlite3_stmt*` pointers as class members during database initialization and reset them (`sqlite3_reset`) after each use.

## 2. [Architecture / Performance] Naive Context Token Estimation
**Location**: [history_manager.hpp](file:///Users/conorrybacki/Programs/zoo-keeper/include/zoo/engine/history_manager.hpp)
**Impact**: **High**
The `HistoryManager` estimates context size using a highly simplistic heuristic: `text.length() / 4`. 
This is inaccurate and can lead to silently overflowing the LLM context window (causing API/inference errors) or drastically underutilizing the available context.
**Improvement**: Pass the `IBackend` (or a dedicated tokenizer instance) into the `HistoryManager` so it can accurately calculate token counts using `backend->tokenize()`.

## 3. [Architecture] Leaked Thread Synchronization in `HistoryManager`
**Location**: [agentic_loop.hpp](file:///Users/conorrybacki/Programs/zoo-keeper/include/zoo/engine/agentic_loop.hpp) & [history_manager.hpp](file:///Users/conorrybacki/Programs/zoo-keeper/include/zoo/engine/history_manager.hpp)
**Impact**: **High**
`HistoryManager` states it is single-threaded, but `AgenticLoop` accepts a raw `std::mutex* history_mutex_` and manually manages locking via `lock_history()` before accessing the history. This breaks encapsulation and is highly prone to race conditions if other threads access `HistoryManager` without knowing about the external mutex.
**Improvement**: Move the `std::mutex` inside `HistoryManager` and make its public methods thread-safe internally.

## 4. [Performance] String and Memory Copying Overhead in Tool Loops
**Location**: [agentic_loop.hpp](file:///Users/conorrybacki/Programs/zoo-keeper/include/zoo/engine/agentic_loop.hpp)
**Impact**: **Medium-High**
Inside `AgenticLoop::process_request`, the `build_prompt_messages` function creates entirely new `std::vector<Message>` copies and heavily concatenates massive `rag_context` strings on *every single tool iteration*. For long contexts executing multiple tool calls, this causes severe memory allocation overhead and heap fragmentation.
**Improvement**: Use `std::string_view` where possible or utilize a view-based abstraction for passing message context to the backend without deep copying the entire history array repeatedly.

## 5. [Code Robustness] Incomplete SQL Query in Fallback Retrieval
**Location**: [context_database.hpp#L295](file:///Users/conorrybacki/Programs/zoo-keeper/include/zoo/engine/context_database.hpp#L295)
**Impact**: **Medium**
In `retrieve_with_like` (used when FTS5 is disabled), the search pattern is constructed as:
```cpp
std::string pattern = "%";
pattern += terms.front();
pattern += "%";
```
This is a bug where the search query completely discards all search terms except the very first word (`terms.front()`). Multi-word fallback queries will yield extremely poor or irrelevant results.
**Improvement**: Iterate over all `terms` and dynamically construct the query to include `LIKE` clauses for all terms via `AND` or `OR`.

## 6. [Architecture / Robustness] Global API Initialization Tied to Instance Constructor
**Location**: [llama_backend.cpp#L7](file:///Users/conorrybacki/Programs/zoo-keeper/src/backend/llama_backend.cpp#L7)
**Impact**: **Medium**
The `LlamaBackend` constructor calls `llama_backend_init()` and `ggml_backend_load_all()`. Furthermore, the destructor intentionally omits `llama_backend_free()`.
While the `llama.cpp` initialization is idempotent, tying global process-wide state initialization to a specific class instance constructor is an anti-pattern. It prevents cleanly unloading the backend and hinders safe multi-instance initialization.
**Improvement**: Move global `llama.cpp` process initialization to a dedicated, process-level initialization routine (e.g., a `ZooKeeperConfig::init_globals()` function called once at startup).
