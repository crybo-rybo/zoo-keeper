# RAG Retrieval

Zoo-Keeper supports per-request Retrieval-Augmented Generation (RAG) with ephemeral context injection. Retrieved content is used for the current turn only and is never stored in long-term conversation history.

## Overview

There are two ways to inject retrieval context:

1. **Automatic retrieval** via a configured `IRetriever` implementation
2. **Manual override** by providing precomputed context in `ChatOptions`

## Using `InMemoryRagStore`

The built-in `InMemoryRagStore` provides in-process lexical retrieval with no external dependencies.

### Adding Documents

```cpp
auto store = std::make_shared<zoo::engine::InMemoryRagStore>();

// Add individual chunks
store->add_chunk({"chunk-1", "Zoo-Keeper is a C++17 agent engine.", "docs"});
store->add_chunk({"chunk-2", "It supports tool calling and RAG.", "docs"});

// Or ingest a full document with automatic chunking
std::string document = /* load file contents */;
store->add_document(
    "readme",        // source identifier
    document,        // full text
    800,             // chunk size (chars)
    120              // overlap (chars)
);

// Install the retriever
agent->set_retriever(store);
```

### Persistence

Save and load the store to/from JSON:

```cpp
store->save("rag_store.json");
store->load("rag_store.json");
```

### Retrieval Scoring

`InMemoryRagStore` uses lexical cosine similarity:
- Text is tokenized into lowercase alphanumeric terms
- An inverted index maps terms to chunk indices
- Query-chunk overlap is normalized by `sqrt(|query_terms| * |chunk_terms|)`
- Top-k chunks are returned sorted by score

## Per-Request RAG Options

Enable RAG on a per-request basis via `ChatOptions`:

```cpp
zoo::ChatOptions options;
options.rag.enabled = true;
options.rag.top_k = 4;  // number of chunks to retrieve

auto future = agent->chat(zoo::Message::user("Tell me about Zoo-Keeper"), options);
```

### Context Override

Bypass the retriever and inject your own context:

```cpp
zoo::ChatOptions options;
options.rag.enabled = true;
options.rag.context_override = "Relevant information: Zoo-Keeper uses llama.cpp...";

auto future = agent->chat(zoo::Message::user("Summarize this"), options);
```

## RagOptions Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable RAG for this request |
| `top_k` | `int` | `4` | Number of chunks to retrieve |
| `context_override` | `optional<string>` | empty | Precomputed context to inject directly |

## How Injection Works

1. The agentic loop queries the retriever with the user's message
2. Retrieved chunks are formatted into a system message with instructions:
   ```
   Use the following retrieved context when relevant.
   If the context is insufficient, say what is missing.

   Retrieved Context:
   [1] source=docs chunk_id=chunk-1
   Zoo-Keeper is a C++17 agent engine.
   ```
3. This message is injected ephemerally before the user's message
4. After inference, the KV cache is cleared to prevent stale ephemeral state

Retrieved chunks are available in `Response::rag_chunks` for provenance tracking.

## Custom Retriever

Implement the `IRetriever` interface for custom backends (vector databases, APIs, etc.):

```cpp
class MyRetriever : public zoo::engine::IRetriever {
public:
    zoo::Expected<std::vector<zoo::RagChunk>> retrieve(
        const zoo::engine::RagQuery& query
    ) override {
        // query.text  -- the user's message
        // query.top_k -- number of results requested
        std::vector<zoo::RagChunk> results;
        // ... your retrieval logic ...
        return results;
    }
};

agent->set_retriever(std::make_shared<MyRetriever>());
```

## See Also

- [Context Database](context-database.md) -- SQLite-backed long-term memory (also implements `IRetriever`)
- [Getting Started](getting-started.md) -- basic Agent setup
- [Examples](examples.md) -- complete RAG usage snippets
