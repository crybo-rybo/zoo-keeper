# Context Database

The Context Database provides durable long-term memory for conversations that exceed the model's context window. It uses SQLite to archive pruned messages and retrieves relevant history on future turns.

## Enabling

```cpp
auto init = agent->enable_context_database("memory.sqlite");
if (!init) {
    std::cerr << init.error().to_string() << std::endl;
}
```

Or with manual control:

```cpp
auto db_result = zoo::engine::ContextDatabase::open("memory.sqlite");
if (db_result) {
    agent->set_context_database(std::move(*db_result));
}
```

## How It Works

When a context database is configured, the `AgenticLoop` automatically manages memory pressure:

1. **Detection**: Before each inference, the engine checks if estimated token usage exceeds the context window.
2. **Pruning**: Oldest user/assistant message pairs are removed from active history until token usage drops to ~75% of the context window (configurable via `memory_prune_target_ratio_`). A minimum of 6 messages is always kept in active history.
3. **Archival**: Pruned messages are written to the SQLite database with a `history_archive` source tag.
4. **Retrieval**: On each new turn, the engine queries the database using the current user message as a search query. Relevant archived messages are injected ephemerally (they do not pollute long-term history).
5. **KV Cache Reset**: After ephemeral injection, the backend's KV cache is cleared to ensure prompt consistency.

## Schema

The database creates two tables:

```sql
-- Main message storage
CREATE TABLE memory_messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    created_at INTEGER NOT NULL
);

-- Full-text search index (if FTS5 is available)
CREATE VIRTUAL TABLE memory_fts USING fts5(
    message_id UNINDEXED,
    content
);
```

## Retrieval

The `ContextDatabase` implements the `IRetriever` interface, so it integrates seamlessly with the RAG pipeline.

**With FTS5** (preferred): Uses SQLite's built-in BM25 ranking for relevance scoring.

**Without FTS5** (fallback): Uses `LIKE` queries on the first search term. FTS5 availability is detected at runtime.

Retrieved context is injected as an ephemeral system message immediately before the user's message, following the same pattern as RAG retrieval.

## When to Use

- **Long conversations** where important context would otherwise be lost to context window limits
- **Persistent assistants** that need to recall information across sessions (the SQLite file persists on disk)
- **Memory-constrained models** with small context windows

## Limitations

- Token counting uses character-based estimation (4 chars ~ 1 token)
- Retrieval quality depends on lexical overlap between the current query and archived messages
- No vector/semantic search (planned for future)

## See Also

- [RAG Retrieval](rag.md) -- ephemeral retrieval from external knowledge sources
- [Architecture](architecture.md) -- how pruning integrates with the agentic loop
- [Configuration](configuration.md) -- context size and related settings
