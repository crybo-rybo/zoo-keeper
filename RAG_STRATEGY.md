# RAG Strategy for Zoo-Keeper

## Current Implementation (Now)

Zoo-Keeper now supports per-request RAG with **ephemeral context injection**:

- `ChatOptions::rag.enabled` enables retrieval for a request.
- `ChatOptions::rag.context_override` injects caller-provided context directly.
- `Agent::set_retriever(...)` installs a retriever implementation.
- Retrieved chunks are exposed in `Response::rag_chunks`.
- Injected context is never persisted in long-term `HistoryManager` state.

The default retriever implementation is `engine::InMemoryRagStore`:

- In-process storage and retrieval.
- Deterministic lexical scoring.
- JSON save/load for persistence.
- No external service dependency.

## Recommended Production Storage

For production-grade retrieval quality and scale in C++:

1. Use **SQLite** as the canonical metadata/chunk store.
2. Use an ANN vector index (e.g. **HNSW**) for semantic nearest-neighbor search.
3. Optionally add SQLite FTS5 for hybrid lexical + semantic retrieval.

Suggested schema:

- `documents(id, source_uri, title, checksum, metadata_json, created_at)`
- `chunks(id, document_id, chunk_index, text, token_count, metadata_json)`
- `chunk_fts` (FTS5 virtual table for lexical fallback/rerank)

Vector index:

- Store vectors in HNSW index keyed by `chunk_id`.
- Persist index to disk and rebuild incrementally from `chunks` as needed.

## Why This Split

- SQLite gives durable, portable, queryable storage with transactional safety.
- HNSW gives fast ANN search for large chunk corpora.
- The separation keeps ingestion, filtering, and retrieval maintainable in C++.

## Migration Path from Current Code

1. Keep `engine::IRetriever` as the stable runtime retrieval interface.
2. Add a new `SqliteHnswRetriever : IRetriever`.
3. Reuse `ChatOptions` and `AgenticLoop` unchanged.
4. Keep ephemeral injection semantics exactly as-is to satisfy FR-403.
