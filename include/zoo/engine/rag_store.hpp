#pragma once

#include "../types.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace zoo {
namespace engine {

/**
 * @brief Query parameters passed to retrievers.
 */
struct RagQuery {
    std::string text;     ///< Natural-language query text
    int top_k = 4;        ///< Number of chunks to retrieve
};

/**
 * @brief Interface for pluggable RAG retrievers.
 */
class IRetriever {
public:
    virtual ~IRetriever() = default;
    virtual Expected<std::vector<RagChunk>> retrieve(const RagQuery& query) = 0;
};

/**
 * @brief In-memory retriever with JSON persistence and lexical scoring.
 *
 * This implementation is intentionally dependency-light:
 * - Embedded in-process storage
 * - No external services required
 * - File persistence via JSON
 *
 * It provides a deterministic baseline retriever and a clean abstraction
 * boundary for production vector backends (e.g., SQLite + HNSW) later.
 */
class InMemoryRagStore : public IRetriever {
public:
    struct ChunkRecord {
        std::string id;
        std::string content;
        std::optional<std::string> source;
    };

    /**
     * @brief Add a chunk record and update retrieval index.
     */
    Expected<void> add_chunk(ChunkRecord record) {
        if (record.id.empty()) {
            return tl::unexpected(Error{
                ErrorCode::InvalidConfig,
                "RAG chunk id cannot be empty"
            });
        }
        if (record.content.empty()) {
            return tl::unexpected(Error{
                ErrorCode::InvalidConfig,
                "RAG chunk content cannot be empty"
            });
        }

        const auto it = id_to_index_.find(record.id);
        if (it != id_to_index_.end()) {
            erase_terms_for_index(it->second);
            chunks_[it->second] = std::move(record);
            index_terms_for_index(it->second);
            return {};
        }

        const size_t idx = chunks_.size();
        id_to_index_[record.id] = idx;
        chunks_.push_back(std::move(record));
        chunk_terms_.emplace_back();
        index_terms_for_index(idx);
        return {};
    }

    /**
     * @brief Split a document into overlapping chunks and index them.
     */
    Expected<void> add_document(
        const std::string& source_id,
        const std::string& text,
        size_t chunk_size_chars = 800,
        size_t overlap_chars = 120
    ) {
        if (source_id.empty()) {
            return tl::unexpected(Error{
                ErrorCode::InvalidConfig,
                "RAG source_id cannot be empty"
            });
        }
        if (text.empty()) {
            return tl::unexpected(Error{
                ErrorCode::InvalidConfig,
                "RAG document text cannot be empty"
            });
        }
        if (chunk_size_chars == 0 || overlap_chars >= chunk_size_chars) {
            return tl::unexpected(Error{
                ErrorCode::InvalidConfig,
                "Invalid chunking settings for RAG document ingest"
            });
        }

        size_t pos = 0;
        size_t chunk_index = 0;
        const size_t step = chunk_size_chars - overlap_chars;
        while (pos < text.size()) {
            const size_t len = std::min(chunk_size_chars, text.size() - pos);
            ChunkRecord record;
            record.id = source_id + ":" + std::to_string(chunk_index++);
            record.content = text.substr(pos, len);
            record.source = source_id;

            auto add_result = add_chunk(std::move(record));
            if (!add_result) {
                return tl::unexpected(add_result.error());
            }

            if (pos + len >= text.size()) {
                break;
            }
            pos += step;
        }

        return {};
    }

    /**
     * @brief Save all chunks to disk as JSON.
     */
    Expected<void> save(const std::string& path) const {
        nlohmann::json root;
        root["chunks"] = nlohmann::json::array();
        for (const auto& chunk : chunks_) {
            root["chunks"].push_back(nlohmann::json{
                {"id", chunk.id},
                {"content", chunk.content},
                {"source", chunk.source.value_or("")}
            });
        }

        std::ofstream out(path);
        if (!out.is_open()) {
            return tl::unexpected(Error{
                ErrorCode::Unknown,
                "Failed to open RAG store file for writing",
                path
            });
        }
        out << root.dump(2);
        return {};
    }

    /**
     * @brief Load chunks from disk and rebuild index.
     */
    Expected<void> load(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            return tl::unexpected(Error{
                ErrorCode::Unknown,
                "Failed to open RAG store file for reading",
                path
            });
        }

        nlohmann::json root;
        try {
            in >> root;
        } catch (const std::exception& e) {
            return tl::unexpected(Error{
                ErrorCode::Unknown,
                std::string("Failed to parse RAG store JSON: ") + e.what(),
                path
            });
        }

        if (!root.contains("chunks") || !root["chunks"].is_array()) {
            return tl::unexpected(Error{
                ErrorCode::Unknown,
                "Invalid RAG store JSON: missing 'chunks' array",
                path
            });
        }

        clear();
        for (const auto& item : root["chunks"]) {
            if (!item.contains("id") || !item.contains("content")) {
                continue;
            }
            ChunkRecord record;
            record.id = item["id"].get<std::string>();
            record.content = item["content"].get<std::string>();
            if (item.contains("source")) {
                const std::string source = item["source"].get<std::string>();
                if (!source.empty()) {
                    record.source = source;
                }
            }

            auto add_result = add_chunk(std::move(record));
            if (!add_result) {
                return tl::unexpected(add_result.error());
            }
        }

        return {};
    }

    /**
     * @brief Retrieve top-k chunks using simple lexical cosine score.
     */
    Expected<std::vector<RagChunk>> retrieve(const RagQuery& query) override {
        const auto query_terms = tokenize_terms(query.text);
        if (query_terms.empty()) {
            return std::vector<RagChunk>{};
        }

        const int top_k = std::max(1, query.top_k);

        std::unordered_map<size_t, int> overlap_count;
        for (const auto& term : query_terms) {
            const auto postings_it = inverted_index_.find(term);
            if (postings_it == inverted_index_.end()) {
                continue;
            }
            for (const size_t idx : postings_it->second) {
                overlap_count[idx] += 1;
            }
        }

        struct Candidate {
            size_t idx = 0;
            double score = std::numeric_limits<double>::lowest();
        };

        std::vector<Candidate> candidates;
        candidates.reserve(overlap_count.size());
        for (const auto& [idx, overlap] : overlap_count) {
            const double denom = std::sqrt(
                static_cast<double>(query_terms.size()) *
                static_cast<double>(std::max<size_t>(1, chunk_terms_[idx].size())));
            const double score = (denom > 0.0) ? (static_cast<double>(overlap) / denom) : 0.0;
            candidates.push_back(Candidate{idx, score});
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.score == b.score) {
                return a.idx < b.idx;
            }
            return a.score > b.score;
        });

        std::vector<RagChunk> results;
        results.reserve(static_cast<size_t>(top_k));
        for (const auto& c : candidates) {
            if (static_cast<int>(results.size()) >= top_k) {
                break;
            }
            const auto& chunk = chunks_[c.idx];
            results.push_back(RagChunk{
                chunk.id,
                chunk.content,
                c.score,
                chunk.source
            });
        }

        return results;
    }

    /**
     * @brief Remove all indexed data.
     */
    void clear() {
        chunks_.clear();
        chunk_terms_.clear();
        id_to_index_.clear();
        inverted_index_.clear();
    }

    /**
     * @brief Number of indexed chunks.
     */
    size_t size() const {
        return chunks_.size();
    }

private:
    std::vector<ChunkRecord> chunks_;
    std::vector<std::unordered_set<std::string>> chunk_terms_;
    std::unordered_map<std::string, size_t> id_to_index_;
    std::unordered_map<std::string, std::vector<size_t>> inverted_index_;

    static std::vector<std::string> tokenize_terms(const std::string& text) {
        std::vector<std::string> terms;
        std::string current;
        current.reserve(32);

        for (const unsigned char ch : text) {
            if (std::isalnum(ch) != 0) {
                current.push_back(static_cast<char>(std::tolower(ch)));
            } else if (!current.empty()) {
                terms.push_back(current);
                current.clear();
            }
        }
        if (!current.empty()) {
            terms.push_back(current);
        }

        std::sort(terms.begin(), terms.end());
        terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
        return terms;
    }

    void erase_terms_for_index(size_t idx) {
        if (idx >= chunk_terms_.size()) {
            return;
        }
        for (const auto& term : chunk_terms_[idx]) {
            auto it = inverted_index_.find(term);
            if (it == inverted_index_.end()) {
                continue;
            }
            auto& postings = it->second;
            postings.erase(std::remove(postings.begin(), postings.end(), idx), postings.end());
            if (postings.empty()) {
                inverted_index_.erase(it);
            }
        }
        chunk_terms_[idx].clear();
    }

    void index_terms_for_index(size_t idx) {
        if (idx >= chunks_.size()) {
            return;
        }
        auto terms = tokenize_terms(chunks_[idx].content);
        auto& term_set = chunk_terms_[idx];
        for (const auto& term : terms) {
            term_set.insert(term);
            inverted_index_[term].push_back(idx);
        }
    }
};

} // namespace engine
} // namespace zoo
