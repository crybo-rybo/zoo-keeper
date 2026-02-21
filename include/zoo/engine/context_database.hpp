#pragma once

#include "../types.hpp"
#include "rag_store.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <memory>
#include <mutex>
#include <sqlite3.h>
#include <string>
#include <vector>

namespace zoo {
namespace engine {

/**
 * @brief Durable SQLite-backed conversation memory used for long-context retrieval.
 *
 * Stores archived conversation messages and provides lexical retrieval
 * through SQLite FTS5 when available.
 */
class ContextDatabase : public IRetriever {
public:
    ~ContextDatabase() {
        // Finalize all cached prepared statements before closing the database.
        if (stmt_insert_ != nullptr) {
            sqlite3_finalize(stmt_insert_);
            stmt_insert_ = nullptr;
        }
        if (stmt_fts_insert_ != nullptr) {
            sqlite3_finalize(stmt_fts_insert_);
            stmt_fts_insert_ = nullptr;
        }
        if (stmt_fts_select_ != nullptr) {
            sqlite3_finalize(stmt_fts_select_);
            stmt_fts_select_ = nullptr;
        }
        if (stmt_size_ != nullptr) {
            sqlite3_finalize(stmt_size_);
            stmt_size_ = nullptr;
        }

        if (db_ != nullptr) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }

    ContextDatabase(const ContextDatabase&) = delete;
    ContextDatabase& operator=(const ContextDatabase&) = delete;

    static Expected<std::shared_ptr<ContextDatabase>> open(const std::string& path) {
        if (path.empty()) {
            return tl::unexpected(Error{
                ErrorCode::InvalidConfig,
                "Context database path cannot be empty"
            });
        }

        sqlite3* db = nullptr;
        if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) {
            std::string message = "Failed to open context database";
            if (db != nullptr && sqlite3_errmsg(db) != nullptr) {
                message += std::string(": ") + sqlite3_errmsg(db);
            }
            if (db != nullptr) {
                sqlite3_close(db);
            }
            return tl::unexpected(Error{ErrorCode::Unknown, std::move(message), path});
        }

        auto instance = std::shared_ptr<ContextDatabase>(new ContextDatabase(db, path));
        auto init_result = instance->initialize_schema();
        if (!init_result) {
            return tl::unexpected(init_result.error());
        }

        return instance;
    }

    Expected<void> add_message(const Message& message, std::optional<std::string> source = std::nullopt) {
        if (message.content.empty()) {
            return {};
        }

        std::lock_guard<std::mutex> lock(mutex_);

        const auto created_at = static_cast<sqlite3_int64>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());

        const char* role = role_to_string(message.role);
        sqlite3_bind_text(stmt_insert_, 1, role, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt_insert_, 2, message.content.c_str(), -1, SQLITE_TRANSIENT);

        if (source.has_value()) {
            sqlite3_bind_text(stmt_insert_, 3, source->c_str(), -1, SQLITE_TRANSIENT);
        } else {
            sqlite3_bind_null(stmt_insert_, 3);
        }
        sqlite3_bind_int64(stmt_insert_, 4, created_at);

        if (sqlite3_step(stmt_insert_) != SQLITE_DONE) {
            sqlite3_reset(stmt_insert_);
            sqlite3_clear_bindings(stmt_insert_);
            return tl::unexpected(make_sql_error("Failed to insert message into context database"));
        }
        sqlite3_reset(stmt_insert_);
        sqlite3_clear_bindings(stmt_insert_);

        const sqlite3_int64 row_id = sqlite3_last_insert_rowid(db_);

        if (fts_enabled_ && stmt_fts_insert_ != nullptr) {
            sqlite3_bind_int64(stmt_fts_insert_, 1, row_id);
            sqlite3_bind_text(stmt_fts_insert_, 2, message.content.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt_fts_insert_) != SQLITE_DONE) {
                sqlite3_reset(stmt_fts_insert_);
                sqlite3_clear_bindings(stmt_fts_insert_);
                return tl::unexpected(make_sql_error("Failed to update context FTS index"));
            }
            sqlite3_reset(stmt_fts_insert_);
            sqlite3_clear_bindings(stmt_fts_insert_);
        }

        return {};
    }

    Expected<void> add_messages(
        const std::vector<Message>& messages,
        std::optional<std::string> source = std::nullopt
    ) {
        for (const auto& message : messages) {
            auto result = add_message(message, source);
            if (!result) {
                return tl::unexpected(result.error());
            }
        }
        return {};
    }

    Expected<std::vector<RagChunk>> retrieve(const RagQuery& query) override {
        const int top_k = std::max(1, query.top_k);
        auto terms = tokenize_terms(query.text);
        if (terms.empty()) {
            return std::vector<RagChunk>{};
        }

        std::lock_guard<std::mutex> lock(mutex_);

        if (fts_enabled_) {
            auto fts_results = retrieve_with_fts(terms, top_k);
            if (fts_results) {
                return fts_results;
            }
        }

        return retrieve_with_like(terms, top_k);
    }

    Expected<size_t> size() const {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t count = 0;
        if (sqlite3_step(stmt_size_) == SQLITE_ROW) {
            count = static_cast<size_t>(sqlite3_column_int64(stmt_size_, 0));
        }
        sqlite3_reset(stmt_size_);
        return count;
    }

private:
    ContextDatabase(sqlite3* db, std::string db_path)
        : db_(db)
        , db_path_(std::move(db_path))
    {}

    Expected<void> initialize_schema() {
        auto exec = [this](const char* sql) -> Expected<void> {
            char* err_msg = nullptr;
            if (sqlite3_exec(db_, sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
                std::string message = err_msg != nullptr ? err_msg : "Unknown SQLite error";
                sqlite3_free(err_msg);
                return tl::unexpected(Error{ErrorCode::Unknown, std::move(message), db_path_});
            }
            return {};
        };

        auto table_result = exec(
            "CREATE TABLE IF NOT EXISTS memory_messages("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "role TEXT NOT NULL,"
            "content TEXT NOT NULL,"
            "source TEXT,"
            "created_at INTEGER NOT NULL"
            ")");
        if (!table_result) {
            return table_result;
        }

        auto fts_result = exec(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
            "message_id UNINDEXED,"
            "content"
            ")");
        if (fts_result) {
            fts_enabled_ = true;
            auto sync_result = sync_fts_from_messages();
            if (!sync_result) {
                return sync_result;
            }
        } else {
            fts_enabled_ = false;
        }

        // Prepare cached statements that are always needed.
        constexpr const char* insert_sql =
            "INSERT INTO memory_messages(role, content, source, created_at) VALUES (?1, ?2, ?3, ?4)";
        if (sqlite3_prepare_v2(db_, insert_sql, -1, &stmt_insert_, nullptr) != SQLITE_OK) {
            return tl::unexpected(make_sql_error("Failed to prepare cached insert statement"));
        }

        constexpr const char* size_sql = "SELECT COUNT(*) FROM memory_messages";
        if (sqlite3_prepare_v2(db_, size_sql, -1, &stmt_size_, nullptr) != SQLITE_OK) {
            return tl::unexpected(make_sql_error("Failed to prepare cached size statement"));
        }

        // Prepare FTS-specific cached statements only when FTS is enabled.
        if (fts_enabled_) {
            constexpr const char* fts_insert_sql =
                "INSERT INTO memory_fts(message_id, content) VALUES (?1, ?2)";
            if (sqlite3_prepare_v2(db_, fts_insert_sql, -1, &stmt_fts_insert_, nullptr) != SQLITE_OK) {
                return tl::unexpected(make_sql_error("Failed to prepare cached FTS insert statement"));
            }

            constexpr const char* fts_select_sql =
                "SELECT m.id, m.content, m.source, -bm25(memory_fts) AS score "
                "FROM memory_fts "
                "JOIN memory_messages m ON m.id = memory_fts.message_id "
                "WHERE memory_fts MATCH ?1 "
                "ORDER BY bm25(memory_fts), m.id DESC "
                "LIMIT ?2";
            if (sqlite3_prepare_v2(db_, fts_select_sql, -1, &stmt_fts_select_, nullptr) != SQLITE_OK) {
                return tl::unexpected(make_sql_error("Failed to prepare cached FTS select statement"));
            }
        }

        return {};
    }

    Expected<void> sync_fts_from_messages() {
        if (!fts_enabled_) {
            return {};
        }

        char* err_msg = nullptr;
        if (sqlite3_exec(db_, "DELETE FROM memory_fts", nullptr, nullptr, &err_msg) != SQLITE_OK) {
            std::string message = err_msg != nullptr ? err_msg : "Unknown SQLite error";
            sqlite3_free(err_msg);
            return tl::unexpected(Error{ErrorCode::Unknown, std::move(message), db_path_});
        }

        constexpr const char* sql =
            "INSERT INTO memory_fts(message_id, content) "
            "SELECT id, content FROM memory_messages";
        if (sqlite3_exec(db_, sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
            std::string message = err_msg != nullptr ? err_msg : "Unknown SQLite error";
            sqlite3_free(err_msg);
            return tl::unexpected(Error{ErrorCode::Unknown, std::move(message), db_path_});
        }

        return {};
    }

    Expected<std::vector<RagChunk>> retrieve_with_fts(
        const std::vector<std::string>& terms,
        int top_k
    ) const {
        std::string fts_query;
        for (size_t i = 0; i < terms.size(); ++i) {
            if (i > 0) {
                fts_query += " OR ";
            }
            fts_query += terms[i];
        }

        sqlite3_bind_text(stmt_fts_select_, 1, fts_query.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt_fts_select_, 2, top_k);

        auto rows = read_rows_as_chunks(stmt_fts_select_);
        sqlite3_reset(stmt_fts_select_);
        sqlite3_clear_bindings(stmt_fts_select_);
        return rows;
    }

    Expected<std::vector<RagChunk>> retrieve_with_like(
        const std::vector<std::string>& terms,
        int top_k
    ) const {
        // Build a dynamic SQL query with one LIKE clause per term joined with OR.
        std::string sql = "SELECT id, content, source FROM memory_messages WHERE ";
        for (size_t i = 0; i < terms.size(); ++i) {
            if (i > 0) {
                sql += " OR ";
            }
            sql += "content LIKE ?" + std::to_string(i + 1);
        }
        sql += " ORDER BY id DESC LIMIT ?" + std::to_string(terms.size() + 1);

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            return tl::unexpected(make_sql_error("Failed to prepare fallback memory query"));
        }

        // Bind each term as a LIKE pattern.
        for (size_t i = 0; i < terms.size(); ++i) {
            std::string pattern = "%" + terms[i] + "%";
            sqlite3_bind_text(stmt, static_cast<int>(i + 1), pattern.c_str(), -1, SQLITE_TRANSIENT);
        }
        sqlite3_bind_int(stmt, static_cast<int>(terms.size() + 1), top_k);

        std::vector<RagChunk> chunks;
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const sqlite3_int64 id = sqlite3_column_int64(stmt, 0);
            const char* content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const unsigned char* source_raw = sqlite3_column_text(stmt, 2);

            RagChunk chunk;
            chunk.id = "memory:" + std::to_string(id);
            chunk.content = content != nullptr ? content : "";
            chunk.score = 0.0;
            if (source_raw != nullptr) {
                chunk.source = std::string(reinterpret_cast<const char*>(source_raw));
            } else {
                chunk.source = std::string("context_db");
            }
            chunks.push_back(std::move(chunk));
        }

        sqlite3_finalize(stmt);
        return chunks;
    }

    Expected<std::vector<RagChunk>> read_rows_as_chunks(sqlite3_stmt* stmt) const {
        std::vector<RagChunk> chunks;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const sqlite3_int64 id = sqlite3_column_int64(stmt, 0);
            const char* content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const unsigned char* source_raw = sqlite3_column_text(stmt, 2);
            const double score = sqlite3_column_double(stmt, 3);

            RagChunk chunk;
            chunk.id = "memory:" + std::to_string(id);
            chunk.content = content != nullptr ? content : "";
            chunk.score = score;
            if (source_raw != nullptr) {
                chunk.source = std::string(reinterpret_cast<const char*>(source_raw));
            } else {
                chunk.source = std::string("context_db");
            }
            chunks.push_back(std::move(chunk));
        }

        return chunks;
    }

    static std::vector<std::string> tokenize_terms(const std::string& text) {
        std::vector<std::string> terms;
        std::string current;
        current.reserve(32);

        for (unsigned char ch : text) {
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

    Error make_sql_error(const std::string& prefix) const {
        return Error{
            ErrorCode::Unknown,
            prefix + ": " + sqlite3_errmsg(db_),
            db_path_
        };
    }

    sqlite3* db_ = nullptr;
    std::string db_path_;
    bool fts_enabled_ = false;
    mutable std::mutex mutex_;

    // Cached prepared statements (prepared once in initialize_schema, finalized in destructor).
    sqlite3_stmt* stmt_insert_     = nullptr;
    sqlite3_stmt* stmt_fts_insert_ = nullptr;
    mutable sqlite3_stmt* stmt_fts_select_ = nullptr;
    mutable sqlite3_stmt* stmt_size_       = nullptr;
};

} // namespace engine
} // namespace zoo
