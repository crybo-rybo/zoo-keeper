#include <gtest/gtest.h>

#include "zoo/engine/context_database.hpp"
#include "zoo/engine/agentic_loop.hpp"
#include "zoo/engine/history_manager.hpp"
#include "mocks/mock_backend.hpp"

#include <chrono>
#include <filesystem>

using namespace zoo;
using namespace zoo::engine;
using namespace zoo::testing;

namespace {

std::filesystem::path make_temp_db_path(const std::string& prefix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (prefix + "_" + std::to_string(now) + ".sqlite");
}

} // namespace

TEST(ContextDatabaseTest, PersistAndRetrieve) {
    const auto db_path = make_temp_db_path("zoo_context_db_test");

    auto db_result = ContextDatabase::open(db_path.string());
    ASSERT_TRUE(db_result.has_value());
    auto db = *db_result;

    ASSERT_TRUE(db->add_message(Message::user("My project codename is Cedar."), "conversation").has_value());
    ASSERT_TRUE(db->add_message(Message::assistant("Acknowledged. The codename is Cedar."), "conversation").has_value());

    auto retrieval = db->retrieve(RagQuery{"What is the project codename?", 3});
    ASSERT_TRUE(retrieval.has_value());
    ASSERT_FALSE(retrieval->empty());

    db.reset();

    auto reopened_result = ContextDatabase::open(db_path.string());
    ASSERT_TRUE(reopened_result.has_value());
    auto reopened = *reopened_result;

    auto count_result = reopened->size();
    ASSERT_TRUE(count_result.has_value());
    EXPECT_GE(*count_result, 2U);

    auto retrieval_after_reopen = reopened->retrieve(RagQuery{"codename cedar", 2});
    ASSERT_TRUE(retrieval_after_reopen.has_value());
    ASSERT_FALSE(retrieval_after_reopen->empty());

    bool found_cedar = false;
    for (const auto& chunk : *retrieval_after_reopen) {
        if (chunk.content.find("Cedar") != std::string::npos) {
            found_cedar = true;
            break;
        }
    }
    EXPECT_TRUE(found_cedar);

    std::error_code ec;
    std::filesystem::remove(db_path, ec);
}

TEST(ContextDatabaseIntegrationTest, PrunesAndRetrievesArchivedContext) {
    auto backend = std::make_shared<MockBackend>();
    Config config;
    config.model_path = "/path/to/model.gguf";
    // Use a larger context size to accommodate per-message template overhead (8 tokens/msg)
    // while still being small enough to trigger pruning after several turns.
    config.context_size = 256;
    config.max_tokens = 64;
    ASSERT_TRUE(backend->initialize(config).has_value());

    auto history = std::make_shared<HistoryManager>(config.context_size);
    auto loop = std::make_unique<AgenticLoop>(backend, history, config);

    const auto db_path = make_temp_db_path("zoo_context_db_integration");
    auto db_result = ContextDatabase::open(db_path.string());
    ASSERT_TRUE(db_result.has_value());
    auto db = *db_result;
    loop->set_context_database(db);

    backend->enqueue_response("Stored that detail.");
    auto first = loop->process_request(Request(Message::user("Remember this exactly: launch-code zebra42.")));
    ASSERT_TRUE(first.has_value());

    for (int i = 0; i < 7; ++i) {
        backend->enqueue_response("Filler response " + std::to_string(i));
        auto result = loop->process_request(Request(Message::user(
            "Filler turn " + std::to_string(i) + " with enough text to pressure the context window.")));
        ASSERT_TRUE(result.has_value());
    }

    auto count_result = db->size();
    ASSERT_TRUE(count_result.has_value());
    EXPECT_GT(*count_result, 0U);

    backend->enqueue_response("The launch code is zebra42.");
    auto recall = loop->process_request(Request(Message::user("What is the launch code?")));
    ASSERT_TRUE(recall.has_value());

    EXPECT_NE(backend->last_formatted_prompt.find("Retrieved Context"), std::string::npos);
    EXPECT_NE(backend->last_formatted_prompt.find("zebra42"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(db_path, ec);
}
