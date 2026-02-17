#include <gtest/gtest.h>
#include "zoo/engine/rag_store.hpp"
#include "zoo/engine/agentic_loop.hpp"
#include "zoo/engine/history_manager.hpp"
#include "mocks/mock_backend.hpp"
#include <filesystem>

using namespace zoo;
using namespace zoo::engine;
using namespace zoo::testing;

TEST(RagStoreTest, AddAndRetrieveChunks) {
    InMemoryRagStore store;

    auto add1 = store.add_chunk({"doc:0", "Paris is the capital of France.", "doc"});
    auto add2 = store.add_chunk({"doc:1", "Tokyo is the capital of Japan.", "doc"});

    ASSERT_TRUE(add1.has_value());
    ASSERT_TRUE(add2.has_value());

    auto result = store.retrieve({"What is the capital of France?", 2});
    ASSERT_TRUE(result.has_value());
    ASSERT_FALSE(result->empty());
    EXPECT_EQ((*result)[0].id, "doc:0");
}

TEST(RagStoreTest, SaveAndLoad) {
    InMemoryRagStore store;
    ASSERT_TRUE(store.add_chunk({"s:0", "Mercury is the closest planet to the Sun.", "solar"}).has_value());
    ASSERT_TRUE(store.add_chunk({"s:1", "Venus is the second planet from the Sun.", "solar"}).has_value());

    const auto path = std::filesystem::temp_directory_path() / "zoo_rag_store_test.json";
    ASSERT_TRUE(store.save(path.string()).has_value());

    InMemoryRagStore loaded;
    ASSERT_TRUE(loaded.load(path.string()).has_value());
    EXPECT_EQ(loaded.size(), 2U);

    auto result = loaded.retrieve({"closest planet to the sun", 1});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1U);
    EXPECT_EQ((*result)[0].id, "s:0");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(RagIntegrationTest, EphemeralContextInjectedWithoutHistoryPollution) {
    auto backend = std::make_shared<MockBackend>();
    Config config;
    config.model_path = "/path/to/model.gguf";
    config.context_size = 8192;
    config.max_tokens = 128;
    ASSERT_TRUE(backend->initialize(config).has_value());
    backend->enqueue_response("The Eiffel Tower is in Paris.");

    auto history = std::make_shared<HistoryManager>(config.context_size);
    auto loop = std::make_unique<AgenticLoop>(backend, history, config);

    auto retriever = std::make_shared<InMemoryRagStore>();
    ASSERT_TRUE(retriever->add_document(
        "wiki:eiffel",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    ).has_value());
    loop->set_retriever(retriever);

    ChatOptions options;
    options.rag.enabled = true;
    options.rag.top_k = 2;

    Request request(Message::user("Where is the Eiffel Tower?"), options);
    auto result = loop->process_request(request);

    ASSERT_TRUE(result.has_value());
    ASSERT_FALSE(result->rag_chunks.empty());
    EXPECT_NE(backend->last_formatted_prompt.find("Retrieved Context"), std::string::npos);
    EXPECT_NE(backend->last_formatted_prompt.find("wiki:eiffel"), std::string::npos);
    EXPECT_EQ(backend->clear_kv_cache_calls, 2);

    // Only user + assistant are persisted in history, not ephemeral RAG system context.
    const auto& messages = history->get_messages();
    ASSERT_EQ(messages.size(), 2U);
    EXPECT_EQ(messages[0].role, Role::User);
    EXPECT_EQ(messages[1].role, Role::Assistant);
    EXPECT_EQ(messages[1].content, "The Eiffel Tower is in Paris.");
}
