/**
 * @file test_batch.cpp
 * @brief Unit tests for prompt prefill chunk planning.
 */

#include "core/batch.hpp"
#include <gtest/gtest.h>

#include <utility>

using zoo::core::BatchChunk;
using zoo::core::compute_prefill_chunks;
using zoo::core::LlamaBatchHandle;

TEST(ComputePrefillChunksTest, SingleChunkFitsExactly) {
    auto chunks = compute_prefill_chunks(512, 512);
    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0], (BatchChunk{0, 512, true}));
}

TEST(ComputePrefillChunksTest, SingleChunkSmallerThanBatch) {
    auto chunks = compute_prefill_chunks(100, 512);
    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0], (BatchChunk{0, 100, true}));
}

TEST(ComputePrefillChunksTest, TwoEvenChunks) {
    auto chunks = compute_prefill_chunks(1024, 512);
    ASSERT_EQ(chunks.size(), 2);
    EXPECT_EQ(chunks[0], (BatchChunk{0, 512, false}));
    EXPECT_EQ(chunks[1], (BatchChunk{512, 512, true}));
}

TEST(ComputePrefillChunksTest, UnevenChunks) {
    auto chunks = compute_prefill_chunks(1000, 512);
    ASSERT_EQ(chunks.size(), 2);
    EXPECT_EQ(chunks[0], (BatchChunk{0, 512, false}));
    EXPECT_EQ(chunks[1], (BatchChunk{512, 488, true}));
}

TEST(ComputePrefillChunksTest, ManyChunks) {
    auto chunks = compute_prefill_chunks(2050, 512);
    ASSERT_EQ(chunks.size(), 5);
    EXPECT_EQ(chunks[0], (BatchChunk{0, 512, false}));
    EXPECT_EQ(chunks[1], (BatchChunk{512, 512, false}));
    EXPECT_EQ(chunks[2], (BatchChunk{1024, 512, false}));
    EXPECT_EQ(chunks[3], (BatchChunk{1536, 512, false}));
    EXPECT_EQ(chunks[4], (BatchChunk{2048, 2, true}));
}

TEST(ComputePrefillChunksTest, SingleToken) {
    auto chunks = compute_prefill_chunks(1, 512);
    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0], (BatchChunk{0, 1, true}));
}

TEST(ComputePrefillChunksTest, BatchSizeOne) {
    auto chunks = compute_prefill_chunks(3, 1);
    ASSERT_EQ(chunks.size(), 3);
    EXPECT_EQ(chunks[0], (BatchChunk{0, 1, false}));
    EXPECT_EQ(chunks[1], (BatchChunk{1, 1, false}));
    EXPECT_EQ(chunks[2], (BatchChunk{2, 1, true}));
}

TEST(ComputePrefillChunksTest, InvalidInputsReturnEmpty) {
    EXPECT_TRUE(compute_prefill_chunks(0, 512).empty());
    EXPECT_TRUE(compute_prefill_chunks(-1, 512).empty());
    EXPECT_TRUE(compute_prefill_chunks(100, 0).empty());
    EXPECT_TRUE(compute_prefill_chunks(100, -1).empty());
    EXPECT_TRUE(compute_prefill_chunks(0, 0).empty());
}

TEST(ComputePrefillChunksTest, OnlyLastChunkEmitsLogits) {
    auto chunks = compute_prefill_chunks(1500, 512);
    for (size_t i = 0; i < chunks.size() - 1; ++i) {
        EXPECT_FALSE(chunks[i].emit_logits) << "Chunk " << i << " should not emit logits";
    }
    EXPECT_TRUE(chunks.back().emit_logits);
}

TEST(ComputePrefillChunksTest, OffsetsAreContinuous) {
    auto chunks = compute_prefill_chunks(2000, 512);
    int expected_offset = 0;
    for (const auto& chunk : chunks) {
        EXPECT_EQ(chunk.offset, expected_offset);
        expected_offset += chunk.count;
    }
    EXPECT_EQ(expected_offset, 2000);
}

TEST(ComputePrefillChunksTest, AllCountsPositive) {
    auto chunks = compute_prefill_chunks(2000, 512);
    for (const auto& chunk : chunks) {
        EXPECT_GT(chunk.count, 0);
    }
}

TEST(ComputePrefillChunksTest, NoChunkExceedsBatchSize) {
    auto chunks = compute_prefill_chunks(2000, 512);
    for (const auto& chunk : chunks) {
        EXPECT_LE(chunk.count, 512);
    }
}

TEST(LlamaBatchHandleTest, MoveConstructionTransfersBatchState) {
    LlamaBatchHandle batch(2, 0, 1);
    batch.get().n_tokens = 1;
    batch.get().token[0] = 42;

    LlamaBatchHandle moved(std::move(batch));

    EXPECT_EQ(moved.get().n_tokens, 1);
    EXPECT_EQ(moved.get().token[0], 42);
}

TEST(LlamaBatchHandleTest, MoveAssignmentReleasesPreviousBatchAndTransfersState) {
    LlamaBatchHandle source(2, 0, 1);
    source.get().n_tokens = 1;
    source.get().token[0] = 7;

    LlamaBatchHandle target(1, 0, 1);
    target = std::move(source);

    EXPECT_EQ(target.get().n_tokens, 1);
    EXPECT_EQ(target.get().token[0], 7);
}
