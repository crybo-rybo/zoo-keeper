/**
 * @file test_prompt_bookkeeping.cpp
 * @brief Unit tests for incremental prompt bookkeeping rules.
 */

#include "zoo/internal/core/prompt_bookkeeping.hpp"
#include <gtest/gtest.h>

using zoo::core::PromptHistoryMutation;
using zoo::core::commit_rendered_prompt;
using zoo::core::history_mutation_requires_kv_reset;
using zoo::core::note_history_mutation;
using zoo::core::rendered_prompt_requires_kv_reset;

TEST(PromptBookkeepingTest, AppendKeepsCommittedPromptLength) {
    bool cached_messages_dirty = false;
    int committed_prompt_len = 48;

    note_history_mutation(PromptHistoryMutation::Append, cached_messages_dirty,
                          committed_prompt_len);

    EXPECT_TRUE(cached_messages_dirty);
    EXPECT_EQ(committed_prompt_len, 48);
    EXPECT_FALSE(history_mutation_requires_kv_reset(PromptHistoryMutation::Append));
}

TEST(PromptBookkeepingTest, SameSizeRewriteInvalidatesCachedMessagesAndCheckpoint) {
    bool cached_messages_dirty = false;
    int committed_prompt_len = 48;

    note_history_mutation(PromptHistoryMutation::Rewrite, cached_messages_dirty,
                          committed_prompt_len);

    EXPECT_TRUE(cached_messages_dirty);
    EXPECT_EQ(committed_prompt_len, 0);
    EXPECT_TRUE(history_mutation_requires_kv_reset(PromptHistoryMutation::Rewrite));
}

TEST(PromptBookkeepingTest, ResetClearsCommittedPromptLength) {
    bool cached_messages_dirty = false;
    int committed_prompt_len = 64;

    note_history_mutation(PromptHistoryMutation::Reset, cached_messages_dirty, committed_prompt_len);

    EXPECT_TRUE(cached_messages_dirty);
    EXPECT_EQ(committed_prompt_len, 0);
    EXPECT_TRUE(history_mutation_requires_kv_reset(PromptHistoryMutation::Reset));
}

TEST(PromptBookkeepingTest, ShorterRenderedPromptRequiresKvReset) {
    EXPECT_TRUE(rendered_prompt_requires_kv_reset(32, 31));
    EXPECT_FALSE(rendered_prompt_requires_kv_reset(32, 32));
    EXPECT_FALSE(rendered_prompt_requires_kv_reset(32, 40));
}

TEST(PromptBookkeepingTest, CommitRenderedPromptAdvancesOnlyOnSuccessfulRender) {
    int committed_prompt_len = 12;

    commit_rendered_prompt(committed_prompt_len, 48);
    EXPECT_EQ(committed_prompt_len, 48);

    commit_rendered_prompt(committed_prompt_len, 0);
    EXPECT_EQ(committed_prompt_len, 48);

    commit_rendered_prompt(committed_prompt_len, -1);
    EXPECT_EQ(committed_prompt_len, 48);
}
