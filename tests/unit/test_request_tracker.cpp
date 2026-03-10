/**
 * @file test_request_tracker.cpp
 * @brief Unit tests for agent request tracking and cancellation bookkeeping.
 */

#include "zoo/internal/agent/request_tracker.hpp"
#include <gtest/gtest.h>

TEST(RequestTrackerTest, PrepareAssignsIncreasingRequestIds) {
    zoo::internal::agent::RequestTracker tracker;

    auto first = tracker.prepare(zoo::Message::user("first"));
    auto second = tracker.prepare(zoo::Message::user("second"));

    EXPECT_EQ(first.request.id, 1u);
    EXPECT_EQ(second.request.id, 2u);
    EXPECT_EQ(tracker.size(), 2u);
}

TEST(RequestTrackerTest, CancelSetsSharedCancellationFlag) {
    zoo::internal::agent::RequestTracker tracker;

    auto prepared = tracker.prepare(zoo::Message::user("cancel me"));
    ASSERT_TRUE(prepared.request.cancelled);
    EXPECT_FALSE(prepared.request.cancelled->load(std::memory_order_acquire));

    tracker.cancel(prepared.request.id);

    EXPECT_TRUE(prepared.request.cancelled->load(std::memory_order_acquire));
}

TEST(RequestTrackerTest, FailResolvesFutureWithErrorAndCleansUpState) {
    zoo::internal::agent::RequestTracker tracker;

    auto prepared = tracker.prepare(zoo::Message::user("fail me"));
    ASSERT_TRUE(tracker.fail(prepared.request.id,
                             zoo::Error{zoo::ErrorCode::QueueFull, "Request queue is full"}));

    auto result = prepared.future.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::QueueFull);
    EXPECT_EQ(tracker.size(), 0u);
}

TEST(RequestTrackerTest, CleanupRemovesTrackedRequest) {
    zoo::internal::agent::RequestTracker tracker;

    auto prepared = tracker.prepare(zoo::Message::user("done"));
    EXPECT_EQ(tracker.size(), 1u);

    tracker.cleanup(prepared.request.id);

    EXPECT_EQ(tracker.size(), 0u);
    EXPECT_FALSE(
        tracker.fail(prepared.request.id, zoo::Error{zoo::ErrorCode::Unknown, "already removed"}));
}

TEST(RequestTrackerTest, CancelNonExistentIdIsNoOp) {
    zoo::internal::agent::RequestTracker tracker;

    tracker.cancel(999);
    EXPECT_EQ(tracker.size(), 0u);
}

TEST(RequestTrackerTest, FailReturnsFalseForUnknownId) {
    zoo::internal::agent::RequestTracker tracker;

    EXPECT_FALSE(tracker.fail(42, zoo::Error{zoo::ErrorCode::Unknown, "not tracked"}));
}
