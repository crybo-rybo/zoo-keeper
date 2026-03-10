/**
 * @file test_agent_mailbox.cpp
 * @brief Unit tests for the agent request mailbox.
 */

#include "zoo/internal/agent/mailbox.hpp"
#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace {

zoo::internal::agent::Request make_request(zoo::RequestId id, std::string text) {
    zoo::internal::agent::Request request(zoo::Message::user(std::move(text)));
    request.id = id;
    return request;
}

} // namespace

TEST(RuntimeMailboxTest, PopsRequestsInSubmissionOrder) {
    zoo::internal::agent::RuntimeMailbox mailbox(2);

    ASSERT_TRUE(mailbox.push_request(make_request(1, "first")));
    ASSERT_TRUE(mailbox.push_request(make_request(2, "second")));

    auto first = mailbox.pop_request();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->id, 1u);
    EXPECT_EQ(first->message.content, "first");

    auto second = mailbox.pop_request();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->id, 2u);
    EXPECT_EQ(second->message.content, "second");
}

TEST(RuntimeMailboxTest, RejectsRequestsPastCapacity) {
    zoo::internal::agent::RuntimeMailbox mailbox(1);

    ASSERT_TRUE(mailbox.push_request(make_request(1, "first")));
    EXPECT_FALSE(mailbox.push_request(make_request(2, "second")));
    EXPECT_EQ(mailbox.size(), 1u);
}

TEST(RuntimeMailboxTest, ShutdownDrainsQueuedRequestsThenStops) {
    zoo::internal::agent::RuntimeMailbox mailbox(2);

    ASSERT_TRUE(mailbox.push_request(make_request(7, "queued")));
    mailbox.shutdown();

    auto queued = mailbox.pop_request();
    ASSERT_TRUE(queued.has_value());
    EXPECT_EQ(queued->id, 7u);

    auto drained = mailbox.pop_request();
    EXPECT_FALSE(drained.has_value());
}

TEST(RuntimeMailboxTest, RejectsNewRequestsAfterShutdown) {
    zoo::internal::agent::RuntimeMailbox mailbox(2);

    mailbox.shutdown();
    EXPECT_FALSE(mailbox.push_request(make_request(3, "late")));
}

TEST(RuntimeMailboxTest, ZeroCapacityAllowsUnboundedPushes) {
    zoo::internal::agent::RuntimeMailbox mailbox(0);

    for (zoo::RequestId i = 1; i <= 10; ++i) {
        ASSERT_TRUE(mailbox.push_request(make_request(i, "msg")));
    }
    EXPECT_EQ(mailbox.size(), 10u);
}
