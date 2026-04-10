/**
 * @file test_agent_mailbox.cpp
 * @brief Unit tests for the agent runtime mailbox.
 */

#include "agent/mailbox.hpp"
#include <gtest/gtest.h>
#include <utility>

namespace {

using namespace zoo::internal::agent;

QueuedRequest make_request(uint32_t slot, uint32_t generation = 1) {
    return QueuedRequest{slot, generation};
}

const QueuedRequest& as_request(const WorkItem& item) {
    return std::get<QueuedRequest>(item);
}

} // namespace

TEST(RuntimeMailboxTest, PopsRequestsInSubmissionOrder) {
    RuntimeMailbox mailbox;

    ASSERT_TRUE(mailbox.push_request(make_request(1, 11)));
    ASSERT_TRUE(mailbox.push_request(make_request(2, 22)));

    auto first = mailbox.pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(as_request(*first).slot, 1u);
    EXPECT_EQ(as_request(*first).generation, 11u);

    auto second = mailbox.pop();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(as_request(*second).slot, 2u);
    EXPECT_EQ(as_request(*second).generation, 22u);
}

TEST(RuntimeMailboxTest, ShutdownDrainsQueuedRequestsThenStops) {
    RuntimeMailbox mailbox;

    ASSERT_TRUE(mailbox.push_request(make_request(7, 9)));
    mailbox.shutdown();

    auto queued = mailbox.pop();
    ASSERT_TRUE(queued.has_value());
    EXPECT_EQ(as_request(*queued).slot, 7u);

    auto drained = mailbox.pop();
    EXPECT_FALSE(drained.has_value());
}

TEST(RuntimeMailboxTest, RejectsNewRequestsAfterShutdown) {
    RuntimeMailbox mailbox;

    mailbox.shutdown();
    EXPECT_FALSE(mailbox.push_request(make_request(3, 4)));
}

TEST(RuntimeMailboxTest, CommandsArePrioritizedOverRequests) {
    RuntimeMailbox mailbox;

    ASSERT_TRUE(mailbox.push_request(make_request(1, 1)));

    auto promise = std::make_shared<std::promise<void>>();
    ASSERT_TRUE(mailbox.push_command(SetSystemPromptCmd{"hello", promise}));

    auto first = mailbox.pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_TRUE(std::holds_alternative<Command>(*first));

    auto second = mailbox.pop();
    ASSERT_TRUE(second.has_value());
    EXPECT_TRUE(std::holds_alternative<QueuedRequest>(*second));
}

TEST(RuntimeMailboxTest, RejectsCommandsAfterShutdown) {
    RuntimeMailbox mailbox;

    mailbox.shutdown();
    auto promise = std::make_shared<std::promise<void>>();
    EXPECT_FALSE(mailbox.push_command(SetSystemPromptCmd{"late", promise}));
}

TEST(RuntimeMailboxTest, AllCommandsDrainBeforeAnyRequest) {
    RuntimeMailbox mailbox;

    ASSERT_TRUE(mailbox.push_request(make_request(1, 1)));

    auto p1 = std::make_shared<std::promise<void>>();
    auto p2 = std::make_shared<std::promise<void>>();
    ASSERT_TRUE(mailbox.push_command(ClearHistoryCmd{p1}));
    ASSERT_TRUE(mailbox.push_command(SetSystemPromptCmd{"hi", p2}));

    auto first = mailbox.pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_TRUE(std::holds_alternative<Command>(*first));

    auto second = mailbox.pop();
    ASSERT_TRUE(second.has_value());
    EXPECT_TRUE(std::holds_alternative<Command>(*second));

    auto third = mailbox.pop();
    ASSERT_TRUE(third.has_value());
    EXPECT_TRUE(std::holds_alternative<QueuedRequest>(*third));
}

TEST(RuntimeMailboxTest, ShutdownDrainsCommandsThenStops) {
    RuntimeMailbox mailbox;

    auto promise = std::make_shared<std::promise<void>>();
    ASSERT_TRUE(mailbox.push_command(ClearHistoryCmd{promise}));
    mailbox.shutdown();

    auto item = mailbox.pop();
    ASSERT_TRUE(item.has_value());
    EXPECT_TRUE(std::holds_alternative<Command>(*item));

    auto drained = mailbox.pop();
    EXPECT_FALSE(drained.has_value());
}
