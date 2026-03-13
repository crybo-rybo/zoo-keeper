/**
 * @file test_agent_mailbox.cpp
 * @brief Unit tests for the agent runtime mailbox.
 */

#include "zoo/internal/agent/mailbox.hpp"
#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace {

using namespace zoo::internal::agent;

Request make_request(zoo::RequestId id, std::string text) {
    Request request(zoo::Message::user(std::move(text)));
    request.id = id;
    return request;
}

const Request& as_request(const WorkItem& item) {
    return std::get<Request>(item);
}

} // namespace

TEST(RuntimeMailboxTest, PopsRequestsInSubmissionOrder) {
    RuntimeMailbox mailbox(2);

    ASSERT_TRUE(mailbox.push_request(make_request(1, "first")));
    ASSERT_TRUE(mailbox.push_request(make_request(2, "second")));

    auto first = mailbox.pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(as_request(*first).id, 1u);
    ASSERT_EQ(as_request(*first).messages.size(), 1u);
    EXPECT_EQ(as_request(*first).messages.front().content, "first");

    auto second = mailbox.pop();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(as_request(*second).id, 2u);
    ASSERT_EQ(as_request(*second).messages.size(), 1u);
    EXPECT_EQ(as_request(*second).messages.front().content, "second");
}

TEST(RuntimeMailboxTest, RejectsRequestsPastCapacity) {
    RuntimeMailbox mailbox(1);

    ASSERT_TRUE(mailbox.push_request(make_request(1, "first")));
    EXPECT_FALSE(mailbox.push_request(make_request(2, "second")));
    EXPECT_EQ(mailbox.size(), 1u);
}

TEST(RuntimeMailboxTest, ShutdownDrainsQueuedRequestsThenStops) {
    RuntimeMailbox mailbox(2);

    ASSERT_TRUE(mailbox.push_request(make_request(7, "queued")));
    mailbox.shutdown();

    auto queued = mailbox.pop();
    ASSERT_TRUE(queued.has_value());
    EXPECT_EQ(as_request(*queued).id, 7u);

    auto drained = mailbox.pop();
    EXPECT_FALSE(drained.has_value());
}

TEST(RuntimeMailboxTest, RejectsNewRequestsAfterShutdown) {
    RuntimeMailbox mailbox(2);

    mailbox.shutdown();
    EXPECT_FALSE(mailbox.push_request(make_request(3, "late")));
}

TEST(RuntimeMailboxTest, ZeroCapacityAllowsUnboundedPushes) {
    RuntimeMailbox mailbox(0);

    for (zoo::RequestId i = 1; i <= 10; ++i) {
        ASSERT_TRUE(mailbox.push_request(make_request(i, "msg")));
    }
    EXPECT_EQ(mailbox.size(), 10u);
}

TEST(RuntimeMailboxTest, CommandsArePrioritizedOverRequests) {
    RuntimeMailbox mailbox(2);

    ASSERT_TRUE(mailbox.push_request(make_request(1, "request")));

    auto promise = std::make_shared<std::promise<void>>();
    ASSERT_TRUE(mailbox.push_command(SetSystemPromptCmd{"hello", promise}));

    auto first = mailbox.pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_TRUE(std::holds_alternative<Command>(*first));

    auto second = mailbox.pop();
    ASSERT_TRUE(second.has_value());
    EXPECT_TRUE(std::holds_alternative<Request>(*second));
}

TEST(RuntimeMailboxTest, RejectsCommandsAfterShutdown) {
    RuntimeMailbox mailbox(2);

    mailbox.shutdown();
    auto promise = std::make_shared<std::promise<void>>();
    EXPECT_FALSE(mailbox.push_command(SetSystemPromptCmd{"late", promise}));
}

TEST(RuntimeMailboxTest, AllCommandsDrainBeforeAnyRequest) {
    RuntimeMailbox mailbox(4);

    ASSERT_TRUE(mailbox.push_request(make_request(1, "req")));

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
    EXPECT_TRUE(std::holds_alternative<Request>(*third));
}

TEST(RuntimeMailboxTest, ShutdownDrainsCommandsThenStops) {
    RuntimeMailbox mailbox(2);

    auto promise = std::make_shared<std::promise<void>>();
    ASSERT_TRUE(mailbox.push_command(ClearHistoryCmd{promise}));
    mailbox.shutdown();

    auto item = mailbox.pop();
    ASSERT_TRUE(item.has_value());
    EXPECT_TRUE(std::holds_alternative<Command>(*item));

    auto drained = mailbox.pop();
    EXPECT_FALSE(drained.has_value());
}
