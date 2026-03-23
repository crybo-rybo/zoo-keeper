/**
 * @file test_request_tracker.cpp
 * @brief Unit tests for slot-backed async request storage.
 */

#include "zoo/internal/agent/request_slots.hpp"
#include <gtest/gtest.h>

namespace {

using zoo::Error;
using zoo::ErrorCode;
using zoo::Expected;
using zoo::GenerationOptions;
using zoo::HistorySnapshot;
using zoo::Message;
using zoo::TextResponse;
using zoo::internal::agent::HistoryMode;
using zoo::internal::agent::QueuedRequest;
using zoo::internal::agent::RequestPayload;
using zoo::internal::agent::RequestSlots;
using zoo::internal::agent::ResultKind;

RequestPayload make_text_request(std::string text) {
    RequestPayload payload;
    payload.messages.push_back(Message::user(std::move(text)));
    payload.history_mode = HistoryMode::Append;
    payload.options = GenerationOptions{};
    payload.result_kind = ResultKind::Text;
    return payload;
}

} // namespace

TEST(RequestSlotsTest, EmplaceAssignsIncreasingRequestIds) {
    RequestSlots slots(2);

    auto first = slots.emplace(make_text_request("first"));
    auto second = slots.emplace(make_text_request("second"));

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first->id, 1u);
    EXPECT_EQ(second->id, 2u);
    EXPECT_EQ(slots.size(), 2u);
}

TEST(RequestSlotsTest, CancelMarksActiveRequestFlag) {
    RequestSlots slots(1);

    auto reservation = slots.emplace(make_text_request("cancel me"));
    ASSERT_TRUE(reservation.has_value());

    auto active = slots.active_request(QueuedRequest{reservation->slot, reservation->generation});
    ASSERT_TRUE(active.has_value());
    ASSERT_NE(active->cancelled, nullptr);
    EXPECT_FALSE(active->cancelled->load(std::memory_order_acquire));

    slots.cancel(reservation->id);

    auto after_cancel =
        slots.active_request(QueuedRequest{reservation->slot, reservation->generation});
    ASSERT_TRUE(after_cancel.has_value());
    EXPECT_TRUE(after_cancel->cancelled->load(std::memory_order_acquire));
}

TEST(RequestSlotsTest, ResolveErrorPropagatesThroughAwaitHandle) {
    RequestSlots slots(1);

    auto reservation = slots.emplace(make_text_request("fail me"));
    ASSERT_TRUE(reservation.has_value());

    slots.resolve_error(reservation->slot, reservation->generation,
                        Error{ErrorCode::QueueFull, "Request queue is full"});

    auto result =
        RequestSlots::await_text_handle(&slots, reservation->slot, reservation->generation);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::QueueFull);
    EXPECT_EQ(slots.size(), 0u);
}

TEST(RequestSlotsTest, ReleaseBeforeResolveDropsResultAndFreesSlot) {
    RequestSlots slots(1);

    auto reservation = slots.emplace(make_text_request("orphan"));
    ASSERT_TRUE(reservation.has_value());

    RequestSlots::release_handle(&slots, reservation->slot, reservation->generation);
    slots.resolve_text(reservation->slot, reservation->generation,
                       Expected<TextResponse>(TextResponse{.text = "done"}));

    EXPECT_EQ(slots.size(), 0u);
}

TEST(RequestSlotsTest, FailAllResolvesOutstandingRequests) {
    RequestSlots slots(2);

    auto first = slots.emplace(make_text_request("one"));
    auto second = slots.emplace(make_text_request("two"));
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    slots.fail_all(Error{ErrorCode::AgentNotRunning, "shutdown"});

    auto r1 = RequestSlots::await_text_handle(&slots, first->slot, first->generation);
    auto r2 = RequestSlots::await_text_handle(&slots, second->slot, second->generation);

    ASSERT_FALSE(r1.has_value());
    ASSERT_FALSE(r2.has_value());
    EXPECT_EQ(r1.error().code, ErrorCode::AgentNotRunning);
    EXPECT_EQ(r2.error().code, ErrorCode::AgentNotRunning);
}
