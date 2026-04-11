/**
 * @file test_request_tracker.cpp
 * @brief Unit tests for slot-backed async request storage.
 */

#include "agent/request_slots.hpp"
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <thread>

namespace zoo::internal::agent::test_support {

class RequestSlotsTestPeer {
  public:
    static std::unique_lock<std::mutex> lock_slot(RequestSlots& slots, uint32_t slot_index) {
        return std::unique_lock<std::mutex>(slots.slots_.at(slot_index)->mutex);
    }
};

} // namespace zoo::internal::agent::test_support

namespace {

using namespace std::chrono_literals;

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

TEST(RequestSlotsTest, CancelDoesNotBlockNewReservationsWhileWaitingOnSlotLock) {
    RequestSlots slots(3);

    auto first = slots.emplace(make_text_request("one"));
    auto second = slots.emplace(make_text_request("two"));
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    auto slot_lock =
        zoo::internal::agent::test_support::RequestSlotsTestPeer::lock_slot(slots, first->slot);

    std::promise<void> cancel_started;
    auto cancel_started_future = cancel_started.get_future();
    auto cancel_future = std::async(std::launch::async, [&] {
        cancel_started.set_value();
        slots.cancel(first->id);
    });

    ASSERT_EQ(cancel_started_future.wait_for(1s), std::future_status::ready);
    std::this_thread::sleep_for(20ms);

    auto emplace_future =
        std::async(std::launch::async, [&] { return slots.emplace(make_text_request("three")); });
    EXPECT_EQ(emplace_future.wait_for(50ms), std::future_status::ready);

    slot_lock.unlock();
    cancel_future.wait();

    auto third = emplace_future.get();
    ASSERT_TRUE(third.has_value()) << third.error().to_string();
}
