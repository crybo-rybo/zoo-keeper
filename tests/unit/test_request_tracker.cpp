/**
 * @file test_request_tracker.cpp
 * @brief Unit tests for slot-backed async request storage.
 */

#include "agent/request_slots.hpp"
#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <mutex>
#include <optional>
#include <thread>

namespace zoo::internal::agent::test_support {

class RequestSlotsTestPeer {
  public:
    static std::unique_lock<std::mutex> lock_slot(RequestSlots& slots, uint32_t slot_index) {
        return std::unique_lock<std::mutex>(slots.slots_.at(slot_index)->mutex);
    }

    static std::unique_lock<std::mutex> lock_table(RequestSlots& slots) {
        return std::unique_lock<std::mutex>(slots.mutex_);
    }

    static std::optional<bool> try_read_slot_occupied(RequestSlots& slots, uint32_t slot_index) {
        std::unique_lock<std::mutex> lock(slots.slots_.at(slot_index)->mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            return std::nullopt;
        }
        return slots.slots_.at(slot_index)->occupied;
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

bool wait_until_slot_unoccupied(RequestSlots& slots, uint32_t slot_index) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        auto occupied =
            zoo::internal::agent::test_support::RequestSlotsTestPeer::try_read_slot_occupied(
                slots, slot_index);
        if (occupied.has_value() && !*occupied) {
            return true;
        }
        std::this_thread::sleep_for(5ms);
    }
    return false;
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

    auto result = slots.await_result<TextResponse>(reservation->slot, reservation->generation);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::QueueFull);
    EXPECT_EQ(slots.size(), 0u);
}

TEST(RequestSlotsTest, ReleaseBeforeResolveDropsResultAndFreesSlot) {
    RequestSlots slots(1);

    auto reservation = slots.emplace(make_text_request("orphan"));
    ASSERT_TRUE(reservation.has_value());

    slots.release(reservation->slot, reservation->generation);
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

    auto r1 = slots.await_result<TextResponse>(first->slot, first->generation);
    auto r2 = slots.await_result<TextResponse>(second->slot, second->generation);

    ASSERT_FALSE(r1.has_value());
    ASSERT_FALSE(r2.has_value());
    EXPECT_EQ(r1.error().code, ErrorCode::AgentNotRunning);
    EXPECT_EQ(r2.error().code, ErrorCode::AgentNotRunning);
}

TEST(RequestSlotsTest, FailAllResolvesOutstandingAndFreesOrphanedRequests) {
    RequestSlots slots(2);

    auto active = slots.emplace(make_text_request("active"));
    auto orphan = slots.emplace(make_text_request("orphan"));
    ASSERT_TRUE(active.has_value());
    ASSERT_TRUE(orphan.has_value());

    slots.release(orphan->slot, orphan->generation);
    slots.fail_all(Error{ErrorCode::AgentNotRunning, "shutdown"});

    EXPECT_EQ(slots.size(), 1u);

    auto result = slots.await_result<TextResponse>(active->slot, active->generation);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::AgentNotRunning);
    EXPECT_EQ(slots.size(), 0u);
}

TEST(RequestSlotsTest, AwaitResultDoesNotHoldSlotLockWhileReturningSlotToTable) {
    RequestSlots slots(1);

    auto reservation = slots.emplace(make_text_request("ready"));
    ASSERT_TRUE(reservation.has_value());
    slots.resolve_text(reservation->slot, reservation->generation,
                       Expected<TextResponse>(TextResponse{.text = "done"}));

    auto table_lock = zoo::internal::agent::test_support::RequestSlotsTestPeer::lock_table(slots);
    auto await_future = std::async(std::launch::async, [&] {
        return slots.await_result<TextResponse>(reservation->slot, reservation->generation);
    });

    EXPECT_TRUE(wait_until_slot_unoccupied(slots, reservation->slot));
    EXPECT_EQ(await_future.wait_for(0ms), std::future_status::timeout);

    table_lock.unlock();
    auto result = await_future.get();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ(result->text, "done");
    EXPECT_EQ(slots.size(), 0u);
}

TEST(RequestSlotsTest, OrphanedResolveDoesNotHoldSlotLockWhileReturningSlotToTable) {
    RequestSlots slots(1);

    auto reservation = slots.emplace(make_text_request("orphan"));
    ASSERT_TRUE(reservation.has_value());
    slots.release(reservation->slot, reservation->generation);

    auto table_lock = zoo::internal::agent::test_support::RequestSlotsTestPeer::lock_table(slots);
    auto resolve_future = std::async(std::launch::async, [&] {
        slots.resolve_text(reservation->slot, reservation->generation,
                           Expected<TextResponse>(TextResponse{.text = "done"}));
    });

    EXPECT_TRUE(wait_until_slot_unoccupied(slots, reservation->slot));
    EXPECT_EQ(resolve_future.wait_for(0ms), std::future_status::timeout);

    table_lock.unlock();
    resolve_future.get();
    EXPECT_EQ(slots.size(), 0u);
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
