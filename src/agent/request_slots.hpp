/**
 * @file request_slots.hpp
 * @brief Slot-backed async request storage for the agent runtime.
 */

#pragma once

#include "request.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace zoo::internal::agent {

namespace test_support {
class RequestSlotsTestPeer;
}

/**
 * @brief Reservation metadata returned when a request claims a slot.
 */
struct RequestReservation {
    RequestId id = 0;
    uint32_t slot = 0;
    uint32_t generation = 0;
};

/**
 * @brief Stable request view exposed to the inference thread.
 */
struct ActiveRequest {
    RequestId id = 0;
    HistoryMode history_mode = HistoryMode::Append;
    const std::vector<Message>* messages = nullptr;
    const GenerationOptions* options = nullptr;
    AsyncTextCallback* streaming_callback = nullptr;
    const std::optional<nlohmann::json>* extraction_schema = nullptr;
    const std::atomic<bool>* cancelled = nullptr;
    ResultKind result_kind = ResultKind::Text;
};

/**
 * @brief Fixed-capacity request state table.
 */
class RequestSlots {
    struct ReleasedSlot {
        RequestId request_id = 0;
        uint32_t index = 0;
    };

  public:
    explicit RequestSlots(size_t capacity) : slots_() {
        slots_.reserve(capacity);
        free_list_.reserve(capacity);
        for (size_t index = 0; index < capacity; ++index) {
            auto slot = std::make_unique<Slot>();
            slot->index = static_cast<uint32_t>(index);
            slots_.push_back(std::move(slot));
        }
        for (size_t index = capacity; index > 0; --index) {
            free_list_.push_back(static_cast<uint32_t>(index - 1));
        }
    }

    [[nodiscard]] Expected<RequestReservation> emplace(RequestPayload&& payload) {
        uint32_t slot_index;
        RequestId request_id;
        {
            std::lock_guard<std::mutex> table_lock(mutex_);
            if (free_list_.empty()) {
                return std::unexpected(
                    Error{ErrorCode::QueueFull, "Request queue is full or agent is shutting down"});
            }
            slot_index = free_list_.back();
            free_list_.pop_back();
            request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
            request_index_.emplace(request_id, slot_index);
        }

        Slot& slot = *slots_[slot_index];
        std::lock_guard<std::mutex> slot_lock(slot.mutex);
        slot.occupied = true;
        slot.orphaned = false;
        slot.ready = false;
        slot.request_id = request_id;
        slot.cancelled.store(false, std::memory_order_release);
        slot.payload = std::move(payload);
        slot.result = std::monostate{};
        return RequestReservation{
            slot.request_id,
            slot_index,
            slot.generation,
        };
    }

    [[nodiscard]] std::optional<ActiveRequest>
    active_request(const QueuedRequest& request) noexcept {
        if (request.slot >= slots_.size()) {
            return std::nullopt;
        }

        Slot& slot = *slots_[request.slot];
        std::lock_guard<std::mutex> lock(slot.mutex);
        if (!slot.occupied || slot.generation != request.generation) {
            return std::nullopt;
        }

        return ActiveRequest{
            slot.request_id,
            slot.payload.history_mode,
            &slot.payload.messages,
            &slot.payload.options,
            &slot.payload.streaming_callback,
            &slot.payload.extraction_schema,
            &slot.cancelled,
            slot.payload.result_kind,
        };
    }

    void cancel(RequestId id) {
        std::optional<uint32_t> slot_index;
        {
            std::lock_guard<std::mutex> table_lock(mutex_);
            auto it = request_index_.find(id);
            if (it == request_index_.end()) {
                return;
            }
            slot_index = it->second;
        }

        Slot& slot = *slots_[*slot_index];
        std::lock_guard<std::mutex> slot_lock(slot.mutex);
        if (slot.occupied && slot.request_id == id) {
            slot.cancelled.store(true, std::memory_order_release);
        }
    }

    void resolve_text(uint32_t slot_index, uint32_t generation, Expected<TextResponse> result) {
        resolve(slot_index, generation, std::move(result));
    }

    void resolve_extraction(uint32_t slot_index, uint32_t generation,
                            Expected<ExtractionResponse> result) {
        resolve(slot_index, generation, std::move(result));
    }

    void resolve_error(uint32_t slot_index, uint32_t generation, Error error) {
        std::optional<ReleasedSlot> released;
        if (slot_index >= slots_.size()) {
            return;
        }

        {
            Slot& slot = *slots_[slot_index];
            std::lock_guard<std::mutex> lock(slot.mutex);
            if (!slot.occupied || slot.generation != generation) {
                return;
            }

            if (slot.payload.result_kind == ResultKind::Extraction) {
                released = resolve_locked(
                    slot, Expected<ExtractionResponse>(std::unexpected(std::move(error))));
            } else {
                released =
                    resolve_locked(slot, Expected<TextResponse>(std::unexpected(std::move(error))));
            }
        }

        return_slot_to_free_list(std::move(released));
    }

    void fail_all(const Error& error) {
        for (auto& slot_ptr : slots_) {
            std::optional<ReleasedSlot> released;
            Slot& slot = *slot_ptr;
            {
                std::lock_guard<std::mutex> slot_lock(slot.mutex);
                if (!slot.occupied) {
                    continue;
                }

                if (slot.orphaned) {
                    released = clear_slot_for_reuse_locked(slot);
                } else if (slot.payload.result_kind == ResultKind::Extraction) {
                    slot.result = Expected<ExtractionResponse>(std::unexpected(error));
                    slot.ready = true;
                    slot.cv.notify_all();
                } else {
                    slot.result = Expected<TextResponse>(std::unexpected(error));
                    slot.ready = true;
                    slot.cv.notify_all();
                }
            }

            return_slot_to_free_list(std::move(released));
        }
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> table_lock(mutex_);
        return slots_.size() - free_list_.size();
    }

    [[nodiscard]] bool ready(uint32_t slot_index, uint32_t generation) const {
        if (slot_index >= slots_.size()) {
            return false;
        }

        const Slot& slot = *slots_[slot_index];
        std::lock_guard<std::mutex> lock(slot.mutex);
        return slot.occupied && slot.generation == generation && slot.ready;
    }

    template <typename Result>
    [[nodiscard]] Expected<Result>
    await_result(uint32_t slot_index, uint32_t generation,
                 std::optional<std::chrono::nanoseconds> timeout = std::nullopt,
                 bool* completed = nullptr) {
        if (completed != nullptr) {
            *completed = false;
        }

        if (slot_index >= slots_.size()) {
            if (completed != nullptr) {
                *completed = true;
            }
            return std::unexpected(
                Error{ErrorCode::AgentNotRunning, "Request result is no longer available"});
        }

        std::optional<Expected<Result>> result;
        std::optional<ReleasedSlot> released;

        {
            Slot& slot = *slots_[slot_index];
            std::unique_lock<std::mutex> lock(slot.mutex);

            auto pred = [&slot, generation] {
                return (!slot.occupied || slot.generation != generation) || slot.ready;
            };

            if (timeout) {
                if (!slot.cv.wait_for(lock, *timeout, pred)) {
                    return std::unexpected(
                        Error{ErrorCode::RequestTimeout, "Timed out waiting for request result"});
                }
            } else {
                slot.cv.wait(lock, pred);
            }

            if (completed != nullptr) {
                *completed = true;
            }
            if (!slot.occupied || slot.generation != generation) {
                return std::unexpected(
                    Error{ErrorCode::AgentNotRunning, "Request result is no longer available"});
            }

            result = std::move(std::get<Expected<Result>>(slot.result));
            released = clear_slot_for_reuse_locked(slot);
        }

        return_slot_to_free_list(std::move(released));
        return std::move(*result);
    }

    void release(uint32_t slot_index, uint32_t generation) {
        std::optional<ReleasedSlot> released;
        if (slot_index >= slots_.size()) {
            return;
        }

        {
            Slot& slot = *slots_[slot_index];
            std::lock_guard<std::mutex> lock(slot.mutex);
            if (!slot.occupied || slot.generation != generation) {
                return;
            }

            if (slot.ready) {
                released = clear_slot_for_reuse_locked(slot);
            } else {
                slot.orphaned = true;
            }
        }

        return_slot_to_free_list(std::move(released));
    }

  private:
    friend class test_support::RequestSlotsTestPeer;

    struct Slot {
        mutable std::mutex mutex;
        std::condition_variable cv;
        uint32_t index = 0;
        uint32_t generation = 1;
        bool occupied = false;
        bool orphaned = false;
        bool ready = false;
        RequestId request_id = 0;
        std::atomic<bool> cancelled{false};
        RequestPayload payload;
        std::variant<std::monostate, Expected<TextResponse>, Expected<ExtractionResponse>> result;
    };

    template <typename Result>
    void resolve(uint32_t slot_index, uint32_t generation, Expected<Result> result) {
        std::optional<ReleasedSlot> released;
        if (slot_index >= slots_.size()) {
            return;
        }

        {
            Slot& slot = *slots_[slot_index];
            std::lock_guard<std::mutex> lock(slot.mutex);
            if (!slot.occupied || slot.generation != generation) {
                return;
            }

            released = resolve_locked(slot, std::move(result));
        }

        return_slot_to_free_list(std::move(released));
    }

    template <typename Result>
    [[nodiscard]] std::optional<ReleasedSlot> resolve_locked(Slot& slot, Expected<Result> result) {
        if (slot.orphaned) {
            return clear_slot_for_reuse_locked(slot);
        }

        slot.result = std::move(result);
        slot.ready = true;
        slot.cv.notify_all();
        return std::nullopt;
    }

    [[nodiscard]] ReleasedSlot clear_slot_for_reuse_locked(Slot& slot) {
        ReleasedSlot released{slot.request_id, slot.index};
        slot.occupied = false;
        slot.orphaned = false;
        slot.ready = false;
        slot.request_id = 0;
        slot.cancelled.store(false, std::memory_order_release);
        slot.payload = RequestPayload{};
        slot.result = std::monostate{};
        ++slot.generation;
        if (slot.generation == 0) {
            slot.generation = 1;
        }
        return released;
    }

    void return_slot_to_free_list(std::optional<ReleasedSlot> released) {
        if (!released) {
            return;
        }
        std::lock_guard<std::mutex> table_lock(mutex_);
        request_index_.erase(released->request_id);
        free_list_.push_back(released->index);
    }

    mutable std::mutex mutex_;
    std::atomic<RequestId> next_request_id_{1};
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<uint32_t> free_list_;
    std::unordered_map<RequestId, uint32_t> request_index_;
};

} // namespace zoo::internal::agent
