/**
 * @file request_state.hpp
 * @brief Type-erased state behind `zoo::RequestHandle<R>`.
 *
 * `RequestStateBase<R>` is the abstract interface; `SlotRequestState<R>` adapts
 * a slot in `RequestSlots`, and `ImmediateRequestState<R>` carries an already-
 * resolved one-shot result (used for failures detected before enqueue).
 */

#pragma once

#include "request_slots.hpp"
#include "zoo/core/types.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace zoo::internal::agent {

template <typename Result> class RequestStateBase {
  public:
    virtual ~RequestStateBase() = default;
    [[nodiscard]] virtual bool ready() const = 0;
    [[nodiscard]] virtual Expected<Result> await() = 0;
    [[nodiscard]] virtual Expected<Result> await_for(std::chrono::nanoseconds timeout,
                                                     bool* completed) = 0;
    virtual void release() = 0;
};

template <typename Result> class SlotRequestState final : public RequestStateBase<Result> {
  public:
    SlotRequestState(std::shared_ptr<RequestSlots> slots, uint32_t slot,
                     uint32_t generation) noexcept
        : slots_(std::move(slots)), slot_(slot), generation_(generation) {}

    [[nodiscard]] bool ready() const override {
        return slots_->ready(slot_, generation_);
    }

    [[nodiscard]] Expected<Result> await() override {
        return slots_->template await_result<Result>(slot_, generation_);
    }

    [[nodiscard]] Expected<Result> await_for(std::chrono::nanoseconds timeout,
                                             bool* completed) override {
        return slots_->template await_result<Result>(slot_, generation_, timeout, completed);
    }

    void release() override {
        slots_->release(slot_, generation_);
    }

  private:
    std::shared_ptr<RequestSlots> slots_;
    uint32_t slot_;
    uint32_t generation_;
};

template <typename Result> class ImmediateRequestState final : public RequestStateBase<Result> {
  public:
    explicit ImmediateRequestState(Expected<Result> result) noexcept
        : result_(std::move(result)) {}

    [[nodiscard]] bool ready() const override {
        return true;
    }

    [[nodiscard]] Expected<Result> await() override {
        if (!result_.has_value()) {
            return std::unexpected(
                Error{ErrorCode::AgentNotRunning, "Request result is no longer available"});
        }
        auto out = std::move(*result_);
        result_.reset();
        return out;
    }

    [[nodiscard]] Expected<Result> await_for(std::chrono::nanoseconds, bool* completed) override {
        if (completed != nullptr) {
            *completed = true;
        }
        return await();
    }

    void release() override {
        result_.reset();
    }

  private:
    std::optional<Expected<Result>> result_;
};

} // namespace zoo::internal::agent
