/**
 * @file request_handle.cpp
 * @brief Out-of-line definitions for `zoo::RequestHandle<R>`.
 *
 * Methods that touch the polymorphic state need the full definition of
 * `RequestStateBase<R>`, so they live here with explicit instantiations for
 * `TextResponse` and `ExtractionResponse`.
 */

#include "agent/request_state.hpp"
#include "zoo/agent.hpp"

namespace zoo {

template <typename Result> RequestHandle<Result>::~RequestHandle() {
    reset();
}

template <typename Result>
RequestHandle<Result>& RequestHandle<Result>::operator=(RequestHandle&& other) noexcept {
    if (this != &other) {
        reset();
        id_ = other.id_;
        state_ = std::move(other.state_);
        other.id_ = 0;
    }
    return *this;
}

template <typename Result> bool RequestHandle<Result>::ready() const {
    return state_ && state_->ready();
}

template <typename Result> Expected<Result> RequestHandle<Result>::await_result() {
    if (!state_) {
        return std::unexpected(
            Error{ErrorCode::AgentNotRunning, "Request handle is no longer valid"});
    }
    auto result = state_->await();
    state_.reset();
    id_ = 0;
    return result;
}

template <typename Result>
Expected<Result> RequestHandle<Result>::await_result_for(std::chrono::nanoseconds timeout) {
    if (!state_) {
        return std::unexpected(
            Error{ErrorCode::AgentNotRunning, "Request handle is no longer valid"});
    }
    bool completed = false;
    auto result = state_->await_for(timeout, &completed);
    if (completed) {
        state_.reset();
        id_ = 0;
    }
    return result;
}

template <typename Result> void RequestHandle<Result>::reset() noexcept {
    if (state_) {
        state_->release();
        state_.reset();
    }
    id_ = 0;
}

template class RequestHandle<TextResponse>;
template class RequestHandle<ExtractionResponse>;

} // namespace zoo
