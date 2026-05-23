/**
 * @file tool_executor.cpp
 * @brief Bounded worker pool for off-thread tool handler execution.
 */

#include "agent/tool_executor.hpp"

#include "log.hpp"

#include <exception>
#include <utility>

namespace zoo::internal::agent {

namespace {

void set_tool_handler_result(const std::shared_ptr<std::promise<Expected<nlohmann::json>>>& promise,
                             tools::ToolHandler& handler, nlohmann::json& args) {
    try {
        promise->set_value(handler(args));
    } catch (const std::exception& e) {
        ZOO_LOG("error", "tool handler threw: %s", e.what());
        promise->set_value(std::unexpected(
            Error{ErrorCode::ToolExecutionFailed, std::string("Tool handler threw: ") + e.what()}));
    } catch (...) {
        ZOO_LOG("error", "tool handler threw unknown exception");
        promise->set_value(std::unexpected(
            Error{ErrorCode::ToolExecutionFailed, "Tool handler threw unknown exception"}));
    }
}

} // namespace

ToolExecutor::Handle::Handle(std::future<Expected<nlohmann::json>> future,
                             std::shared_ptr<JobControl> control,
                             std::weak_ptr<State> state) noexcept
    : future_(std::move(future)), control_(std::move(control)), state_(std::move(state)) {}

Expected<nlohmann::json> ToolExecutor::Handle::get() {
    return future_.get();
}

void ToolExecutor::Handle::abandon() {
    if (!control_) {
        return;
    }
    bool should_replace = false;
    {
        std::lock_guard lock(control_->mutex);
        if (control_->abandoned) {
            return;
        }
        control_->abandoned = true;
        should_replace = control_->running && !control_->completed;
    }
    if (!should_replace) {
        return;
    }
    if (auto state = state_.lock()) {
        replace_abandoned_worker(state, control_);
    }
}

ToolExecutor::ToolExecutor(size_t worker_count) : state_(std::make_shared<State>()) {
    if (worker_count == 0) {
        state_->shutdown.store(true, std::memory_order_release);
        return;
    }

    for (size_t i = 0; i < worker_count; ++i) {
        if (!start_worker(state_)) {
            state_->shutdown.store(true, std::memory_order_release);
            state_->cv.notify_all();
            break;
        }
    }
}

ToolExecutor::~ToolExecutor() {
    {
        std::lock_guard lock(state_->mutex);
        state_->shutdown.store(true, std::memory_order_release);
        fail_pending_jobs_locked(*state_,
                                 Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"});
    }
    state_->cv.notify_all();
}

ToolExecutor::Handle ToolExecutor::submit(tools::ToolHandler handler, nlohmann::json args) {
    auto promise = std::make_shared<std::promise<Expected<nlohmann::json>>>();
    auto future = promise->get_future();
    if (state_->shutdown.load(std::memory_order_acquire)) {
        promise->set_value(
            std::unexpected(Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"}));
        return Handle(std::move(future), {}, {});
    }

    auto control = std::make_shared<JobControl>();
    {
        std::lock_guard lock(state_->mutex);
        if (state_->shutdown.load(std::memory_order_acquire)) {
            promise->set_value(
                std::unexpected(Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"}));
            return Handle(std::move(future), {}, {});
        }
        if (state_->pooled_workers == 0) {
            promise->set_value(std::unexpected(
                Error{ErrorCode::ToolExecutionFailed, "Tool executor has no worker threads"}));
            return Handle(std::move(future), {}, {});
        }
        state_->queue.push(Job{std::move(handler), std::move(args), std::move(promise), control});
    }
    state_->cv.notify_one();
    return Handle(std::move(future), std::move(control), state_);
}

bool ToolExecutor::start_worker(const std::shared_ptr<State>& state) noexcept {
    {
        std::lock_guard lock(state->mutex);
        if (state->shutdown.load(std::memory_order_acquire)) {
            return false;
        }
        ++state->pooled_workers;
    }

    try {
        std::thread([state]() { worker_loop(state); }).detach();
        return true;
    } catch (const std::exception& e) {
        retire_worker(state);
        ZOO_LOG("error", "failed to start tool worker thread: %s", e.what());
    } catch (...) {
        retire_worker(state);
        ZOO_LOG("error", "failed to start tool worker thread");
    }
    return false;
}

void ToolExecutor::worker_loop(const std::shared_ptr<State>& state) {
    while (true) {
        Job job;
        {
            std::unique_lock lock(state->mutex);
            state->cv.wait(lock, [&state]() {
                return state->shutdown.load(std::memory_order_acquire) || !state->queue.empty();
            });
            if (state->shutdown.load(std::memory_order_acquire) && state->queue.empty()) {
                --state->pooled_workers;
                return;
            }
            job = std::move(state->queue.front());
            state->queue.pop();
        }

        bool abandoned_before_run = false;
        {
            std::lock_guard control_lock(job.control->mutex);
            abandoned_before_run = job.control->abandoned;
            if (abandoned_before_run) {
                job.control->completed = true;
            } else {
                job.control->running = true;
            }
        }
        if (abandoned_before_run) {
            job.promise->set_value(std::unexpected(
                Error{ErrorCode::RequestCancelled, "Tool execution was abandoned"}));
            continue;
        }

        set_tool_handler_result(job.promise, job.handler, job.args);
        bool abandoned_after_run = false;
        {
            std::lock_guard control_lock(job.control->mutex);
            job.control->completed = true;
            abandoned_after_run = job.control->abandoned;
        }
        if (abandoned_after_run) {
            if (replace_abandoned_worker(state, job.control)) {
                retire_worker(state);
                return;
            }
        }
    }
}

void ToolExecutor::retire_worker(const std::shared_ptr<State>& state) {
    std::lock_guard lock(state->mutex);
    if (state->pooled_workers > 0) {
        --state->pooled_workers;
    }
}

bool ToolExecutor::replace_abandoned_worker(const std::shared_ptr<State>& state,
                                            const std::shared_ptr<JobControl>& control) noexcept {
    if (state->shutdown.load(std::memory_order_acquire)) {
        return true;
    }
    {
        std::lock_guard lock(control->mutex);
        if (control->replacement_started) {
            return true;
        }
        control->replacement_started = true;
    }
    if (start_worker(state)) {
        return true;
    }
    if (state->shutdown.load(std::memory_order_acquire)) {
        return true;
    }
    {
        std::lock_guard lock(control->mutex);
        control->replacement_started = false;
    }
    ZOO_LOG("error", "failed to replace abandoned tool worker");
    return false;
}

void ToolExecutor::fail_pending_jobs_locked(State& state, Error error) {
    while (!state.queue.empty()) {
        auto job = std::move(state.queue.front());
        state.queue.pop();
        job.promise->set_value(std::unexpected(error));
    }
}

} // namespace zoo::internal::agent
