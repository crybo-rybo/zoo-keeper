/**
 * @file request.hpp
 * @brief Internal request payload shared by agent runtime components.
 */

#pragma once

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <zoo/core/types.hpp>

namespace zoo::internal::agent {

/**
 * @brief One queued agent request plus its shared completion state.
 */
struct Request {
    Message message;
    std::optional<std::function<void(std::string_view)>> streaming_callback;
    std::shared_ptr<std::promise<Expected<Response>>> promise;
    RequestId id = 0;
    std::shared_ptr<std::atomic<bool>> cancelled;

    Request(Message msg,
            std::optional<std::function<void(std::string_view)>> callback = std::nullopt)
        : message(std::move(msg)), streaming_callback(std::move(callback)),
          cancelled(std::make_shared<std::atomic<bool>>(false)) {}

    Request(Message msg, std::optional<std::function<void(std::string_view)>> callback,
            std::shared_ptr<std::promise<Expected<Response>>> request_promise, RequestId request_id,
            std::shared_ptr<std::atomic<bool>> cancel_flag)
        : message(std::move(msg)), streaming_callback(std::move(callback)),
          promise(std::move(request_promise)), id(request_id), cancelled(std::move(cancel_flag)) {}
};

} // namespace zoo::internal::agent
