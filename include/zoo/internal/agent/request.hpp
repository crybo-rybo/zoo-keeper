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
#include <vector>
#include <zoo/core/types.hpp>

namespace zoo::internal::agent {

enum class HistoryMode {
    Append,
    Replace,
};

/**
 * @brief One queued agent request plus its shared completion state.
 */
struct Request {
    std::vector<Message> messages;
    HistoryMode history_mode = HistoryMode::Append;
    std::optional<std::function<void(std::string_view)>> streaming_callback;
    std::shared_ptr<std::promise<Expected<Response>>> promise;
    RequestId id = 0;
    std::shared_ptr<std::atomic<bool>> cancelled;

    Request(Message msg,
            std::optional<std::function<void(std::string_view)>> callback = std::nullopt)
        : streaming_callback(std::move(callback)) {
        messages.emplace_back(std::move(msg));
    }

    Request(std::vector<Message> seed_messages, HistoryMode mode,
            std::optional<std::function<void(std::string_view)>> callback = std::nullopt)
        : messages(std::move(seed_messages)), history_mode(mode),
          streaming_callback(std::move(callback)) {}

    Request(Message msg, std::optional<std::function<void(std::string_view)>> callback,
            std::shared_ptr<std::promise<Expected<Response>>> request_promise, RequestId request_id,
            std::shared_ptr<std::atomic<bool>> cancel_flag)
        : streaming_callback(std::move(callback)), promise(std::move(request_promise)),
          id(request_id), cancelled(std::move(cancel_flag)) {
        messages.emplace_back(std::move(msg));
    }

    Request(std::vector<Message> seed_messages, HistoryMode mode,
            std::optional<std::function<void(std::string_view)>> callback,
            std::shared_ptr<std::promise<Expected<Response>>> request_promise, RequestId request_id,
            std::shared_ptr<std::atomic<bool>> cancel_flag)
        : messages(std::move(seed_messages)), history_mode(mode),
          streaming_callback(std::move(callback)), promise(std::move(request_promise)),
          id(request_id), cancelled(std::move(cancel_flag)) {}
};

} // namespace zoo::internal::agent
