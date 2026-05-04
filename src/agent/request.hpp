/**
 * @file request.hpp
 * @brief Internal request payload and queued descriptor types.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>
#include <zoo/core/types.hpp>

namespace zoo::internal::agent {

/**
 * @brief Specifies whether a request appends to or replaces retained history.
 */
enum class HistoryMode {
    Append,
    Replace,
};

/**
 * @brief Final result family expected for the request.
 */
enum class ResultKind {
    Text,
    Extraction,
};

/**
 * @brief Full per-request payload stored in a request slot.
 */
struct RequestPayload {
    std::vector<Message> messages;
    HistoryMode history_mode = HistoryMode::Append;
    GenerationOptions options;
    AsyncTokenCallback streaming_callback;
    std::optional<nlohmann::json> extraction_schema;
    ResultKind result_kind = ResultKind::Text;
};

/**
 * @brief Small queued descriptor pointing at one occupied request slot.
 */
struct QueuedRequest {
    uint32_t slot = 0;
    uint32_t generation = 0;

    bool operator==(const QueuedRequest& other) const = default;
};

} // namespace zoo::internal::agent
