/**
 * @file log_callback.cpp
 * @brief Implements the consumer-configurable log callback storage.
 */

#include "log.hpp"

#include <atomic>

namespace {
std::atomic<zoo::LogCallback> g_callback{nullptr};
std::atomic<void*> g_user_data{nullptr};
} // namespace

namespace zoo {

void set_log_callback(LogCallback callback, void* user_data) {
    g_user_data.store(user_data, std::memory_order_release);
    g_callback.store(callback, std::memory_order_release);
}

namespace internal {

LogCallback get_log_callback() noexcept {
    return g_callback.load(std::memory_order_acquire);
}

void* get_log_user_data() noexcept {
    return g_user_data.load(std::memory_order_acquire);
}

} // namespace internal

} // namespace zoo
