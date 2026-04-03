/**
 * @file backend_init.hpp
 * @brief Shared llama.cpp backend initialization for core and hub layers.
 */

#pragma once

#include <llama.h>
#include <mutex>

namespace zoo::core {

/**
 * @brief Ensures the llama.cpp backend is initialized exactly once.
 *
 * Thread-safe via `std::call_once`. Both `core::Model` and `hub::GgufInspector`
 * call this before using llama.cpp APIs.
 */
inline void ensure_backend_initialized() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        llama_backend_init();
        ggml_backend_load_all();
    });
}

} // namespace zoo::core
