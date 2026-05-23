/**
 * @file gpu_fit.hpp
 * @brief Private helpers for checking llama.cpp GPU memory-fit results.
 */

#pragma once

#include <llama.h>
#include <span>

namespace zoo::core {

[[nodiscard]] inline bool tensor_split_changed(std::span<const float> tensor_split) noexcept {
    for (const float share : tensor_split) {
        if (share != 0.0f) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool
tensor_overrides_changed(std::span<const llama_model_tensor_buft_override> overrides) noexcept {
    for (const auto& override : overrides) {
        if (override.pattern != nullptr || override.buft != nullptr) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool gpu_fit_preserves_requested_config(
    const llama_model_params& requested_model, const llama_model_params& fitted_model,
    const llama_context_params& requested_context, const llama_context_params& fitted_context,
    std::span<const float> tensor_split,
    std::span<const llama_model_tensor_buft_override> overrides) noexcept {
    return fitted_model.n_gpu_layers == requested_model.n_gpu_layers &&
           fitted_context.n_ctx == requested_context.n_ctx && !tensor_split_changed(tensor_split) &&
           !tensor_overrides_changed(overrides);
}

} // namespace zoo::core
