/**
 * @file test_gpu_fit.cpp
 * @brief Unit tests for GPU memory-fit comparison helpers.
 */

#include "core/gpu_fit.hpp"

#include <gtest/gtest.h>

#include <array>
#include <llama.h>

TEST(GpuFitTest, UnchangedFitPreservesRequestedConfig) {
    const auto requested_model = llama_model_default_params();
    const auto fitted_model = requested_model;
    const auto requested_context = llama_context_default_params();
    const auto fitted_context = requested_context;
    const std::array<float, 1> tensor_split = {0.0f};
    const std::array<llama_model_tensor_buft_override, 1> overrides = {{{nullptr, nullptr}}};

    EXPECT_TRUE(zoo::core::gpu_fit_preserves_requested_config(
        requested_model, fitted_model, requested_context, fitted_context, tensor_split, overrides));
}

TEST(GpuFitTest, ChangedLayerCountRequiresRejection) {
    const auto requested_model = llama_model_default_params();
    auto fitted_model = requested_model;
    fitted_model.n_gpu_layers = 1;
    const auto requested_context = llama_context_default_params();
    const auto fitted_context = requested_context;
    const std::array<float, 1> tensor_split = {0.0f};
    const std::array<llama_model_tensor_buft_override, 1> overrides = {{{nullptr, nullptr}}};

    EXPECT_FALSE(zoo::core::gpu_fit_preserves_requested_config(
        requested_model, fitted_model, requested_context, fitted_context, tensor_split, overrides));
}

TEST(GpuFitTest, ChangedTensorPlacementRequiresRejection) {
    const auto requested_model = llama_model_default_params();
    const auto fitted_model = requested_model;
    const auto requested_context = llama_context_default_params();
    const auto fitted_context = requested_context;
    const std::array<float, 1> tensor_split = {1.0f};
    const std::array<llama_model_tensor_buft_override, 1> overrides = {{{"blk.*", nullptr}}};

    EXPECT_FALSE(zoo::core::gpu_fit_preserves_requested_config(
        requested_model, fitted_model, requested_context, fitted_context, tensor_split, overrides));
}
