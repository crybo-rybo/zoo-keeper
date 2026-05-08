/**
 * @file test_system_probe.cpp
 * @brief Sanity checks for core::SystemProbe. Values are system-dependent;
 *        only structural invariants are asserted.
 */

#include "zoo/core/system_probe.hpp"

#include <gtest/gtest.h>

TEST(SystemProbeTest, ReturnsPopulatedSnapshot) {
    auto info = zoo::core::SystemProbe::probe();
    ASSERT_TRUE(info.has_value()) << info.error().to_string();
    EXPECT_GT(info->total_ram_bytes, 0u);
    EXPECT_GT(info->logical_cpu_count, 0u);
}

TEST(SystemProbeTest, GpuVectorMatchesSupportFlag) {
    auto info = zoo::core::SystemProbe::probe();
    ASSERT_TRUE(info.has_value());
    if (!info->gpu_offload_supported) {
        EXPECT_TRUE(info->gpus.empty());
    }
    for (const auto& gpu : info->gpus) {
        EXPECT_GE(gpu.total_vram_bytes, gpu.free_vram_bytes);
    }
}
