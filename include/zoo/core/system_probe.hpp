/**
 * @file system_probe.hpp
 * @brief Host-system capability probe used by hardware-aware auto-configuration.
 */

#pragma once

#include "zoo/core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zoo::core {

/**
 * @brief Capabilities of a single GPU device exposed by the ggml backend.
 */
struct GpuInfo {
    std::string name;              ///< Backend-reported device name.
    uint64_t total_vram_bytes = 0; ///< Total VRAM reported by the device.
    uint64_t free_vram_bytes = 0;  ///< Free VRAM at the moment of probing.

    bool operator==(const GpuInfo& other) const = default;
};

/**
 * @brief Snapshot of host hardware relevant to model loading decisions.
 *
 * Populated by `SystemProbe::probe()` and consumed by
 * `GgufInspector::auto_configure(info, sys)`.
 */
struct SystemInfo {
    uint64_t total_ram_bytes = 0;       ///< Physical RAM reported by the OS.
    uint64_t available_ram_bytes = 0;   ///< Currently free/available RAM (best-effort).
    uint32_t logical_cpu_count = 0;     ///< `std::thread::hardware_concurrency()`.
    bool gpu_offload_supported = false; ///< `llama_supports_gpu_offload()`.
    std::vector<GpuInfo> gpus;          ///< Enumerated GPU devices (may be empty).

    bool operator==(const SystemInfo& other) const = default;
};

/**
 * @brief Probes the host for resources relevant to model loading.
 */
class SystemProbe {
  public:
    /**
     * @brief Returns a snapshot of host RAM, CPU, and GPU capabilities.
     *
     * Failures are non-fatal in spirit — the probe always returns a populated
     * `SystemInfo`, with zero-initialized fields where a particular query
     * could not be answered. The `Expected` wrapper is reserved for hard
     * failures (e.g. unsupported platform).
     */
    static Expected<SystemInfo> probe();
};

} // namespace zoo::core
