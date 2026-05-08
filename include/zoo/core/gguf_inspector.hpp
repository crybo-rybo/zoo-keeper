/**
 * @file gguf_inspector.hpp
 * @brief GGUF file inspection and hardware-aware model configuration.
 */

#pragma once

#include "zoo/core/model_info.hpp"
#include "zoo/core/system_probe.hpp"
#include "zoo/core/types.hpp"

#include <string>

namespace zoo::core {

/**
 * @brief Reads GGUF file metadata and generates sensible model configurations.
 *
 * `GgufInspector` extracts model metadata from GGUF files without loading
 * tensor weights. This enables fast introspection for model catalogs and
 * auto-configuration without the cost of a full model load.
 */
class GgufInspector {
  public:
    /**
     * @brief Reads metadata from a GGUF file without loading model weights.
     *
     * Uses a two-phase approach: raw GGUF KV reading for metadata, then a
     * vocab-only model load for derived statistics (parameter count, size).
     *
     * @param file_path Absolute path to a GGUF model file.
     * @return ModelInfo populated with extracted metadata, or an error.
     */
    static Expected<ModelInfo> inspect(const std::string& file_path);

    /**
     * @brief Generates a hardware-aware ModelConfig from inspection metadata
     *        and a probe of the host system.
     *
     * Heuristics:
     * - `context_size` is bounded by training context, KV-cache RAM budget,
     *   and a v1 sanity ceiling of 32k.
     * - `n_gpu_layers` reflects whether the model fits in available VRAM
     *   (full offload, partial layer count, or CPU only).
     * - `use_mmap` is always enabled.
     * - `use_mlock` is enabled when total RAM comfortably exceeds model size.
     */
    static Expected<ModelConfig> auto_configure(const ModelInfo& info, const SystemInfo& sys);

    /**
     * @brief Convenience overload that probes the host system internally.
     *
     * Equivalent to calling `auto_configure(info, *SystemProbe::probe())`.
     */
    static Expected<ModelConfig> auto_configure(const ModelInfo& info);

    /**
     * @brief Convenience overload that inspects a file and auto-configures in one step.
     */
    static Expected<ModelConfig> auto_configure(const std::string& file_path);
};

} // namespace zoo::core
