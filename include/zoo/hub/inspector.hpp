/**
 * @file inspector.hpp
 * @brief GGUF file inspection and automatic model configuration.
 */

#pragma once

#include "zoo/core/types.hpp"
#include "zoo/hub/types.hpp"

#include <string>

namespace zoo::hub {

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
     * @brief Generates a ModelConfig with sensible defaults from inspection metadata.
     *
     * Auto-configuration logic:
     * - `context_size` = min(training context, 8192) to avoid OOM
     * - `n_gpu_layers` = -1 (offload all layers)
     * - `use_mmap` = true, `use_mlock` = false
     *
     * @param info Previously inspected model metadata.
     * @return A populated ModelConfig ready for Model::load().
     */
    static Expected<ModelConfig> auto_configure(const ModelInfo& info);

    /**
     * @brief Convenience overload that inspects a file and auto-configures in one step.
     */
    static Expected<ModelConfig> auto_configure(const std::string& file_path);
};

} // namespace zoo::hub
