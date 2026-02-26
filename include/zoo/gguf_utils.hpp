#pragma once

#include "zoo/types.hpp"
#include <string>

// gguf.h is in ggml/include/ which is a PUBLIC include directory of the ggml-base
// CMake target. It is transitively available when linking against zoo_backend (which
// links llama -> ggml). Users who include this header must link against zoo_backend.
#include "gguf.h"

namespace zoo {

/// Lightweight metadata extracted from a GGUF file without loading the model.
struct GgufModelInfo {
    std::string architecture;         ///< e.g. "gemma2", "llama", "phi3"
    int training_context_length = 0;  ///< From <arch>.context_length metadata key
    int n_layers = 0;                 ///< From <arch>.block_count
    int n_embd = 0;                   ///< From <arch>.embedding_length
    int n_head = 0;                   ///< From <arch>.attention.head_count
};

/// Read metadata from a GGUF file header without loading tensor data.
///
/// Uses gguf_init_from_file with no_alloc=true — only reads KV pairs from the
/// file header, not the multi-GB tensor blob. This makes it very fast and
/// suitable for model discovery, pre-load OOM estimation, and smart default
/// context-size selection.
///
/// @param model_path Path to a .gguf file
/// @return GgufModelInfo on success, or Error if the file cannot be opened or parsed
inline Expected<GgufModelInfo> read_gguf_metadata(const std::string& model_path) {
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context* ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        return tl::unexpected(Error{
            ErrorCode::ModelLoadFailed,
            "Failed to read GGUF metadata from: " + model_path
        });
    }

    GgufModelInfo info;

    // Read architecture: "general.architecture" -> e.g. "gemma2", "llama", "phi3"
    int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id >= 0) {
        info.architecture = gguf_get_val_str(ctx, arch_id);
    }

    if (!info.architecture.empty()) {
        // Read training context length: "<arch>.context_length"
        std::string ctx_key = info.architecture + ".context_length";
        int64_t ctx_id = gguf_find_key(ctx, ctx_key.c_str());
        if (ctx_id >= 0) {
            info.training_context_length = static_cast<int>(gguf_get_val_u32(ctx, ctx_id));
        }

        // Read layer count: "<arch>.block_count"
        std::string layers_key = info.architecture + ".block_count";
        int64_t layers_id = gguf_find_key(ctx, layers_key.c_str());
        if (layers_id >= 0) {
            info.n_layers = static_cast<int>(gguf_get_val_u32(ctx, layers_id));
        }

        // Read embedding dimension: "<arch>.embedding_length"
        std::string embd_key = info.architecture + ".embedding_length";
        int64_t embd_id = gguf_find_key(ctx, embd_key.c_str());
        if (embd_id >= 0) {
            info.n_embd = static_cast<int>(gguf_get_val_u32(ctx, embd_id));
        }

        // Read attention head count: "<arch>.attention.head_count"
        std::string head_key = info.architecture + ".attention.head_count";
        int64_t head_id = gguf_find_key(ctx, head_key.c_str());
        if (head_id >= 0) {
            info.n_head = static_cast<int>(gguf_get_val_u32(ctx, head_id));
        }
    }

    gguf_free(ctx);
    return info;
}

} // namespace zoo
