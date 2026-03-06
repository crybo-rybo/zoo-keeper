// Stub for create_llama_backend() so tests link without llama.cpp.
// Tests always pass a MockBackend explicitly, so this is never called.

#include "zoo/core/model.hpp"

namespace zoo::core {

std::unique_ptr<IBackend> create_llama_backend() {
    return nullptr;
}

} // namespace zoo::core
