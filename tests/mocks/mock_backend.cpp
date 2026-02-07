// Mock backend implementation
// This file provides stubs needed for linking tests without llama.cpp

#include "mock_backend.hpp"

namespace zoo {
namespace backend {

// Stub for create_backend() so Agent::create() links without zoo_backend.
// Tests always pass a MockBackend explicitly, so this is never called.
std::unique_ptr<IBackend> create_backend() {
    return nullptr;
}

} // namespace backend
} // namespace zoo
