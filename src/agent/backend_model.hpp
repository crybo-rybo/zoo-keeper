/**
 * @file backend_model.hpp
 * @brief Internal factory for the concrete `core::Model` agent backend adapter.
 */

#pragma once

#include "backend.hpp"
#include <memory>

namespace zoo::core {
class Model;
}

namespace zoo::internal::agent {

/**
 * @brief Wraps a loaded `core::Model` in the internal agent backend seam.
 */
std::unique_ptr<AgentBackend> make_model_backend(std::unique_ptr<core::Model> model);

} // namespace zoo::internal::agent
