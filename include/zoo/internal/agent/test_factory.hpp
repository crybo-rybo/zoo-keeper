/**
 * @file test_factory.hpp
 * @brief Internal test-only factory for constructing `zoo::Agent` with a fake backend.
 */

#pragma once

#include "backend.hpp"
#include "zoo/agent.hpp"
#include <memory>

namespace zoo::internal::agent {

/**
 * @brief Constructs an Agent using a caller-supplied backend.
 *
 * Available only in test builds via `ZOO_TESTING_HOOKS`.
 */
std::unique_ptr<Agent> make_test_agent(const Config& config, std::unique_ptr<AgentBackend> backend);

} // namespace zoo::internal::agent
