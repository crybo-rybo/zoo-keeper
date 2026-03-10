/**
 * @file zoo.hpp
 * @brief Umbrella include for the public zoo-keeper API surface.
 *
 * Consumers can include this header to access the core model wrapper, tool
 * system, and asynchronous agent orchestration layers.
 */

#pragma once

// Version
#include "version.hpp"

// Core types and model
#include "core/model.hpp"
#include "core/types.hpp"

// Tool system
#include "tools/parser.hpp"
#include "tools/registry.hpp"
#include "tools/types.hpp"
#include "tools/validation.hpp"

// Agent (async orchestration)
#include "agent.hpp"
