/**
 * @file zoo.hpp
 * @brief Umbrella include for the public zoo-keeper API surface.
 *
 * Consumers can include this header to access the core model wrapper, tool
 * system, asynchronous agent orchestration layer, and the optional hub layer
 * when built with `ZOO_BUILD_HUB=ON`.
 */

#pragma once

// Version
#include "version.hpp"

// Logging
#include "log.hpp"

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

// Hub layer (model lifecycle management) — optional
#ifdef ZOO_HUB_ENABLED
#include "hub/huggingface.hpp"
#include "hub/inspector.hpp"
#include "hub/store.hpp"
#include "hub/types.hpp"
#endif
