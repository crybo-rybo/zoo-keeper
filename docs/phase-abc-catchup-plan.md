# Phase A-C Catch-Up Plan

## Purpose

This document captures the current state of roadmap compliance for the completed cleanup phases:

- Phase A - Shape the surface
- Phase B - Unify the tool system
- Phase C - Simplify the async runtime

It is based on a code-and-doc audit of the repository as it exists today, not on roadmap intent alone. The goal is to close the remaining gaps cleanly and keep the broader cleanup program aligned with its stated end state:

- one obvious public story,
- smaller and clearer public API,
- private runtime implementation boundaries,
- unified tool registration and explicit tool execution data,
- deterministic behavior,
- docs and package behavior that match reality.

## Executive Summary

### Current assessment

- Phase A: partial
- Phase B: mostly complete
- Phase C: complete on current evidence

### Main blockers to calling Phases A-C complete

1. The package/build story is still inconsistent between build-tree and install-tree consumption.
2. The install surface still vendors `nlohmann` headers, which the roadmap explicitly called out for cleanup.
3. The target story is clearer than before, but not fully collapsed into one obvious recommended public target in all docs.
4. The tool event/result model exists in code, but there is still room to tighten test coverage around runtime-populated `Response::tool_invocations`.

## Audit Evidence Summary

### Verified during audit

- `Agent` is now a thin public facade with runtime implementation moved into private files.
- Internal headers under `include/zoo/internal/*` are excluded from install.
- Manual-schema registration is available directly on `Agent`.
- Manual schema support is explicitly narrowed and enforced.
- Tool ordering is deterministic and backed by tests.
- The async runtime now uses a private backend seam and focused internal units.
- Agent orchestration behavior is covered by fake-backend unit tests.
- The installed CMake package configures and builds successfully in a minimal consumer project.

### Gaps confirmed during audit

- Build-tree `find_package(ZooKeeper CONFIG)` consumption is not working as documented.
- `nlohmann` headers are still installed into the package include tree.
- The docs still present `zoo_core` as a recommended consumer target in at least one place.
- There is no direct runtime test that asserts real `Response::tool_invocations` contents across tool success and failure paths.

## Phase A - Shape the Surface

### Roadmap scope

- Epic 1 - Public API and Build Surface Simplification
- Epic 6.1 - Export real CMake package targets
- Epic 6.2 - Align docs with actual behavior

### Status

- Issue 1.1: partial
- Issue 1.2: complete
- Issue 1.3: partial
- Issue 6.1: partial
- Issue 6.2: mostly complete

### What is already done

- `zoo` is the real compiled public target.
- `Agent` implementation has been moved out of the public header into private runtime/backend files.
- Internal runtime headers are no longer installed.
- The docs now largely teach `ZooKeeper::zoo` as the consumer-facing target.
- Installed-package export uses an actual exported target set rather than hand-built imported-target recreation.

### Findings

#### 1.1 Collapse the public target story

Progress:

- Good: `zoo` is the primary compiled target.
- Good: examples link against `zoo`.
- Good: the build guide teaches `ZooKeeper::zoo` for FetchContent, subdirectory use, and installed-package use.
- Remaining problem: `zoo_core` is still exported and still documented as a recommended link target in architecture docs, which weakens the "one obvious public story" goal.

Remaining tasks:

1. Decide whether `zoo_core` remains only as a compatibility shim or is fully removed from public guidance.
2. Update docs so `ZooKeeper::zoo` is the sole recommended target for normal consumers.
3. If `zoo_core` stays exported, describe it as compatibility-only and not part of the primary public story.

Exit criteria:

- One recommended consumer target appears consistently in docs and examples.
- Compatibility targets, if retained, are explicitly non-primary.

#### 1.2 Move `Agent` implementation out of the public header

Progress:

- This is effectively complete.
- The public header is declaration-oriented.
- Worker loop, runtime coordination, and concrete backend integration live in private implementation files.

Remaining tasks:

1. No structural work required for Phase A closure.
2. Keep future `Agent` additions from re-leaking runtime implementation into public headers.

Exit criteria:

- None beyond preserving current shape.

#### 1.3 Stop installing internal headers and dependency headers directly

Progress:

- Good: `include/zoo/internal/*` is excluded from install.
- Remaining problem: vendored `nlohmann` headers are still installed as part of the package.

Remaining tasks:

1. Remove the install rule that copies `nlohmann` headers into `${CMAKE_INSTALL_INCLUDEDIR}`.
2. Replace vendored-header installation with a clearer dependency story for package consumers.
3. Re-test installed-package consumption after that change.
4. Update package docs if the dependency model changes.

Exit criteria:

- Installed headers contain only the supported zoo public API.
- Third-party headers are not copied into the consumer include tree by this project.

#### 6.1 Export real CMake package targets

Progress:

- Good: install-tree export is based on actual exported targets.
- Good: installed-package consumption works in a minimal consumer project.
- Remaining problem: build-tree package consumption is not equivalent.
- Remaining problem: the generated build-tree `ZooKeeperConfig.cmake` expects `ZooKeeperTargets.cmake`, but that file is not produced next to it in the build root.

Remaining tasks:

1. Add proper build-tree export support so `find_package(ZooKeeper CONFIG)` works against the build tree.
2. Ensure build-tree and install-tree configs are generated from the same public target story.
3. Re-run a minimal consumer smoke test for both:
   - build-tree package consumption
   - install-tree package consumption
4. Revisit whether the `llama` dependency attachment can be simplified further without custom patch-up logic.

Exit criteria:

- Build-tree and install-tree package consumption both work.
- The public imported targets exposed to consumers are the same in both modes.

#### 6.2 Align docs with actual behavior

Progress:

- Good: tool docs now document the actual schema subset and explicit tool invocation records.
- Good: getting started and examples reflect the modern `Agent` API.
- Remaining problem: target guidance is still mixed because `zoo_core` is still presented as recommended in architecture docs.

Remaining tasks:

1. Update architecture docs so target guidance matches the actual intended public story.
2. Re-check all build/package docs after the build-tree export fix.
3. Keep schema-support docs aligned with the actual validator and grammar implementation.

Exit criteria:

- A new contributor can follow the docs without running into target/package mismatches.

## Phase B - Unify the Tool System

### Roadmap scope

- Epic 2 - Tooling API Unification and Contract Hardening
- Additional tests for deterministic ordering and explicit tool event/result types

### Status

- Issue 2.1: complete
- Issue 2.2: complete
- Issue 2.3: implemented, with one testing gap
- Issue 2.4: complete

### What is already done

- `Agent` exposes both typed and manual-schema tool registration.
- Manual schema support is explicitly narrowed and validated at registration time.
- Unsupported schema constructs fail fast.
- Tool metadata ordering is deterministic and registration-order based.
- `Response` now contains explicit `ToolInvocation` records instead of opaque message reuse.
- Docs and examples teach the unified `Agent`-first tool workflow.

### Findings

#### 2.1 Add manual-schema registration directly to `Agent`

Progress:

- Complete.
- Advanced example code now uses `Agent::register_tool(..., schema, handler)` directly.

Remaining tasks:

1. None for roadmap closure.

#### 2.2 Define the supported schema subset explicitly

Progress:

- Complete.
- The code now enforces a small supported subset and rejects unsupported keywords, nested objects, arrays, `$ref`, and bounds-like keywords.
- Docs reflect the same narrowed subset.

Remaining tasks:

1. None for roadmap closure.
2. If schema support is expanded later, do it as a new deliberate follow-up rather than in-place drift.

#### 2.3 Introduce explicit tool execution domain types

Progress:

- Implemented in code and reflected in docs/examples.
- `ToolInvocation` captures id, name, serialized arguments, status, optional result, and optional error.
- Runtime populates these records during the agent tool loop.

Remaining tasks:

1. Add focused runtime tests that assert actual `Response::tool_invocations` contents for:
   - successful tool execution
   - validation failure
   - execution failure
2. Verify that response examples and docs still match any future changes to the invocation record shape.

Exit criteria:

- Tool execution records are not only defined, but proven through runtime behavior tests.

#### 2.4 Make tool/schema ordering deterministic

Progress:

- Complete.
- Registration order is preserved and covered by tests for schema listing and grammar generation.

Remaining tasks:

1. None for roadmap closure.

## Phase C - Simplify the Async Runtime

### Roadmap scope

- Epic 3 - Agent Runtime Simplification
- Expanded unit-test coverage for agent orchestration behavior

### Status

- Issue 3.1: complete
- Issue 3.2: complete
- Issue 3.3: complete

### What is already done

- Model access is effectively inference-thread owned through the private backend/runtime split.
- Cross-thread mutating operations are expressed as commands through the mailbox.
- Queueing, cancellation bookkeeping, command routing, runtime orchestration, and concrete backend adaptation are now split into focused private units.
- `src/agent.cpp` is a thin facade rather than the full worker implementation.
- Fake-backend tests now cover key orchestration paths.

### Findings

#### 3.1 Make model access inference-thread-owned

Progress:

- Complete on current evidence.
- The previous coarse request-scoped mutex model has been replaced by inference-thread ownership plus explicit command routing.

Remaining tasks:

1. No Phase C catch-up work required.
2. Preserve the command-based model for future mutating operations.

#### 3.2 Extract request queue and cancellation logic into focused units

Progress:

- Complete.
- Request tracking, mailboxing, runtime loop, and backend adaptation are distinct internal units with narrower responsibilities.

Remaining tasks:

1. No Phase C catch-up work required.

#### 3.3 Add unit-test seams for agent behavior

Progress:

- Complete.
- There is now a private fakeable backend seam plus focused unit tests for queue full handling, cancellation, shutdown draining, retry exhaustion, loop limits, and command serialization.

Remaining tasks:

1. No Phase C catch-up work required.
2. The only adjacent improvement worth considering is the Phase B test gap around `tool_invocations`, since that behavior lives in the runtime.

## Recommended Catch-Up Sequence

1. Finish Phase A first.
   This is the only phase that still blocks an honest "A-C complete" claim.
2. Fix the package/export story.
   Build-tree and install-tree consumption need to tell the same story.
3. Stop installing vendored `nlohmann` headers.
   This is a direct roadmap miss and also simplifies the package surface.
4. Tighten the target/documentation story.
   Make `ZooKeeper::zoo` the only primary documented target.
5. Add the missing runtime assertions for `Response::tool_invocations`.
   This closes the last meaningful Phase B proof gap.

## Definition of Catch-Up Done

Phases A-C can be treated as fully caught up when all of the following are true:

1. `ZooKeeper::zoo` is the single clearly documented primary consumer target.
2. Build-tree and install-tree package consumption both work and expose the same public target story.
3. Installed headers are limited to the supported zoo public API, without vendored `nlohmann` headers.
4. Documentation no longer contradicts the actual target/package behavior.
5. Runtime tests explicitly prove the contents of `Response::tool_invocations` across core tool execution outcomes.

## Not in Scope for This Catch-Up

The following roadmap work remains outside this document because it belongs to later phases:

- Phase D - Core model decomposition
- Phase E - Config/example ergonomics polish
- Epic 6.3 maintainer-facing architecture note

Those items should remain separate so the current catch-up effort stays focused on making the claimed completed phases actually complete.
