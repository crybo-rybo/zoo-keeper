# FetchContent llama.cpp Dependency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let downstream CMake consumers use `zoo-keeper` through `FetchContent` without depending on submodule initialization, while preserving the vendored `llama.cpp` workflow for maintainers.

**Architecture:** Teach the dependency layer to resolve `llama.cpp` from an existing target, an installed package, the vendored submodule, or an opt-in internal `FetchContent` fallback. Keep the rest of the build using the same `llama`/`common` targets so public packaging behavior stays stable. Add a dedicated downstream `FetchContent` smoke test and CI coverage.

**Tech Stack:** CMake 3.18+, C++23, GitHub Actions, llama.cpp, nlohmann/json

---

### Task 1: Update dependency resolution in CMake

**Files:**
- Modify: `cmake/FetchDependencies.cmake`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing consumer coverage**

Add a new packaging smoke test configuration that consumes the repository via
`FetchContent` with no submodule assumptions.

- [ ] **Step 2: Run the targeted packaging workflow locally to verify the gap**

Run a focused CMake configure/build for the new smoke test and confirm it fails
before the dependency-resolution change.

- [ ] **Step 3: Implement the minimal dependency resolution changes**

Add `ZOO_FETCH_LLAMA`, `ZOO_LLAMA_REPOSITORY`, and `ZOO_LLAMA_TAG` options,
then resolve `llama.cpp` in this order: existing `llama` and `common`
targets, vendored submodule, opt-in internal `FetchContent`, hard error.

- [ ] **Step 4: Re-run the targeted packaging workflow**

Confirm the new consumer path configures and builds successfully.

### Task 2: Document downstream consumption

**Files:**
- Modify: `README.md`
- Modify: `docs/building.md`

- [ ] **Step 1: Update the FetchContent examples**

Document the recommended downstream usage and explain when
`-DZOO_FETCH_LLAMA=ON` is needed.

- [ ] **Step 2: Re-read docs for consistency**

Check that the new wording matches the actual resolution order and does not
contradict the maintainer submodule workflow.

### Task 3: Extend CI coverage

**Files:**
- Modify: `.github/workflows/build_and_test.yml`
- Create: `tests/packaging/fetchcontent_consumer/CMakeLists.txt`
- Create: `tests/packaging/fetchcontent_consumer/main.cpp`

- [ ] **Step 1: Add the downstream FetchContent smoke test**

Create a tiny consumer project that fetches `zoo-keeper`, enables
`ZOO_FETCH_LLAMA`, and links `ZooKeeper::zoo`.

- [ ] **Step 2: Update CI to execute the smoke test**

Add a workflow step that configures and builds the new consumer.

- [ ] **Step 3: Run verification**

Run the relevant local build/test commands and inspect the diff before commit.
