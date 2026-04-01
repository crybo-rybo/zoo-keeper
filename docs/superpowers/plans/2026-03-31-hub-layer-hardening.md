# Hub Layer Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the reviewed hub-layer defects so the implementation is contract-honest, collision-safe, and covered by focused tests.

**Architecture:** Keep `zoo::hub` as an optional layer, but harden the current design instead of expanding it. Add regression tests first, then make minimal production changes: restore process-global logger state after inspection, namespace download targets deterministically, validate aliases consistently, persist the metadata the API says it stores, and narrow misleading API/docs to match real behavior.

**Tech Stack:** C++23, GoogleTest, CMake, llama.cpp common/download APIs, nlohmann/json

---

### Task 1: Cover ModelStore invariants and catalog persistence

**Files:**
- Modify: `tests/unit/test_hub.cpp`
- Modify: `src/hub/store.cpp`
- Modify: `src/hub/store_json.hpp`

- [ ] **Step 1: Write failing tests for alias invariants and metadata/source persistence**

Add tests that:
- open a temporary store directory
- write a catalog JSON containing `metadata`, `source_url`, and `huggingface_repo`
- reopen the store and assert the fields survive round-trip
- assert duplicate or empty aliases are rejected consistently when loading or adding

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `ctest --test-dir build --output-on-failure -R "Hub|ModelStore"`
Expected: failures showing missing metadata/source round-trip or missing alias validation

- [ ] **Step 3: Implement minimal store/catalog fixes**

Update:
- JSON serialization to include `ModelInfo.metadata`
- `ModelStore::add()` / related helpers to reject empty and duplicate aliases before storing entries
- catalog loading validation as needed so persisted aliases obey the same invariant

- [ ] **Step 4: Re-run targeted tests to verify they pass**

Run: `ctest --test-dir build --output-on-failure -R "Hub|ModelStore"`
Expected: targeted hub/store tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_hub.cpp src/hub/store.cpp src/hub/store_json.hpp docs/superpowers/plans/2026-03-31-hub-layer-hardening.md
git commit -m "harden hub store invariants"
```

### Task 2: Make downloads collision-safe and record source data honestly

**Files:**
- Modify: `tests/unit/test_hub.cpp`
- Modify: `src/hub/huggingface.cpp`
- Modify: `src/hub/store.cpp`
- Modify: `include/zoo/hub/types.hpp`
- Modify: `include/zoo/hub/huggingface.hpp`

- [ ] **Step 1: Write failing tests for namespaced download paths and source annotation**

Add tests that:
- exercise the deterministic path-building logic for repo downloads and explicit file pulls
- assert different repos producing the same filename map to different local destinations
- assert pulled entries preserve `source_url` and `huggingface_repo`

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `ctest --test-dir build --output-on-failure -R "Hub"`
Expected: failures showing collisions or missing source annotation

- [ ] **Step 3: Implement minimal download/path fixes**

Update:
- HuggingFace download destination construction to use repo-qualified paths
- store pull path construction to preserve repo/subdirectory identity without collisions
- pull annotation so `source_url` and `huggingface_repo` are both recorded when known

- [ ] **Step 4: Re-run targeted tests to verify they pass**

Run: `ctest --test-dir build --output-on-failure -R "Hub"`
Expected: hub tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_hub.cpp src/hub/huggingface.cpp src/hub/store.cpp include/zoo/hub/types.hpp include/zoo/hub/huggingface.hpp
git commit -m "fix hub download path collisions"
```

### Task 3: Remove global logger side effects from inspection

**Files:**
- Modify: `tests/unit/test_hub.cpp`
- Modify: `src/hub/inspector.cpp`

- [ ] **Step 1: Write a failing regression test for logger restoration**

Add a test that:
- installs a sentinel llama logger callback
- invokes the new logger-guarded inspection path on a controllable failure/smoke path
- asserts the original callback is restored afterwards

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `ctest --test-dir build --output-on-failure -R "Hub"`
Expected: failure showing the logger remains replaced after inspection

- [ ] **Step 3: Implement the minimal logger guard**

Update inspection to:
- capture the current llama logger before suppression
- restore it on every exit path, including model-load failure

- [ ] **Step 4: Re-run targeted tests to verify they pass**

Run: `ctest --test-dir build --output-on-failure -R "Hub"`
Expected: hub tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_hub.cpp src/hub/inspector.cpp
git commit -m "restore llama logger after inspection"
```

### Task 4: Make the hub public contract honest

**Files:**
- Modify: `tests/unit/test_hub.cpp`
- Modify: `include/zoo/hub/huggingface.hpp`
- Modify: `include/zoo/hub/store.hpp`
- Modify: `include/zoo/hub/types.hpp`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `include/zoo/zoo.hpp`

- [ ] **Step 1: Write failing tests for the remaining contract-honest behavior that can be automated**

Add tests that:
- cover any exposed helper semantics retained after the API cleanup
- drop tests that only validate dead config knobs once the contract is narrowed

- [ ] **Step 2: Run targeted tests to verify they fail where behavior/doc comments are still inconsistent**

Run: `ctest --test-dir build --output-on-failure -R "Hub"`
Expected: failures only where automated contract behavior still mismatches

- [ ] **Step 3: Narrow the public contract to real behavior**

Update headers/docs to:
- remove or rename misleading config/options/docs that are not actually implemented
- describe HuggingFace resolution as resolving a file, not enumerating a repo
- remove progress-callback promises until they are actually supported
- refresh umbrella/doc text so the four-layer architecture is described consistently

- [ ] **Step 4: Re-run targeted tests to verify they pass**

Run: `ctest --test-dir build --output-on-failure -R "Hub"`
Expected: hub tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_hub.cpp include/zoo/hub/huggingface.hpp include/zoo/hub/store.hpp include/zoo/hub/types.hpp README.md docs/architecture.md include/zoo/zoo.hpp
git commit -m "tighten hub public contract"
```

### Task 5: Full verification

**Files:**
- No code changes required unless verification exposes regressions

- [ ] **Step 1: Run hub-enabled build**

Run: `scripts/build.sh -DZOO_BUILD_HUB=ON`
Expected: build succeeds with no new warnings promoted to errors

- [ ] **Step 2: Run full tests**

Run: `scripts/test.sh`
Expected: all enabled tests pass; live model tests may remain skipped without `ZOO_INTEGRATION_MODEL`

- [ ] **Step 3: Run formatting if needed**

Run: `scripts/format.sh`
Expected: no diff, or only intentional formatting updates

- [ ] **Step 4: Re-run full tests if formatting changed files**

Run: `scripts/test.sh`
Expected: still green

- [ ] **Step 5: Commit verification-only adjustments if needed**

```bash
git add -A
git commit -m "format hub hardening changes"
```
