# Compatibility Policy

This document describes the intended compatibility contract for Zoo-Keeper 1.0 and later. Before 1.0, changes may still land, but new work should already preserve this boundary unless there is a strong reason not to.

## Public Boundary

The supported public API is:

- installed headers under `include/zoo/`
- the primary CMake target `ZooKeeper::zoo`
- the public runtime/model/tool types documented in the README and docs

The following are not part of the compatibility contract:

- headers under `include/zoo/internal/`
- implementation files under `src/`
- private CMake/package plumbing
- internal audit notes and temporary planning documents under `docs/` that are not part of the published user-facing reference set

## Intended 1.x Guarantees

For 1.x releases, Zoo-Keeper should preserve source compatibility for normal consumers that stay within the public boundary.

That means:

- existing public headers keep compatible names and include paths
- existing public types and functions are not removed or behaviorally repurposed in a minor or patch release
- `ZooKeeper::zoo` remains the primary supported consumer target throughout 1.x
- `ZooKeeper::zoo_core` may remain as a compatibility shim, but it is not part of the primary user story

Breaking changes to the supported public API require a new major version.

## Deprecation Rules

When a public API needs to change after 1.0:

- mark the old API as deprecated in docs and release notes first
- keep the deprecated path available for at least one minor release unless there is a security or correctness reason not to
- document the replacement path clearly in the changelog and migration notes

## Release Validation

Before cutting 1.0 or later releases, verify:

- `ctest --test-dir build --output-on-failure`
- build-tree CMake consumer smoke
- installed-package CMake consumer smoke
- any optional live-model smoke selected for the release candidate
