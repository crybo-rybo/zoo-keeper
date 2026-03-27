# Compatibility Policy

This document describes the current compatibility boundary for Zoo-Keeper's
published release surface.

## Supported Public Boundary

The supported public API is:

- installed headers under `include/zoo/`
- the primary CMake target `ZooKeeper::zoo`
- the public runtime, model, tool, and value types documented in the user docs

The following are not part of the compatibility boundary:

- headers under `include/zoo/internal/`
- implementation files under `src/`
- private CMake/package plumbing
- internal planning notes and draft documents under `docs/` that are not part of
  the published user-facing reference set

## Current Compatibility Expectations

- Public names and include paths documented here should remain stable within the
  normal release line.
- Behavior described in the user-facing docs is the supported contract.
- Undocumented implementation details may change without notice.
- `ZooKeeper::zoo` is the primary supported consumer target.
- `ZooKeeper::zoo_core` may remain as a compatibility shim, but it is not the
  primary user story.

## Release Guidance

When changing the supported public API:

- update the affected docs first
- call out the change in the release notes
- keep examples, guides, and architecture pages aligned with the shipped API

## Verification Before Release

Before cutting a release, verify:

- `scripts/test.sh`
- build-tree CMake consumer smoke
- installed-package CMake consumer smoke
- any optional live-model smoke selected for the release candidate
