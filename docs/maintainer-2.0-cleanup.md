# Zoo-Keeper 2.0 Maintainer Cleanup Findings

## Purpose

This document records maintainer cleanup findings and recommended fixes for the
2.0 line. It is an audit and sequencing document, not approval for a single
large cleanup PR.

The findings below should be used to create small, reviewable changes. The
normal changeset discipline still applies:

- one concern per PR
- keep non-test/non-doc additions under the project limit
- prefer deleting or modifying existing code over adding new abstractions
- do not bundle tactical fixes with broad redesigns
- every high-severity fix must include a regression test that would fail on the
  current code

If a finding genuinely cannot be fixed without a larger redesign, land a short
design note first, then implement it in separately reviewable steps.

## Severity Guide

- **HIGH**: can hang, corrupt memory, create a security issue, silently produce
  wrong behavior in a core workflow, or break package consumers in a way users
  cannot diagnose.
- **MED**: causes maintainability drag, confusing APIs, incomplete robustness,
  or package friction with a workaround.
- **LOW**: cleanup, documentation drift, or polish work that should not block a
  release.

## High-Severity Findings

### H1. Agent shutdown can block forever on a tool handler

**Area:** Agent runtime, tool execution

**Problem:** `AgentRuntime::stop()` and `~AgentRuntime()` join the inference
thread. The inference thread can block indefinitely on a tool future while user
tool code runs. A slow or stuck handler can therefore hang shutdown.

**Why it matters:** This is a lifecycle correctness bug. Library users must be
able to destroy an agent without trusting arbitrary tool code to return.

**Minimal fix:** Replace unconditional waiting on tool futures with a
shutdown-aware wait path. On request cancellation or runtime stop, the inference
thread must stop waiting, resolve the request with `RequestCancelled`, and allow
runtime shutdown to complete.

**Implementation standard:**

- Do not make a detached, permanently blocked single worker the normal executor
  state. That can poison the executor for later tool calls.
- If a legacy handler cannot be killed, the executor must either abandon that
  worker and replace it or otherwise prove subsequent tool calls are not queued
  behind the blocked handler.
- Any detached work must not access request slots, runtime members, callbacks,
  backend state, or stack-owned data after cancellation.
- Prefer an explicit cancellable tool context for new handlers over hidden
  global state.

**Acceptance criteria:**

- `stop()` returns within a bounded time while a handler is blocked.
- `~AgentRuntime()` returns within a bounded time while a handler is blocked.
- Cancelling one blocked tool request does not prevent a later request on the
  same runtime from executing a different tool.
- The abandoned handler result is ignored safely if it eventually returns.
- ASan/UBSan-friendly tests cover the shutdown and executor-recovery paths.

### H2. Streaming callbacks can outlive their request slot

**Area:** Agent runtime, callback dispatcher, request slots

**Problem:** Async streaming callback work can be queued with a raw pointer into
request-slot-owned callback storage. If the slot is resolved and released before
queued callback work finishes, the dispatcher can dereference freed memory.

**Why it matters:** This is a potential use-after-free in a public async path.

**Minimal fix:** Make queued callback work own or share stable callback state,
or otherwise make the callback lifetime independent from the request slot. A
drain-before-resolve rule is useful as a synchronization point, but it should be
a defense, not the only lifetime guarantee.

**Implementation standard:**

- No queued dispatcher entry should store a raw pointer to slot-owned callback
  storage unless the type system proves the slot cannot be released first.
- Callback exceptions should be reported on the owning request when possible.
- Dispatcher drain and shutdown behavior must be bounded or cancellation-aware.

**Acceptance criteria:**

- A void-returning streaming callback that runs asynchronously cannot outlive
  destroyed slot storage.
- A throwing streaming callback fails the request with a clear error.
- Runtime shutdown while callbacks are queued cannot hang indefinitely.
- Tests cover the error-rethrow path that originally exposed the hazard.

### H3. Only the first tool call in an assistant turn is executed

**Area:** Agent tool loop

**Problem:** When a model emits multiple structured tool calls in one assistant
turn, the runtime executes only the first call. Remaining calls are preserved in
history but never invoked.

**Why it matters:** Multi-call tool turns are normal for several tool-calling
formats. Dropping sibling calls silently produces incorrect agent behavior.

**Minimal fix:** Process every structured tool call emitted by a single
assistant turn before asking the model for the next turn. Append one assistant
message containing all tool calls, then append one tool-result message per call.

**Implementation standard:**

- Preserve call order.
- Validation failure for one call should not automatically skip valid siblings.
- Retry accounting should be per tool or per call shape, not a hidden global
  side effect of the first call.
- Keep the existing public `chat()` and `complete()` result shape unchanged.

**Acceptance criteria:**

- A scripted backend emitting two or more tool calls in one generation invokes
  every handler exactly once.
- The conversation history contains the assistant turn and all corresponding
  tool messages in order.
- Tool traces record every invocation.
- A malformed sibling call records a validation failure without suppressing
  valid sibling calls.

### H4. GBNF grammar can allow invalid JSON

**Area:** Tools grammar

**Problem:** Primitive grammar rules can accept JSON-like output that
`nlohmann::json::parse` rejects, including raw control characters in strings and
integer forms with leading zeros.

**Why it matters:** Grammar-constrained output should not pass the decoder and
then fail parser validation. That creates hard-to-debug model behavior.

**Minimal fix:** Tighten the JSON primitive grammar to match the supported JSON
subset:

- integers use `0` or a non-zero leading digit followed by digits
- numbers use the same integer prefix
- string bodies reject raw U+0000 through U+001F
- standard escapes, including `\uXXXX`, are accepted
- numeric enum literals are emitted canonically

**Acceptance criteria:**

- Unit tests assert the generated grammar shape.
- Behavioral tests, where feasible, validate representative accepted and
  rejected JSON snippets against the grammar and parser together.
- Existing schema grammar output remains deterministic.

### H5. Integer validation rejects common LLM integer output

**Area:** Tools validation and typed invocation

**Problem:** JSON Schema `integer` validation rejects values shaped like `3.0`.
Many LLMs emit integral values as floats even when the schema asks for an
integer.

**Severity note:** This is a real correctness issue, but it is lower risk than
the lifecycle and memory-safety findings. Treat it as high only when it blocks
tool calling for supported models.

**Minimal fix:** Accept finite floating-point JSON numbers for integer
parameters only when they are exactly integral and can be converted safely to
the destination integer type.

**Implementation standard:**

- Do not accept arbitrary `double` values near `int64_t` limits and rely on
  implementation-defined or library-specific conversion behavior.
- Normalize accepted integral floats before typed handler invocation, or keep
  accepted values inside the exactly representable integer range.
- Add explicit boundary tests for large values, negative values, NaN, infinity,
  and fractional values.

**Acceptance criteria:**

- `3.0`, `-42.0`, and `0.0` pass integer validation.
- `3.14`, NaN, and infinity fail.
- Values outside the safe conversion range fail.
- Strongly typed `int` handlers receive the intended value without silent clamp
  or wrap behavior.

### H6. llama.cpp FetchContent archive is not hash-verified

**Area:** Build supply chain

**Problem:** The build downloads a llama.cpp archive without `URL_HASH`.
Unverified source downloads are not acceptable for repeatable builds.

**Why it matters:** A build should fail closed if the source archive is missing
or unexpected. GitHub-generated archives are also not a strong long-term
integrity boundary unless the expected digest is recorded.

**Minimal fix:** Require a SHA-256 hash for any downloaded llama.cpp archive.
The default pinned tag may have a known-good hash in CMake. Any custom tag or
URL must provide an explicit hash.

**Acceptance criteria:**

- Default configure verifies the default llama.cpp archive hash.
- Custom `ZOO_LLAMA_TAG` without `ZOO_LLAMA_ARCHIVE_SHA256` fails at configure
  time with an actionable error.
- Custom tag with a correct hash configures successfully.
- The pinned tag and hash live close enough together that future bumps are hard
  to forget.

### H7. Sanitizer and coverage link options leak to consumers

**Area:** Build/package metadata

**Problem:** Sanitizer and coverage link options can be exposed through
interface link options, leaking project-internal instrumentation into consumers
or fetched dependencies.

**Severity note:** Usually medium, but high when it breaks downstream package
consumption or contaminates dependency builds.

**Minimal fix:** Apply sanitizer and coverage compile/link options privately to
zoo-owned targets. Do not export instrumentation flags through installed or
build-tree package interfaces.

**Acceptance criteria:**

- Zoo-owned sanitizer and coverage builds still work.
- A downstream FetchContent or installed-package smoke project does not inherit
  sanitizer or coverage link options unless it explicitly asks for them.
- Package metadata does not contain project-internal instrumentation flags.

### H8. Installed package does not verify the llama.cpp build it was built with

**Area:** Installed CMake package

**Problem:** `find_dependency(llama CONFIG)` is unversioned. An installed
Zoo-Keeper package can be paired with a mismatched llama.cpp package.

**Why it matters:** Mismatched llama.cpp build artifacts can produce confusing
link or runtime failures for package consumers.

**Minimal fix:** Verify the llama.cpp build identity in the generated
`ZooKeeperConfig.cmake` using metadata exported by llama.cpp. Because upstream
llama.cpp package version files do not follow the usual CMake version-file
shape, this must be tested against the actual installed package.

**Implementation standard:**

- Prefer native CMake version checks when upstream supports them.
- If checking `LLAMA_BUILD_NUMBER` manually, fail only when the expected and
  actual values are both meaningful.
- Avoid brittle assumptions about file names or variables unless covered by an
  install-tree smoke test.

**Acceptance criteria:**

- A normal installed consumer configures and links.
- A deliberately mismatched or tampered llama build number fails at configure
  time with a clear error.
- Missing upstream metadata is handled intentionally, either as a clear error or
  a documented compatibility fallback.

### H9. HuggingFace filename component allows path traversal

**Area:** Hub security

**Problem:** Model identifiers with an explicit filename component can accept
path traversal shapes such as `owner/repo::../../etc/passwd`.

**Why it matters:** Any user-controlled path component that is later joined into
cache or destination paths must be constrained before filesystem use.

**Minimal fix:** Treat the explicit filename as a single path segment. Reject
empty names, `.`, `..`, NUL bytes, `/`, and `\`.

**Acceptance criteria:**

- Traversal and separator cases are rejected by `parse_identifier`.
- Valid single-segment GGUF filenames are still accepted.
- Download paths are validated to remain under the intended cache or
  destination root when that root is controlled by Zoo-Keeper.

## Medium-Severity Findings

These findings are worth fixing, but they should not be bundled with high-risk
runtime or supply-chain fixes.

- `Message` / `ToolCallInfo` aliases appear transitional but are now the
  practical public names.
- `ToolCallView` / `ToolCallSpan` have little production value if no producer
  uses borrowed tool-call spans.
- `AgentBackend::replace_history`, `AgentRuntime::extract(json, string_view)`,
  `trim_history_to_fit`, `PromptState::dirty`, and `AsyncTextCallback` appear
  unused or test-only.
- `GenerationResult` and parsed-tool-response types are duplicated between core
  and agent internals.
- The public `Agent` API has several three-overload command patterns that hide
  errors in void-returning convenience methods.
- `ScopeExit` uses `std::function` and has surprising move-assignment
  semantics.
- Auto-configured GPU offload does not appear to account for context-size-driven
  KV-cache pressure.
- `GgufInspector` and `SystemProbe` are documented as Hub-adjacent but live in
  core.
- `core/json.hpp` can throw on parse/schema errors despite the project-wide
  preference for `Expected`.
- Hub catalog writes lack advisory locking and version validation.
- Tests link directly against `llama-common`, which weakens layer-boundary
  checks.
- README, CHANGELOG, and maintainer docs have drifted from current build and
  API behavior.

## Recommended Landing Plan

Do not land the high findings as one broad cleanup PR. The right shape is a
series of small PRs with narrow blast radius.

### PR 1: Documentation only

Land this findings document. No code changes.

### PR 2: Build supply-chain hardening

Fix H6 only. Add the llama archive hash requirement and configure-time tests or
documented smoke commands. Avoid sanitizer, coverage, package-version, or other
build cleanups in the same PR.

### PR 3: Package instrumentation hygiene

Fix H7 only. Move sanitizer and coverage instrumentation off exported
interfaces. Verify with a downstream smoke project.

### PR 4: Installed package llama identity check

Fix H8 only. Use the smallest install-config check that works with the actual
llama.cpp package. Include a normal consumer smoke test and a mismatch/tamper
test.

### PR 5: Hub filename traversal

Fix H9 only. Keep it to identifier validation and path-root containment if that
code path is already local to the hub layer. Retry policies, catalog locking,
and HTTP status cleanup should be separate follow-ups.

### PR 6: Grammar correctness

Fix H4 only. Tighten primitive rules and add focused grammar/parser tests.

### PR 7: Integer validation compatibility

Fix H5 only. Accept safe integral floats and normalize or reject unsafe
boundaries before typed invocation.

### PR 8: Streaming callback lifetime

Fix H2 only. Replace raw queued callback pointers with owned/shared callback
state or an equivalent lifetime-proof design. Include shutdown and exception
tests.

### PR 9: Tool-handler shutdown semantics

Fix H1 only. Make tool waits cancellation-aware and prove a cancelled blocked
handler does not poison later tool execution. This is the hardest fix and should
not be mixed with the multi-tool loop.

### PR 10: Multi-tool turn execution

Fix H3 only after the cancellation and callback lifetime work is settled.
Executing all tool calls per turn is conceptually simple, but it sits in the
same runtime code and should not obscure concurrency review.

### Later 2.0 Cleanup PRs

After the high findings are closed, consider the type/API/layer cleanup items
as separate 2.0 design work. Those changes may be breaking, but they should
still be split by concern:

- remove or bless transitional public aliases
- simplify command overloads
- delete dead internals
- resolve core/config/hub layer ownership
- make JSON config loading consistently return `Expected`
- improve auto-configure VRAM math

## Design Guidance

### Agent Runtime

Prefer ownership and bounded waits over ordering assumptions. A design that
depends on "we always drain before resolving" is fragile unless the queued work
also owns safe state.

Avoid using detached threads as the primary lifecycle mechanism. Detach may be a
last-resort escape hatch for legacy user code, but normal operation should keep
worker ownership explicit and testable.

If tool cancellation becomes public API, expose a small task context rather than
a raw internal token. A context can later carry request id, deadline, logger, or
trace metadata without another public signature break.

### Tool Loop

Keep the loop model simple:

1. generate one assistant turn
2. append that assistant turn
3. execute every tool call from that turn in order
4. append one tool result per call
5. generate the next assistant turn

Do not combine this with type-system cleanup or backend type unification. Those
may be good ideas, but they are not required to fix dropped tool calls.

### Tools Validation

The schema layer should define both validation and conversion semantics. If
validation accepts `3.0` for an integer, invocation must also convert it safely
and predictably. Validation that accepts a value later clamped or wrapped by
`get<int>()` is worse than rejection.

### Hub

Keep security fixes narrow. Filename validation and destination-root containment
are security work. Retry budgets, catalog durability, version validation, and
HTTP status cleanup are robustness work and should land separately.

### Build and Packaging

Build-system fixes need downstream smoke tests. A change that works in the
top-level build can still break FetchContent or installed-package consumers.
Keep each package concern isolated so failures are easy to bisect.

## Verification Matrix

Every implementation PR should list the commands it ran and the behavior its
tests prove.

| Area | Required checks |
| --- | --- |
| Docs only | `git diff --check` |
| Grammar / validation | `scripts/build.sh -DZOO_BUILD_TESTS=ON`, focused unit tests |
| Agent concurrency | unit tests, ASan/UBSan build, bounded-time shutdown tests |
| Hub security | hub unit tests with traversal cases |
| Build supply chain | clean configure from empty build directory, hash failure case |
| Installed package | install-tree consumer smoke, mismatch/tamper failure case |
| Integration-impacting agent changes | live integration tests from `.secret/integration-testing.md` |

## Open Questions

- Should cancellable tools use `TaskContext` in the public API, or remain an
  internal best-effort cancellation mechanism for 2.0?
- If a legacy handler blocks forever, should the executor replace the abandoned
  worker, create per-call workers, or require a configured tool timeout?
- Should `max_tool_iterations` continue to count model turns only, or should
  there also be a `max_tool_calls_per_turn` guard?
- Is a new `zoo::config` layer worth the namespace churn, or should docs simply
  acknowledge that core owns GGUF inspection and system probing?
- Should void-returning `Agent` command overloads be removed in 2.0 to force
  callers to observe `Expected` errors?
