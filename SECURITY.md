# Security Policy

## Reporting a Vulnerability

Do not open a public GitHub issue for potential security vulnerabilities.

Instead, report the issue privately to the maintainers with:

- A description of the affected component and impact.
- Reproduction steps or a proof of concept, if available.
- The versions, platform, and build flags involved.
- Any suggested remediation or mitigation.

If a private reporting channel is not yet configured, contact the repository owner directly and request a secure channel before sending sensitive details.

## Scope

Security reports are especially valuable for:

- Model loading and untrusted file handling.
- Tool execution and prompt-injection boundaries.
- Memory safety, undefined behavior, and concurrency defects.
- Build, install, and package-consumer supply-chain issues.

## Response Expectations

- Reports will be triaged as quickly as possible.
- Valid issues will be acknowledged, reproduced, and tracked privately until a fix is ready.
- Coordinated disclosure after a patch release is preferred.
