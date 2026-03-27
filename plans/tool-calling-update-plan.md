# Archived Plan: Native Tool Calling Update

Historical note: this plan described the tool-calling redesign that is already present
in HEAD. It is retained for context only and should not be treated as current release
guidance.

## Status

Completed and superseded by the implementation.

## What Shipped

- Template-driven tool calling now comes from llama.cpp chat templates.
- The old `<tool_call>` sentinel path is no longer the primary runtime story.
- Schema extraction remains a separate grammar path.
- Tool-calling setup and parsing now live on the core model boundary.

## Current References

- `include/zoo/core/model.hpp`
- `include/zoo/agent.hpp`
- `docs/adr/006-native-only-tool-calling.md`
- `examples/demo_chat.cpp`

## Notes

If you are looking for the active implementation shape, follow the public headers and
the current examples instead of this archived plan.
