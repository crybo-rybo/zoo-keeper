# llama.cpp b8992 Migration Plan

> **Historical document.** This plan was written when llama.cpp was vendored
> as a git submodule at `extern/llama.cpp`. The submodule has since been
> replaced by CMake `FetchContent`; see [UPDATE_LLAMA_CPP.md](UPDATE_LLAMA_CPP.md)
> for the current update process.

## Summary

- Target latest tagged llama.cpp release `b8992` at commit `5cbfb18075c95437e4ac7fb50e3baf88fe137a87`.
- Keep public Zoo-Keeper APIs source-compatible.
- Use the migration to remove Zoo-Keeper-owned Hub download shims that b8992 makes redundant.
- Keep llama.cpp calls inside Core and optional Hub boundaries.

## Implementation Checklist

1. Update `ZOO_LLAMA_TAG` to b8992 and refresh the FetchContent cache.
2. Rename build/package references from `common`/`libcommon.a` to `llama-common`/`libllama-common.a`, including `libllama-common-base.a`.
3. Add a temporary Clang-only CMake workaround for b8992 `common/ngram-mod.cpp` missing `<algorithm>`.
4. Replace legacy tool parser state with `common_chat_parser_params`.
5. Preserve `generation_prompt` from each `common_chat_templates_apply()` result before parsing generated tool-call text.
6. Reject only `COMMON_CHAT_FORMAT_CONTENT_ONLY`; b8992 removed the old generic format.
7. Use `common_download_model()` for Hub repo, repo-file, and raw URL downloads.
8. Remove manual Hub HEAD size probing and duplicate Authorization header plumbing.
9. Let llama.cpp own HuggingFace repo files in its Hugging Face-style cache.
10. Preserve `CachedModelInfo` source compatibility while documenting `size_bytes == 0`.
11. Update migration docs, Hub docs, CMake docs, and llama-update instructions.
12. Verify default, Hub, examples, unit, and packaging-consumer builds.

## Test Gates

- `scripts/build.sh -DZOO_BUILD_TESTS=ON -DZOO_BUILD_HUB=ON -DZOO_BUILD_EXAMPLES=ON`
- `scripts/test.sh`
- install-tree CMake consumer smoke test
- build-tree CMake consumer smoke test
- optional live model smoke:
  - `ZOO_INTEGRATION_MODEL=/path/to/Qwen3-8B-Q4_K_M.gguf scripts/test.sh -R LiveModelIntegrationTest`
  - `build/examples/demo_chat examples/config.example.json`

## Notes

- Do not patch fetched llama.cpp sources; keep compatibility patches in Zoo-Keeper CMake.
- The Hugging Face cache path is now derived by llama.cpp and may be a snapshot path.
- `ModelStore::pull()` should record the best available source URL, but catalog correctness must not depend on being able to derive one.
