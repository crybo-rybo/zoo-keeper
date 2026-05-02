# Maintainer CMake Packaging Notes

This note explains the purpose of `cmake/ZooKeeperConfig.cmake.in` and `cmake/ZooKeeperBuildTreeConfig.cmake.in`, how they are generated, and when each one is actually used.

It is for maintainers and release work, not normal library consumers. For user-facing build instructions, see [building.md](building.md).

## Short Version

- `cmake/ZooKeeperBuildTreeConfig.cmake.in`
  - template for the package config that lives in the build directory
  - supports `find_package(ZooKeeper CONFIG)` against an uninstalled producer build
- `cmake/ZooKeeperConfig.cmake.in`
  - template for the package config that is installed into the package prefix
  - supports `find_package(ZooKeeper CONFIG)` after `cmake --install`

Both exist because "use the library from the build tree" and "use the library from an installed prefix" are similar user stories, but they do not have the same filesystem layout or dependency resolution needs.

## The Three Consumer Modes

There are three main ways another project can consume Zoo-Keeper:

| Consumer mode | Uses package config files? | Primary mechanism |
|---------------|----------------------------|-------------------|
| `add_subdirectory(...)` | No | Targets already exist in the same CMake configure |
| `FetchContent_MakeAvailable(...)` | No | Targets already exist in the same CMake configure |
| `find_package(ZooKeeper CONFIG)` | Yes | CMake loads `ZooKeeperConfig.cmake` from a build tree or install prefix |

That means the two config templates matter only for the `find_package(...)` path.

## High-Level Picture

```text
Same-source-tree consumer
-------------------------
consumer CMakeLists.txt
    |
    +-- add_subdirectory(zoo-keeper)
    |   or
    +-- FetchContent_MakeAvailable(zoo-keeper)
            |
            +-- consumer links directly to existing target: ZooKeeper::zoo


Separate-project consumer
-------------------------
consumer CMakeLists.txt
    |
    +-- find_package(ZooKeeper CONFIG)
            |
            +-- if CMAKE_PREFIX_PATH points at build/
            |      -> build-tree ZooKeeperConfig.cmake is loaded
            |
            +-- if CMAKE_PREFIX_PATH points at install prefix/
                   -> installed ZooKeeperConfig.cmake is loaded
```

## Generation Flow Inside This Repo

```text
configure step
--------------
CMakeLists.txt
    |
    +-- configure_package_config_file(
    |      cmake/ZooKeeperBuildTreeConfig.cmake.in
    |      -> build/ZooKeeperBuildTreeConfig.cmake.in
    |   )
    |
    +-- file(GENERATE
    |      INPUT  build/ZooKeeperBuildTreeConfig.cmake.in
    |      OUTPUT build/ZooKeeperConfig.cmake
    |   )
    |
    +-- configure_package_config_file(
           cmake/ZooKeeperConfig.cmake.in
           -> build/cmake/ZooKeeperConfig.cmake
       )


install step
------------
cmake --install build --prefix <prefix>
    |
    +-- install(EXPORT ZooKeeperTargets ...)
    |      -> <prefix>/lib/cmake/ZooKeeper/ZooKeeperTargets.cmake
    |
    +-- install(FILES build/cmake/ZooKeeperConfig.cmake ...)
    |      -> <prefix>/lib/cmake/ZooKeeper/ZooKeeperConfig.cmake
    |
    +-- install(FILES ZooKeeperConfigVersion.cmake ...)
           -> <prefix>/lib/cmake/ZooKeeper/ZooKeeperConfigVersion.cmake
```

## Why The Build-Tree Path Uses Two Steps

The build-tree config is unusual:

1. `configure_package_config_file(...)` expands `@PACKAGE_INIT@` and `@...@` placeholders.
2. `file(GENERATE ...)` then resolves generator expressions such as:
   - `$<TARGET_FILE:zoo>`
   - `$<TARGET_PROPERTY:zoo,INTERFACE_INCLUDE_DIRECTORIES>`

That is why the repository briefly creates `build/ZooKeeperBuildTreeConfig.cmake.in` and then turns it into the final `build/ZooKeeperConfig.cmake`.

The install-tree config does not need this extra `file(GENERATE ...)` step because it does not embed build-time target file paths directly. It can rely on installed exported targets instead.

## Build-Tree Config: What It Does

Source template: [cmake/ZooKeeperBuildTreeConfig.cmake.in](../cmake/ZooKeeperBuildTreeConfig.cmake.in)

This file creates imported targets that point back into the producer build tree:

- `ZooKeeper::nlohmann_json`
  - points at the fetched JSON headers in the producer build
- `ZooKeeper::zoo`
  - points at the built `libzoo.a` file
  - carries include directories and compile features
  - links transitively to `ZooKeeper::llama` and `ZooKeeper::nlohmann_json`
- `ZooKeeper::llama`
  - points at the built llama/llama-common/ggml archives and platform link flags
- `ZooKeeper::zoo_core`
  - compatibility forwarding target to `ZooKeeper::zoo`

Diagram:

```text
consumer project
    |
    +-- find_package(ZooKeeper CONFIG)
            |
            +-- build/ZooKeeperConfig.cmake
                    |
                    +-- creates imported target ZooKeeper::zoo
                    |       IMPORTED_LOCATION = <producer build>/libzoo.a
                    |       INCLUDE_DIRS      = <producer source>/include + generated headers
                    |
                    +-- creates imported target ZooKeeper::llama
                    +-- creates imported target ZooKeeper::nlohmann_json
```

Use this when:

- you want to smoke-test packaging without installing first
- you want a separate consumer project to point directly at an existing producer build directory

This is what the packaging smoke test under [`tests/packaging/cmake_consumer`](../tests/packaging/cmake_consumer) exercises in CI for the build-tree case.

## Install-Tree Config: What It Does

Source template: [cmake/ZooKeeperConfig.cmake.in](../cmake/ZooKeeperConfig.cmake.in)

This file assumes the project has already been installed and that exported targets exist under the install prefix.

Its job is mostly dependency setup:

- `find_dependency(llama CONFIG)`
- `find_dependency(nlohmann_json CONFIG)`
- create lightweight compatibility shims:
  - `ZooKeeper::llama`
  - `ZooKeeper::nlohmann_json`
- include the installed exported targets file:
  - `ZooKeeperTargets.cmake`

Diagram:

```text
consumer project
    |
    +-- find_package(ZooKeeper CONFIG)
            |
            +-- <prefix>/lib/cmake/ZooKeeper/ZooKeeperConfig.cmake
                    |
                    +-- find_dependency(llama)
                    +-- find_dependency(nlohmann_json)
                    +-- include(ZooKeeperTargets.cmake)
                            |
                            +-- defines imported target ZooKeeper::zoo
```

Use this when:

- you installed Zoo-Keeper into a prefix
- you are validating the packaged install surface
- you are acting like a normal downstream user who only sees the installed package

## Why The Two Files Are Not Identical

They serve different physical layouts:

| Concern | Build-tree config | Install-tree config |
|---------|-------------------|---------------------|
| Where is `libzoo.a`? | In the producer build directory | In the install prefix |
| Where are public headers? | Source tree + generated build headers | Install prefix include dir |
| How are exported targets defined? | Hand-authored imported targets | Installed `ZooKeeperTargets.cmake` |
| How is `nlohmann_json` resolved? | Build-tree shim target to fetched headers | `find_dependency(nlohmann_json)` |
| How is `llama` resolved? | Build-tree shim target to built archives, including `llama-common` | `find_dependency(llama)` plus `libllama-common.a` shim |

So the goal is not byte-for-byte identical files. The goal is that both paths expose the same public consumer story:

- `find_package(ZooKeeper CONFIG REQUIRED)`
- `target_link_libraries(your_target PRIVATE ZooKeeper::zoo)`

## Related Generated Files

- `build/ZooKeeperConfig.cmake`
  - final build-tree package config
- `build/cmake/ZooKeeperConfig.cmake`
  - final install-tree package config before installation
- `build/ZooKeeperConfigVersion.cmake`
  - version compatibility checks for `find_package`
- `<prefix>/lib/cmake/ZooKeeper/ZooKeeperTargets.cmake`
  - exported installed targets

## How To Think About Them

Good mental model:

```text
ZooKeeperBuildTreeConfig.cmake.in
    = "Pretend this build directory is a package prefix"

ZooKeeperConfig.cmake.in
    = "Load the actual installed package correctly"
```

If you keep that distinction in mind, most of the rest follows naturally.

## Practical Debug Checklist

If a package consumer fails, check these in order:

1. Which consumption path is it using?
   - same-configure `add_subdirectory` / `FetchContent`
   - build-tree `find_package`
   - install-tree `find_package`
2. Which `ZooKeeperConfig.cmake` did CMake load?
3. For install-tree failures, are `llama` and `nlohmann_json` discoverable?
4. For build-tree failures, does the generated `build/ZooKeeperConfig.cmake` point at the expected producer artifacts?
5. Does the consumer only link `ZooKeeper::zoo`, or is it reaching around the package surface?

## See Also

- [building.md](building.md) -- user-facing build and install guidance
- [compatibility.md](compatibility.md) -- supported public packaging boundary
- [tests/packaging/cmake_consumer/CMakeLists.txt](../tests/packaging/cmake_consumer/CMakeLists.txt) -- minimal smoke consumer
