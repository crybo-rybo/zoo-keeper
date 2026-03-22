cmake_minimum_required(VERSION 3.18)

add_executable(zoo_benchmarks
    perf_harness.cpp
)

target_link_libraries(zoo_benchmarks
    PRIVATE
        zoo
)

target_include_directories(zoo_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/llama.cpp/common
        ${CMAKE_SOURCE_DIR}/extern/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/extern/llama.cpp/include
)

zoo_set_warnings(zoo_benchmarks PRIVATE)
zoo_enable_sanitizers(zoo_benchmarks)

install(TARGETS zoo_benchmarks
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
