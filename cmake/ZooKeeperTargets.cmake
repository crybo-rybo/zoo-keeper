include_guard(GLOBAL)

include(${PROJECT_SOURCE_DIR}/cmake/ZooKeeperLlama.cmake)

configure_file(
    ${PROJECT_SOURCE_DIR}/include/zoo/version.hpp.in
    ${PROJECT_BINARY_DIR}/generated/version.hpp
    @ONLY
)

add_library(zoo STATIC
    ${PROJECT_SOURCE_DIR}/src/agent/agent_facade.cpp
    ${PROJECT_SOURCE_DIR}/src/agent/backend_model.cpp
    ${PROJECT_SOURCE_DIR}/src/agent/runtime.cpp
    ${PROJECT_SOURCE_DIR}/src/agent/runtime_tool_loop.cpp
    ${PROJECT_SOURCE_DIR}/src/agent/runtime_extraction.cpp
    ${PROJECT_SOURCE_DIR}/src/core/model.cpp
    ${PROJECT_SOURCE_DIR}/src/core/model_init.cpp
    ${PROJECT_SOURCE_DIR}/src/core/model_inference.cpp
    ${PROJECT_SOURCE_DIR}/src/core/model_prompt.cpp
    ${PROJECT_SOURCE_DIR}/src/core/model_history.cpp
    ${PROJECT_SOURCE_DIR}/src/core/model_sampling.cpp
    ${PROJECT_SOURCE_DIR}/src/core/model_tool_calling.cpp
    ${PROJECT_SOURCE_DIR}/src/core/stream_filter.cpp
    ${PROJECT_SOURCE_DIR}/src/log_callback.cpp
)
target_include_directories(zoo PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
    $<BUILD_INTERFACE:${json_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(zoo PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(zoo PUBLIC cxx_std_23)
zoo_target_link_llama(zoo)
set_property(TARGET zoo APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "$<INSTALL_INTERFACE:ZooKeeper::nlohmann_json>"
)

if(ZOO_BUILD_HUB)
    target_sources(zoo PRIVATE
        ${PROJECT_SOURCE_DIR}/src/hub/inspector.cpp
        ${PROJECT_SOURCE_DIR}/src/hub/huggingface.cpp
        ${PROJECT_SOURCE_DIR}/src/hub/store.cpp
    )
    target_compile_definitions(zoo PUBLIC ZOO_HUB_ENABLED)
endif()

if(ZOO_ENABLE_LOGGING)
    target_compile_definitions(zoo PRIVATE ZOO_LOGGING_ENABLED)
endif()

add_library(zoo_core INTERFACE)
target_link_libraries(zoo_core INTERFACE zoo)

add_library(ZooKeeper::zoo ALIAS zoo)
add_library(ZooKeeper::zoo_core ALIAS zoo_core)

zoo_apply_strict_target_options(zoo)
