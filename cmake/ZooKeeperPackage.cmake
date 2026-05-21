include_guard(GLOBAL)

include(CMakePackageConfigHelpers)
include("${CMAKE_CURRENT_LIST_DIR}/ZooKeeperLlama.cmake")

zoo_collect_llama_build_link_libraries(ZOO_BUILD_LLAMA_LINK_LIBRARIES)
set(ZOO_BUILD_ZOO_LOCATION "$<TARGET_FILE:zoo>")
set(ZOO_BUILD_INTERFACE_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:zoo,INTERFACE_INCLUDE_DIRECTORIES>")
set(ZOO_BUILD_INTERFACE_LINK_OPTIONS "$<TARGET_PROPERTY:zoo,INTERFACE_LINK_OPTIONS>")
set(ZOO_BUILD_GGML_OPENMP_ENABLED "${GGML_OPENMP_ENABLED}")
set(ZOO_BUILD_GGML_BLAS_ENABLED "${GGML_BLAS}")
set(ZOO_BUILD_BLAS_LIBRARIES "${BLAS_LIBRARIES}")
set(ZOO_BUILD_BLAS_LINKER_FLAGS "${BLAS_LINKER_FLAGS}")
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/ZooKeeperBuildTreeConfig.cmake.in
    ${PROJECT_BINARY_DIR}/ZooKeeperBuildTreeConfig.cmake.in
    INSTALL_DESTINATION ${PROJECT_BINARY_DIR}
)
file(GENERATE
    OUTPUT "${PROJECT_BINARY_DIR}/ZooKeeperConfig.cmake"
    INPUT "${PROJECT_BINARY_DIR}/ZooKeeperBuildTreeConfig.cmake.in"
)

if(NOT ZOO_ENABLE_INSTALL)
    return()
endif()

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/ZooKeeperConfig.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/ZooKeeperConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZooKeeper
)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/ZooKeeperConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${PROJECT_BINARY_DIR}/cmake/ZooKeeperConfig.cmake
    ${PROJECT_BINARY_DIR}/ZooKeeperConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZooKeeper
)

zoo_collect_llama_pkgconfig_libs(ZOO_PKGCONFIG_LIBS)
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/zoo-keeper.pc.in
    ${PROJECT_BINARY_DIR}/zoo-keeper.pc
    @ONLY
)

install(FILES
    ${PROJECT_BINARY_DIR}/zoo-keeper.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
