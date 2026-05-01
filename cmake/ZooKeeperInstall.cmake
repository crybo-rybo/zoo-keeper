include_guard(GLOBAL)

if(NOT ZOO_ENABLE_INSTALL)
    return()
endif()

include(${PROJECT_SOURCE_DIR}/cmake/ZooKeeperLlama.cmake)

install(TARGETS zoo zoo_core
    EXPORT ZooKeeperTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

zoo_install_llama_common()

install(EXPORT ZooKeeperTargets
    FILE ZooKeeperTargets.cmake
    NAMESPACE ZooKeeper::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZooKeeper
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/zoo DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "internal" EXCLUDE
    PATTERN "*.in" EXCLUDE
)
install(FILES ${PROJECT_BINARY_DIR}/generated/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zoo
)
