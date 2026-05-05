include_guard(GLOBAL)

if(NOT ZOO_BUILD_DOCS)
    return()
endif()

find_package(Doxygen REQUIRED)
find_program(ZOO_DOXYGEN_DOT_EXECUTABLE NAMES dot)

set(ZOO_DOXYGEN_OUTPUT_DIR "${PROJECT_BINARY_DIR}/docs/doxygen")

set(ZOO_DOXYGEN_INPUTS
    "${PROJECT_SOURCE_DIR}/README.md"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}/generated"
)

set(ZOO_DOXYGEN_EXCLUDES
    "${PROJECT_SOURCE_DIR}/extern"
    "${PROJECT_SOURCE_DIR}/build"
    "${PROJECT_BINARY_DIR}"
)

set(ZOO_DOXYGEN_STRIP_PATHS
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_BINARY_DIR}"
)

function(zoo_format_doxygen_list output_var)
    set(formatted "")
    foreach(item IN LISTS ARGN)
        string(APPEND formatted "\"${item}\" ")
    endforeach()
    string(STRIP "${formatted}" formatted)
    set(${output_var} "${formatted}" PARENT_SCOPE)
endfunction()

zoo_format_doxygen_list(ZOO_DOXYGEN_INPUTS_FORMATTED ${ZOO_DOXYGEN_INPUTS})
zoo_format_doxygen_list(ZOO_DOXYGEN_EXCLUDES_FORMATTED ${ZOO_DOXYGEN_EXCLUDES})
zoo_format_doxygen_list(ZOO_DOXYGEN_STRIP_PATHS_FORMATTED ${ZOO_DOXYGEN_STRIP_PATHS})

if(ZOO_DOXYGEN_DOT_EXECUTABLE)
    set(ZOO_DOXYGEN_HAVE_DOT YES)
    get_filename_component(ZOO_DOXYGEN_DOT_PATH "${ZOO_DOXYGEN_DOT_EXECUTABLE}" DIRECTORY)
else()
    set(ZOO_DOXYGEN_HAVE_DOT NO)
    set(ZOO_DOXYGEN_DOT_PATH "")
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/docs/Doxyfile.in"
    "${PROJECT_BINARY_DIR}/Doxyfile"
    @ONLY
)

add_custom_target(zoo_docs
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ZOO_DOXYGEN_OUTPUT_DIR}"
    COMMAND Doxygen::doxygen "${PROJECT_BINARY_DIR}/Doxyfile"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${PROJECT_SOURCE_DIR}/docs/images"
            "${ZOO_DOXYGEN_OUTPUT_DIR}/html/docs/images"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)

message(STATUS "Doxygen enabled: output will be generated under ${ZOO_DOXYGEN_OUTPUT_DIR}")
if(ZOO_DOXYGEN_DOT_EXECUTABLE)
    message(STATUS "Graphviz dot support enabled for Doxygen diagrams")
else()
    message(STATUS "Graphviz dot not found; Doxygen diagrams are disabled")
endif()
