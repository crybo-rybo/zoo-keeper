# Standalone cmake -P script: removes all .gcda files under BUILD_DIR.
# Called by the 'crap' target before running tests to avoid stale coverage data.
if(NOT DEFINED BUILD_DIR)
    message(FATAL_ERROR "zoo_clear_gcda.cmake: BUILD_DIR is not set")
endif()
file(GLOB_RECURSE _gcda_files "${BUILD_DIR}/*.gcda")
foreach(_f IN LISTS _gcda_files)
    file(REMOVE "${_f}")
endforeach()
