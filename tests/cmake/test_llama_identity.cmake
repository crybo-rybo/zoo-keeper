set(case_script "${ZOO_PROJECT_SOURCE_DIR}/tests/cmake/llama_identity_case.cmake")

function(run_identity_case name should_pass expected_commit expected_number actual_commit actual_number)
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DZOO_PROJECT_SOURCE_DIR=${ZOO_PROJECT_SOURCE_DIR}"
            "-DEXPECTED_COMMIT=${expected_commit}"
            "-DEXPECTED_NUMBER=${expected_number}"
            "-DLLAMA_BUILD_COMMIT=${actual_commit}"
            "-DLLAMA_BUILD_NUMBER=${actual_number}"
            -P "${case_script}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error)

    if(should_pass AND NOT result EQUAL 0)
        message(FATAL_ERROR "${name} should pass:\n${output}\n${error}")
    endif()
    if(NOT should_pass AND result EQUAL 0)
        message(FATAL_ERROR "${name} should fail")
    endif()
endfunction()

run_identity_case(exact_identity TRUE "b8992" "8992" "b8992" "8992")
run_identity_case(missing_commit FALSE "b8992" "8992" "" "8992")
run_identity_case(mismatched_commit FALSE "b8992" "8992" "tampered" "8992")
run_identity_case(mismatched_number FALSE "b8992" "8992" "b8992" "1")

set(tmp_dir "${ZOO_PROJECT_BINARY_DIR}/tests/llama_identity_mismatch")
file(REMOVE_RECURSE "${tmp_dir}")
file(MAKE_DIRECTORY
    "${tmp_dir}/consumer"
    "${tmp_dir}/zoo/lib/cmake/ZooKeeper"
    "${tmp_dir}/llama/lib/cmake/llama")

file(COPY
    "${ZOO_PROJECT_BINARY_DIR}/cmake/ZooKeeperConfig.cmake"
    "${ZOO_PROJECT_SOURCE_DIR}/cmake/ZooKeeperLlamaIdentity.cmake"
    DESTINATION "${tmp_dir}/zoo/lib/cmake/ZooKeeper")
file(WRITE "${tmp_dir}/llama/lib/cmake/llama/llama-config.cmake"
    "set(LLAMA_BUILD_COMMIT tampered)\n"
    "set(LLAMA_BUILD_NUMBER 1)\n")
file(WRITE "${tmp_dir}/consumer/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.18)\n"
    "project(zoo_identity_mismatch LANGUAGES CXX)\n"
    "find_package(ZooKeeper CONFIG REQUIRED)\n")

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${tmp_dir}/consumer"
        -B "${tmp_dir}/build"
        "-DCMAKE_PREFIX_PATH=${tmp_dir}/zoo;${tmp_dir}/llama"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)

if(result EQUAL 0)
    message(FATAL_ERROR "install-tree mismatch probe should fail")
endif()

set(combined_output "${output}\n${error}")
if(NOT combined_output MATCHES "llama.cpp build identity mismatch")
    message(FATAL_ERROR
        "install-tree mismatch probe failed for the wrong reason:\n"
        "${combined_output}")
endif()
