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
run_identity_case(mismatched_commit FALSE "b8992" "8992" "tampered" "8992")
