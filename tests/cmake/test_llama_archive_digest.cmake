set(case_script "${ZOO_PROJECT_SOURCE_DIR}/tests/cmake/llama_archive_digest_case.cmake")
set(default_tag "b8992")
set(default_sha "942c56b7e7389edfd19150f886794e3f54fe7d51001ebb072c805c5d05016a48")
set(default_base_url "https://github.com/ggerganov/llama.cpp/archive/refs/tags")

function(run_digest_case name should_pass tag sha base_url)
    set(archive_url "${base_url}/${tag}.tar.gz")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DZOO_PROJECT_SOURCE_DIR=${ZOO_PROJECT_SOURCE_DIR}"
            "-DZOO_LLAMA_DEFAULT_TAG=${default_tag}"
            "-DZOO_LLAMA_DEFAULT_SHA256=${default_sha}"
            "-DZOO_LLAMA_DEFAULT_ARCHIVE_BASE_URL=${default_base_url}"
            "-DZOO_LLAMA_TAG=${tag}"
            "-DZOO_LLAMA_SHA256=${sha}"
            "-DZOO_LLAMA_ARCHIVE_BASE_URL=${base_url}"
            "-DZOO_LLAMA_ARCHIVE_URL=${archive_url}"
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

run_digest_case(default_digest TRUE "${default_tag}" "${default_sha}" "${default_base_url}")
run_digest_case(empty_digest FALSE "${default_tag}" "" "${default_base_url}")
run_digest_case(malformed_digest FALSE "${default_tag}" "not-a-sha" "${default_base_url}")
run_digest_case(custom_tag_requires_new_digest FALSE "custom" "${default_sha}" "${default_base_url}")
run_digest_case(custom_tag_accepts_explicit_digest TRUE "custom"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" "${default_base_url}")
