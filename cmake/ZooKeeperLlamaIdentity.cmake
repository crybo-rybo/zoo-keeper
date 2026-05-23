include_guard(GLOBAL)

function(zoo_verify_llama_build_identity expected_commit expected_number)
    if(expected_commit STREQUAL "")
        message(FATAL_ERROR
            "ZooKeeperConfig.cmake did not record the llama.cpp build commit it "
            "was compiled against.")
    endif()

    if(NOT DEFINED LLAMA_BUILD_COMMIT OR LLAMA_BUILD_COMMIT STREQUAL "")
        message(FATAL_ERROR
            "Zoo-Keeper was built against llama.cpp '${expected_commit}', but "
            "the located llama package does not expose LLAMA_BUILD_COMMIT.")
    endif()

    if(NOT LLAMA_BUILD_COMMIT STREQUAL expected_commit)
        message(FATAL_ERROR
            "llama.cpp build identity mismatch: Zoo-Keeper was built against "
            "'${expected_commit}', but find_package(llama) resolved "
            "'${LLAMA_BUILD_COMMIT}'.")
    endif()

    if(NOT expected_number STREQUAL "")
        if(NOT DEFINED LLAMA_BUILD_NUMBER OR LLAMA_BUILD_NUMBER STREQUAL "")
            message(FATAL_ERROR
                "Zoo-Keeper was built against llama.cpp build number "
                "'${expected_number}', but the located llama package does not "
                "expose LLAMA_BUILD_NUMBER.")
        endif()

        if(NOT "${LLAMA_BUILD_NUMBER}" STREQUAL "${expected_number}")
            message(FATAL_ERROR
                "llama.cpp build number mismatch: Zoo-Keeper was built against "
                "'${expected_number}', but find_package(llama) resolved "
                "'${LLAMA_BUILD_NUMBER}'.")
        endif()
    endif()
endfunction()
