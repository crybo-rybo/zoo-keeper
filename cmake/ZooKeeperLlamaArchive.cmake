include_guard(GLOBAL)

function(zoo_validate_llama_archive_digest)
    if(NOT DEFINED ZOO_LLAMA_SHA256 OR ZOO_LLAMA_SHA256 STREQUAL "")
        message(FATAL_ERROR
            "ZOO_LLAMA_SHA256 must be set to the expected SHA-256 digest for "
            "${ZOO_LLAMA_ARCHIVE_URL}.")
    endif()

    string(LENGTH "${ZOO_LLAMA_SHA256}" ZOO_LLAMA_SHA256_LENGTH)
    if(NOT ZOO_LLAMA_SHA256_LENGTH EQUAL 64 OR ZOO_LLAMA_SHA256 MATCHES "[^0-9a-fA-F]")
        message(FATAL_ERROR
            "ZOO_LLAMA_SHA256 must be a 64-character hexadecimal SHA-256 digest.")
    endif()

    if(DEFINED ZOO_LLAMA_DEFAULT_TAG
            AND DEFINED ZOO_LLAMA_DEFAULT_SHA256
            AND DEFINED ZOO_LLAMA_DEFAULT_ARCHIVE_BASE_URL
            AND (NOT ZOO_LLAMA_TAG STREQUAL ZOO_LLAMA_DEFAULT_TAG
                 OR NOT ZOO_LLAMA_ARCHIVE_BASE_URL STREQUAL
                    ZOO_LLAMA_DEFAULT_ARCHIVE_BASE_URL)
            AND ZOO_LLAMA_SHA256 STREQUAL ZOO_LLAMA_DEFAULT_SHA256)
        message(FATAL_ERROR
            "ZOO_LLAMA_SHA256 must be updated when overriding ZOO_LLAMA_TAG or "
            "ZOO_LLAMA_ARCHIVE_BASE_URL. Set it to the SHA-256 digest for "
            "${ZOO_LLAMA_ARCHIVE_URL}.")
    endif()
endfunction()

function(zoo_llama_archive_hash_arg output_var)
    zoo_validate_llama_archive_digest()
    set(${output_var} "SHA256=${ZOO_LLAMA_SHA256}" PARENT_SCOPE)
endfunction()
