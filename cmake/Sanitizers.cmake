# cmake/Sanitizers.cmake
# Provides zoo_enable_sanitizers(target) to apply ASan + UBSan flags.

function(zoo_enable_sanitizers target)
    if(ZOO_ENABLE_SANITIZERS AND NOT MSVC)
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE
            -fsanitize=address,undefined)
    endif()
endfunction()
