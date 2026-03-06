# cmake/CompilerWarnings.cmake
# Provides zoo_set_warnings(target visibility) to apply platform-aware warning flags.

function(zoo_set_warnings target visibility)
    if(MSVC)
        target_compile_options(${target} ${visibility} /W4)
    else()
        target_compile_options(${target} ${visibility} -Wall -Wextra -Wpedantic)
    endif()
endfunction()
