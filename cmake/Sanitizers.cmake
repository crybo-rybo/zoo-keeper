include_guard(GLOBAL)

function(zoo_enable_sanitizers target)
    target_compile_options(${target} PRIVATE
        "$<$<AND:$<BOOL:${ZOO_ENABLE_SANITIZERS}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=address,undefined>"
        "$<$<AND:$<BOOL:${ZOO_ENABLE_SANITIZERS}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fno-omit-frame-pointer>"
    )
    target_link_options(${target} PRIVATE
        "$<$<AND:$<BOOL:${ZOO_ENABLE_SANITIZERS}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=address,undefined>"
    )
endfunction()
