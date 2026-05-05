include_guard(GLOBAL)

function(zoo_enable_coverage target)
    target_compile_options(${target} PRIVATE
        "$<$<AND:$<BOOL:${ZOO_ENABLE_COVERAGE}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:--coverage>"
    )
    target_link_options(${target} PRIVATE
        "$<$<AND:$<BOOL:${ZOO_ENABLE_COVERAGE}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:--coverage>"
    )
endfunction()
