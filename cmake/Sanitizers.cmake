include_guard(GLOBAL)

function(zoo_enable_sanitizers target)
    set(_san "AND:$<BOOL:${ZOO_ENABLE_SANITIZERS}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>")
    target_compile_options(${target} PRIVATE
        "$<$<${_san}>:-fsanitize=address,undefined>"
        "$<$<${_san}>:-fno-omit-frame-pointer>"
    )
    target_link_options(${target} PUBLIC
        "$<$<${_san}>:-fsanitize=address,undefined>"
    )
endfunction()
