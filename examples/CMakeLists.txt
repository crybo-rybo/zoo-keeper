cmake_minimum_required(VERSION 3.18)

function(zoo_add_example target source)
    add_executable(${target} ${source})
    target_link_libraries(${target}
        PRIVATE
            zoo
    )
    zoo_set_warnings(${target} PRIVATE)
    install(TARGETS ${target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endfunction()

zoo_add_example(demo_chat demo_chat.cpp)
zoo_add_example(model_generate model_generate.cpp)
zoo_add_example(error_handling error_handling.cpp)
zoo_add_example(stream_cancel stream_cancel.cpp)
zoo_add_example(manual_tool_schema manual_tool_schema.cpp)
