cmake_minimum_required(VERSION 3.18)

# Demo chat application
add_executable(demo_chat demo_chat.cpp)

target_link_libraries(demo_chat
    PRIVATE
        zoo
        zoo_backend
)

target_include_directories(demo_chat
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Compiler warnings
if(MSVC)
    target_compile_options(demo_chat PRIVATE /W4)
else()
    target_compile_options(demo_chat PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install demo
install(TARGETS demo_chat
    RUNTIME DESTINATION bin
)
