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
zoo_set_warnings(demo_chat PRIVATE)

# Install demo
install(TARGETS demo_chat
    RUNTIME DESTINATION bin
)
