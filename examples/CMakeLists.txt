cmake_minimum_required(VERSION 3.18)

# Demo chat application
add_executable(demo_chat demo_chat.cpp)

target_link_libraries(demo_chat
    PRIVATE
        zoo_backend
)

# Compiler warnings
zoo_set_warnings(demo_chat PRIVATE)

# Install demo
install(TARGETS demo_chat
    RUNTIME DESTINATION bin
)
