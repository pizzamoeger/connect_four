cmake_minimum_required(VERSION 3.22)
project(connect_four)

set(CMAKE_CXX_STANDARD 17)

add_executable(connect_four main.cpp
        game.cpp)

target_link_libraries(connect_four -lSDL2 -lSDL2_ttf)
