cmake_minimum_required(VERSION 3.16)
project(connect_four)

set(CMAKE_CXX_STANDARD 17)

set(BOOST_ROOT "/home/ossha/boost_1_62_0/")
set(BOOSTROOT "/home/ossha/boost_1_62_0/")
find_package(Boost 1.62.0)
set(Boost_INCLUDE_DIR /home/ossha/boost_1_62_0/)
set(Boost_LIBRARY_DIR /home/ossha/boost_1_62_0/libs/)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funroll-loops")


add_executable(connect_four main.cpp
        game.cpp
        MCTS.cpp
        misc.cpp
        DQL.cpp
        other_bots.cpp)

target_link_libraries(connect_four -lSDL2 -lSDL2_ttf)
