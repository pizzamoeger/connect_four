set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)  # Path to nvcc compiler executable
cmake_minimum_required(VERSION 3.16)
enable_language(CUDA)

project(connect_four)

set(CMAKE_CUDA_ARCHITECTURES 61;70)

#set(CMAKE_CXX_STANDARD 17)

set(BOOST_ROOT "/home/hannah/boost_1_62_0/")
set(BOOSTROOT "/home/hannah/boost_1_62_0/")
find_package(Boost 1.62.0)
set(Boost_INCLUDE_DIR /home/hannah/boost_1_62_0/)
set(Boost_LIBRARY_DIR /home/hannah/boost_1_62_0/libs/)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funroll-loops")


# Collect all CUDA files in the specified directory
file(GLOB MCTS_SOURCES "src/MCTS/*.cpp")
file(GLOB NN_SOURCES "src/neuralnetwork/*.cu")
file(GLOB GAME_SOURCES "src/game/*.cpp")

# Add executable and link CUDA sources
#cuda_add_executable(YourExecutableName ${CUDA_SOURCES})

add_executable(connect_four ${GAME_SOURCES}
        src/MCTS/MCTS.cpp
        src/misc.cpp
        src/DQL/DQL.cpp
        ${NN_SOURCES}
        src/other_bots.cpp)

add_executable(train_MCTS ${MCTS_SOURCES}
        src/DQL/DQL.cpp
        src/other_bots.cpp
        ${NN_SOURCES})

target_link_libraries(connect_four -lSDL2 -lSDL2_ttf)
