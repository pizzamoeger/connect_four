set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)  # Path to nvcc compiler executable
enable_language(CUDA)
cmake_minimum_required(VERSION 3.16)

project(connect_four)

set(CMAKE_CUDA_ARCHITECTURES 61;70)

#set(CMAKE_CXX_STANDARD 17)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funroll-loops")


# Collect all CUDA files in the specified directory
file(GLOB MCTS_SOURCES "src/MCTS/*.cpp")
file(GLOB NN_SOURCES "src/neuralnetwork/*.cu")
file(GLOB GAME_SOURCES "src/game/*.cpp")
file(GLOB DQN_SOURCES "src/DQN/*.cpp")

# Add executable and link CUDA sources
#cuda_add_executable(YourExecutableName ${CUDA_SOURCES})

add_executable(connect_four ${GAME_SOURCES}
        src/MCTS/MCTS.cpp
        src/DQN/DQN.cpp
        src/misc.cpp
        src/other_bots.cpp
        ${NN_SOURCES})

add_executable(train_MCTS ${MCTS_SOURCES}
        src/DQN/DQN.cpp
        src/other_bots.cpp
        src/misc.cpp
        ${NN_SOURCES})

add_executable(train_DQN_b1024 ${DQN_SOURCES}
        src/MCTS/MCTS.cpp
        src/other_bots.cpp
        src/misc.cpp
        ${NN_SOURCES})

add_executable(fast src/fast_game/main.cpp
        src/MCTS/MCTS.cpp
        src/DQN/DQN.cpp
        src/misc.cpp
        src/other_bots.cpp
        ${NN_SOURCES})

target_link_libraries(connect_four -lSDL2 -lSDL2_ttf)
