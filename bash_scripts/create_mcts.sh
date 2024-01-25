#!/bin/bash

# c = 2^pot
pot=10
c=$((2**pot))
for ((j = 0; j <= pot; j++))
do
    #./train_MCTS 0 $((2**j)) $((c/(2**j))) 1
    #./train_MCTS 1 $((2**j)) $((c/(2**j))) 1
    #./train_MCTS 1 8 $((2**j)) 1
    #./train_MCTS 0 8 $((2**j)) 1
    ./train_MCTS 0 8 8 $((2**j))
    ./train_MCTS 1 8 8 $((2**j))
done
