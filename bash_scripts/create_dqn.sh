#!/bin/bash

n=3
# run entire script n times
for ((i = 0; i < n; i++))
do
      sed -i -e '/\/\/ FIND-TAG-FILENAME/{n; r /dev/stdin' -e 'd;}' src/MCTS/main.cpp <<EOF
        filename += ".txt$i";
EOF
    fact=1
    for ((k = 1; k <= fact; k++))
    do

        # c = 2^pot
        pot=10
        c=$((2**pot))
        # loop from 2^0 to 2^pot
        for ((j = 0; j <= pot; j++))
        do
            # run connect_four with args j and c/j
            #./train_DQN "batch_size" $((2**j))
            #./train_DQN "c" $((2**j))
            ./train_DQN "games" $((2**j))

	      done
    done
done
