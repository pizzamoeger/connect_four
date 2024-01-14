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

        make train_MCTS
        # c = 2^pot
        pot=10
        c=$((2**pot))
        # loop from 2^0 to 2^14
        for ((j = 0; j <= pot; j++))
        do
            # run connect_four with args j and c/j
            #./train_MCTS 0 $((2**j)) $((c/(2**j))) 1
            #./train_MCTS 1 $((2**j)) $((c/(2**j))) 1
            #./train_MCTS 1 8 $((2**j)) 1
            #./train_MCTS 0 8 $((2**j)) 1
            ./train_MCTS 0 8 8 $((2**j))
            ./train_MCTS 1 8 8 $((2**j))

	done
    done
done
