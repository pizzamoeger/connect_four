#!/bin/bash

pot=1
for ((j = 0; j <= pot; j++))
do
    #./train_DQN "batch_size" $((2**j))
    #./train_DQN "c" $((2**j))
    ./train_DQN "games" $((2**j)) <<EOF
games_$((2**j)).txt
EOF
done
