#!/bin/bash

# loop from 2^0 to 2^pot
for ((j = 0; j <= pot; j++))
do
    #./train_DQN "batch_size" $((2**j))
    #./train_DQN "c" $((2**j))
    ./train_DQN "games" $((2**j))
done
