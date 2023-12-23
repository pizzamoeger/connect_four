#!/bin/bash

DQL="DQL/test.txt"
MCTS="MCTS/20_20_2.txt"
MCTS="ALMOST_RANDOM"

# Initialize a variable to hold the sum
total=0

echo "$7 $1 $2 $3 $4 $5 $6"
./$7 $1 $2 $3 $4 $5 $6

# Number of times you want to run the program
num_runs=10

# Loop to run the program multiple times and accumulate the output
for ((i = 0; i < num_runs; i++))
do
    # Run the C++ program using heredoc and capture its output
    result=$(./fast <<EOF
${DQL}
${MCTS}
1
EOF
    )
    ((total += result))

    result=$(./fast <<EOF
${MCTS}
${DQL}
-1
EOF
    )

    # Accumulate the output to the total sum
    ((total += result))
done

# Print the total sum
echo "$total"