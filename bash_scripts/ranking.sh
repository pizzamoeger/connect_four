#!/bin/bash

# Define the directories you want to search in
directories=("data/MCTS" "data/DQN" "data/HUMAN" "data/ALMOST_RANDOM" "data/RANDOM")

# Initialize an empty array to store last lines
last_lines=()

# Loop through each directory
for directory in "${directories[@]}"; do

    # Use a subshell to capture the output of the inner loop
    # loop through each file in the directory
    while IFS= read -r -d $'\0' file; do
	if [[ $file == $directory"RANKING."* ]]; then
            continue
        fi

        # Get the last line of the file
        last_line=$(tail -n 1 "$file")

        # Append the last line and filename to the array
        last_lines+=("$last_line:$file")
    done < <(find "$directory" -type f -print0)
done

# Sort the array by the first field (ELO rating)
IFS=$'\n' sorted=($(sort -nr -t':' -k1 <<<"${last_lines[*]}"))
unset IFS

i=1
> "RANKING.txt"
for line in "${sorted[@]}"; do
    # split the line into ELO and filename
    IFS=":" read -ra line_array <<< "$line"
    filename="${line_array[1]}"
    ELO="${line_array[0]}"

    echo "$i. $filename: $ELO" >> "RANKING.txt"
    ((i++))
done
