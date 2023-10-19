#!/bin/bash

# text input via cin
sed -i -e '/\/\/ FIND-TAG-TEXT-INPUT-START/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
    /* SDL_Event event;
EOF
sed -i -e '/\/\/ FIND-TAG-TEXT-INPUT-STOP/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
    */ std::string text; std::cin >> text;
EOF
# menu selection hardcoded
sed -i -e '/\/\/ FIND-TAG-MENU-SELECTION/{n; r /dev/stdin' -e 'N;N;N;d;}' game.cpp <<EOF
    selected = {1, 1};
    playerfile_1 = get_text();
    playerfile_2 = get_text();
    return mode();
EOF
# end screen
sed -i -e '/\/\/ FIND-TAG-END/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
    return 0;
EOF
# play two games
sed -i -e '/\/\/ FIND-TAG-COUNTER/{n; r /dev/stdin' -e 'd;}' main.cpp <<EOF
    while (true && counter < 2) {
EOF

# compile the game
make

# the new bot
# new_file=$1

# get a list of all files in the directories
directories=("MCTS/simulations")
#  "RANDOM" "ALMOST_RANDOM" "DQN"
files=()
for directory in "${directories[@]}"; do
    files+=($(find "$directory" -type f))
done
files=( $(shuf -e "${files[@]}") )

n=$(( ${#files[@]} * 1 )) # every file plays equal number of games

for ((i = 0; i < n; i++))
do
    # shuffle the files
    suffled_files=( $(shuf -e "${files[@]}") )

    echo "$i"

    # the file at pos i mod the number of files is the file playing against all other files
    new_file=${files[i%${#files[@]}]}

    for ((j = 0; j < ${#suffled_files[@]}; j++))
    do
      # skip if the file is the same
      if [[ $new_file == ${files[j]} ]]; then
        continue
      fi
      # run connect_four, giving file[0] and file[j] as input
      ./connect_four  <<EOF
${new_file}
${files[j]}
EOF

    done
done

# set back to normal
sed -i -e '/\/\/ FIND-TAG-TEXT-INPUT-START/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
    SDL_Event event;
EOF
sed -i -e '/\/\/ FIND-TAG-TEXT-INPUT-STOP/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
    // std::string text; cin >> text;

EOF
sed -i -e '/\/\/ FIND-TAG-MENU-SELECTION/{n; r /dev/stdin' -e 'N;N;N;d;}' game.cpp <<EOF
    /* selected = {1, 1};
    playerfile_1 = get_text();
    playerfile_2 = get_text();
    return mode();*/
EOF
sed -i -e '/\/\/ FIND-TAG-END/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
    return CONTINUE;
EOF
sed -i -e '/\/\/ FIND-TAG-COUNTER/{n; r /dev/stdin' -e 'd;}' main.cpp <<EOF
    while (true) {
EOF