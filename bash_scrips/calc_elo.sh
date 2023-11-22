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
# render screen
sed -i -e '/\/\/ FIND-TAG-RENDER-SCREEN-1/{n; r /dev/stdin' -e 'N;N;N;d;}' game.cpp <<EOF
        /*SDL_RenderClear(renderer);
        render_board();
        set_col(renderer, WHITE);
        SDL_RenderPresent(renderer);*/
EOF
sed -i -e '/\/\/ FIND-TAG-RENDER-SCREEN-2/{n; r /dev/stdin' -e 'N;N;d;}' game.cpp <<EOF
        /*SDL_RenderClear(renderer);
        render_board();
        SDL_RenderPresent(renderer);*/
EOF
# disable animation for selecting col
sed -i -e '/\/\/ FIND-TAG-PICK-COL/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
        board.selected_col = col;
EOF
# disable animation for falling
sed -i -e '/\/\/ FIND-TAG-FALLING/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
        // falling();
EOF
# set dimensions of screen to 0
sed -i -e '/\/\/ FIND-TAG-DIMENSIONS/{n; r /dev/stdin' -e 'N;d;}' game.h <<EOF
const int SCREEN_WIDTH = 0;
const int SCREEN_HEIGHT = 0;
EOF

# compile the game
make

# the new bot
# new_file=$1

# get a list of all files in the directories
directories=("MCTS_plot/games2")
#  "RANDOM" "ALMOST_RANDOM" "DQL"

files=()
for directory in "${directories[@]}"; do
    # add all files that are not RANKING.txt or RANKING.svg
    while IFS= read -r -d $'\0' file; do
        if [[ $file == $directory"/RANKING."* ]]; then
            continue
        fi
        files+=("$file")
    done < <(find "$directory" -type f -print0)
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
      # skip if the file only differs in the last char, skip
      if [[ ${new_file%?} == ${suffled_files[j]%?} ]]; then
        continue
      fi

      echo "Playing ${new_file} against ${suffled_files[j]}"

      # run connect_four, giving file[0] and file[j] as input
      ./connect_four  <<EOF
${new_file}
${files[j]}
EOF

      # ./bash_scrips/ranking.sh
      # nohup python3 plot.py

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
sed -i -e '/\/\/ FIND-TAG-RENDER-SCREEN-1/{n; r /dev/stdin' -e 'N;N;N;d;}' game.cpp <<EOF
    SDL_RenderClear(renderer);
    render_board();
    set_col(renderer, WHITE);
    SDL_RenderPresent(renderer);
EOF
sed -i -e '/\/\/ FIND-TAG-RENDER-SCREEN-2/{n; r /dev/stdin' -e 'N;N;d;}' game.cpp <<EOF
            SDL_RenderClear(renderer);
            render_board();
            SDL_RenderPresent(renderer);
EOF
sed -i -e '/\/\/ FIND-TAG-PICK-COL/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
    // board.selected_col = col;
EOF
sed -i -e '/\/\/ FIND-TAG-FALLING/{n; r /dev/stdin' -e 'd;}' game.cpp <<EOF
        falling();
EOF
sed -i -e '/\/\/ FIND-TAG-DIMENSIONS/{n; r /dev/stdin' -e 'N;d;}' game.h <<EOF
const int SCREEN_WIDTH = 1000;
const int SCREEN_HEIGHT = 700+TEXT_DIST+TEXT_SIZE+150+100;
EOF
