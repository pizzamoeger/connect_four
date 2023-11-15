dir="MCTS_plot/games1/*"
for file in $dir; do
    # if file is RANKING.*, skip
    if [[ $file == *"RANKING"* ]]; then
        continue
    fi

    sed -i '$s/.*/1000/' $file

done
