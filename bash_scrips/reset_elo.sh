dir="data/DQN/c/*"
for file in $dir; do
    # if file is RANKING.*, skip
    if [[ $file == *"RANKING"* ]]; then
        continue
    fi

    sed -i '$s/.*/1000/' $file

done
