#!/bin/bash

directories=("$1")

files=()
for directory in "${directories[@]}"; do
    # add all files that are not RANKING.txt or RANKING.svg
    while IFS= read -r -d $'\0' file; do
        if [[ $file == $directory"RANKING."* ]]; then
            continue
        fi
        files+=("$file")
    done < <(find "$directory" -type f -print0)
done
files=( $(shuf -e "${files[@]}") )

n=$(( ${#files[@]} * 1 ))
for ((i = 0; i < n; i++))
do
    # shuffle the files
    suffled_files=( $(shuf -e "${files[@]}") )

    echo "$i"

    # the file at pos i mod the number of files is the file playing against all other files
    new_file=${files[i%${#files[@]}]}

    for ((j = 0; j < ${#suffled_files[@]}; j++))
    do
      # if the file only differs in the last char, skip
      if [[ ${new_file%?} == ${suffled_files[j]%?} ]]; then
        continue
      fi

      echo "Playing ${new_file} against ${suffled_files[j]}"
      prefix="data/"
      new_file_new=${new_file#$prefix}
      shuf_file_new=${suffled_files[j]#$prefix}

      ./fast  <<EOF
${new_file_new}
${shuf_file_new}
EOF
      ./fast  <<EOF
${shuf_file_new}
${new_file_new}
EOF

    done
done
