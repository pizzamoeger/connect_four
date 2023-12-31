#!/bin/bash

n=5
# run entire script n times
for ((i = 0; i < n; i++))
do
    fact=7
    for ((k = 1; k <= fact; k++))
    do
        sed -i -e '/\/\/ FIND-TAG-FILENAME/{n; r /dev/stdin' -e 'd;}' main.cpp <<EOF
        filename += "_$k.txt$i";
EOF

        make
        # c = 2^pot
        pot=10
        #c=$((2**pot))
        # loop from 2^0 to 2^14
        for ((j = 0; j <= pot; j++))
        do
            # run connect_four with args j and c/j
            # ./connect_four 0 $((2**j)) $((c/(2**j))) 10
            # ./connect_four 1 $((2**j)) $((c/(2**j))) 10
            # ./connect_four 0 7 7 $((2**j)) $((k))
            ./connect_four 1 1 1 $((2**j)) $((k))
	done
    done
done
