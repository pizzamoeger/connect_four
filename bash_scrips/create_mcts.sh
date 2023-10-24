n=5
# run entire script n times
for ((i = 0; i < n; i++))
do
    sed -i -e '/\/\/ FIND-TAG-FILENAME/{n; r /dev/stdin' -e 'd;}' main.cpp <<EOF
        filename += ".txt$i";
EOF
    make

    # c = 2^pot
    pot=10
    c=$((2**pot))
    # loop from 2^0 to 2^14
    for ((j = 0; j <= pot; j++))
    do
        # run connect_four with args j and c/j
        ./connect_four 0 $((2**j)) $((c/(2**j))) 10
        ./connect_four 1 $((2**j)) $((c/(2**j))) 10
    done
done
