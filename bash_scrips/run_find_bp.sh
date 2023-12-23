n=10  # Set the number of times to run the command
n="$1"

for ((i=0; i<n; i++))
do
  nohup echo "params/params$i.txt" | python3 src/DQL/bp.py > params/params$i.out &
done