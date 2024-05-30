nodes=`awk '{print $1}' ~/hostfile`
for i in $nodes; do 
    mkdir $i
done

while :
do
	for i in $nodes; do 
		scp -r $i:/dev/shm/stragglers/* ./$i/ > /dev/null 2>&1
	done
    clear
    python stragglers.py
    sleep 10
done
