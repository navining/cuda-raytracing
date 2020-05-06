#! /bin/bash

if [ "$#" -ne 0 ]; then
    echo "Usage: ./run.sh"
    exit 0
fi

echo "cleaning up.."
rm -rf build
rm *.gz

echo "running smallpt.."
rai -p ./smallpt &>out.txt
PID="$!"
wait $PID
LINK=$(grep -o '\http.*gz' out.txt)

echo "downloading result.."
curl $LINK --output result.tar.gz
PID="$!"
wait $PID

tar -xf result.tar.gz
PID="$!"
wait $PID

python3 evaluate.py