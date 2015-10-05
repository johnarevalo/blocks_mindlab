#!/bin/bash
if [ "$#" -ne 2  ]; then
    echo "USAGE: run_all.sh [script] [directory]"
    exit 1
fi
for conf in `find $2 -type f  -name "*.yaml"`; do 
    python $1 $conf
done;
