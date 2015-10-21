#!/bin/bash
if [ "$#" -ne 3  ]; then
    echo "USAGE: run_all.sh [script] [directory] [n_jobs]"
    exit 1
fi
SCRIPT=$1
FOLDER=$2
JOBS=$3
parallel -j $JOBS $SCRIPT ::: `find $FOLDER -type f  -name "*.yaml"`
