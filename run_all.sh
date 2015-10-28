#!/bin/bash
if [ "$#" -ne 4  ]; then
    echo "USAGE: run_all.sh [script] [directory] [n_jobs] [logdir]"
    exit 1
fi
JOB_BASH=$( dirname $( readlink -f $0 ) )/job.sh
SCRIPT="$JOB_BASH $1"
FOLDER=$2
JOBS=$3
LOGDIR=$4
parallel --results $LOGDIR -j $JOBS $SCRIPT ::: `find $FOLDER -type f  -name "*.yaml" | sort` &
