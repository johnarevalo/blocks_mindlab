#!/bin/bash
dir=`mktemp -d`
export THEANO_FLAGS=$THEANO_FLAGS,base_compiledir=$dir
${@:1}
#python2.7 "${@:2}"
