#!/bin/bash

EXPECTED_ARGS=3

if [ "$#" -ne "$EXPECTED_ARGS" ]; then
    echo "Usage: $0 engine1-name engine2-name games" >&2
    echo "Error: Expected $EXPECTED_ARGS arguments, but received $#" >&2
    exit 1
fi

~/Temp/cutechess/build/cutechess-cli -engine cmd=~/Documents/Projects/neurofish_old/dnn_engine.sh  name=$1 -engine cmd=~/Documents/Projects/ChessDNN/dnn_engine.sh name=$2  -pgnout /tmp/cutechess.pgn -games $3 -each proto=uci tc=40/180+2 timemargin=9999 -draw movenumber=40 movecount=3 score=100 -resign movecount=3 score=500 twosided=false -maxmoves 75 -recover
