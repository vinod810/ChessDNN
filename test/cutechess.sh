#!/bin/bash

EXPECTED_ARGS=6

if [ "$#" -ne "$EXPECTED_ARGS" ]; then
    echo "Usage: $0 engine1-cmd engine1-name engine2-cmd engine2-name time-control games" >&2
    echo "Example: test/cutechess.sh ../neurofish_old/engine.sh engine-old engine.sh engine-new 40/120+1 6" >&2
    echo "Error: Expected $EXPECTED_ARGS arguments, but received $#" >&2
    exit 1
fi

~/Temp/cutechess/build/cutechess-cli -engine cmd=$1  name=$2 \
-engine cmd=$3 name=$4  -pgnout /tmp/cutechess.pgn -games $6 \
-each proto=uci tc=$6 timemargin=9999 -draw movenumber=40 movecount=5 score=50 \
-resign movecount=3 score=500 twosided=false -maxmoves 100 -recover # -debug
