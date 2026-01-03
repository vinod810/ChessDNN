#!/bin/bash

EXPECTED_ARGS=4

if [ "$#" -ne "$EXPECTED_ARGS" ]; then
    echo "Usage: $0 oldfish-name neurofish-name time-control games" >&2
    echo "Example: $0 test/cutechess.sh name-old name-neuro 40/120+1 6" >&2
    echo "Error: Expected $EXPECTED_ARGS arguments, but received $#" >&2
    exit 1
fi

dir="$(dirname "$0")"

~/Temp/cutechess/build/cutechess-cli -engine cmd="$dir"/../../oldfish/uci_engine.sh name="$1" \
-engine cmd="$dir"/../uci_engine.sh name="$2"  -pgnout /tmp/cutechess.pgn \
-each proto=uci tc=$3 timemargin=9999 -draw movenumber=40 movecount=5 score=50 \
-resign movecount=3 score=500 twosided=false -maxmoves 100 -recover -games $4 # -debug
