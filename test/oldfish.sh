#!/bin/bash

EXPECTED_ARGS=4

if [ "$#" -ne "$EXPECTED_ARGS" ]; then
    echo "Usage: $0 neurofish-name oldfish-name time-control games" >&2
    echo "Example: $0 neuro512  old1024 40/120+1 6" >&2
    echo "Error: Expected $EXPECTED_ARGS arguments, but received $#" >&2
    exit 1
fi

CMD_DIR="$(dirname "$0")"
OUTFILE_TEMPLATE="/tmp/fileXXXXXX.pgn"
OUTFILE=$(mktemp --dry-run "$OUTFILE_TEMPLATE")
echo "PGN File: $OUTFILE"
#python -c 'from engine import dump_parameters; dump_parameters()'

~/Temp/cutechess/build/cutechess-cli \
-engine cmd="$CMD_DIR"/../uci.sh name="$1"  \
-engine cmd="$CMD_DIR"/../../oldfish/uci.sh name="$2" \
-each proto=uci tc="$3" timemargin=9999 -draw movenumber=40 movecount=5 score=50 \
-resign movecount=3 score=500 twosided=false -maxmoves 100 -recover -games "$4"  -pgnout "$OUTFILE" #-debug

echo "PGN File: $OUTFILE"

