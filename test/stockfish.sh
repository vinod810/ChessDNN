#!/bin/bash

EXPECTED_ARGS=2
CUTECHESS_PATH=~/Temp/cutechess

if [ "$#" -ne "$EXPECTED_ARGS" ]; then
    echo "Usage: $0 stockfish-ELO  games" >&2
    echo "Example: $0 1500 6" >&2
    echo "Error: Expected $EXPECTED_ARGS arguments, but received $#" >&2
    exit 1
fi

CMD_DIR="$(dirname "$0")"
OUTFILE_TEMPLATE="/tmp/fileXXXXXX.pgn"
OUTFILE=$(mktemp --dry-run "$OUTFILE_TEMPLATE")
echo "PGN File: $OUTFILE"

$CUTECHESS_PATH/build/cutechess-cli \
-engine cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo="$1" name=stockfish \
-engine cmd="$CMD_DIR"/../uci_engine.sh name=neurofish  -pgnout "$OUTFILE" -games "$2" \
-each proto=uci tc=40/120+1 timemargin=9999 -draw movenumber=40 movecount=5 score=50 \
-resign movecount=3 score=500 twosided=false -maxmoves 100 -recover # -debug

echo "PGN File: $OUTFILE"
