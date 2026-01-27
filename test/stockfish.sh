#!/bin/bash

EXPECTED_ARGS=3
CUTECHESS_PATH=~/Temp/cutechess

if [ "$#" -ne "$EXPECTED_ARGS" ]; then
    echo "Usage: $0  neurofish-tag stockfish-ELO games" >&2
    echo "Example: $0 DNN512 1500 6" >&2
    echo "Error: Expected $EXPECTED_ARGS arguments, but received $#" >&2
    exit 1
fi

CMD_DIR="$(dirname "$0")"
OUTFILE_TEMPLATE="/tmp/fileXXXXXX.pgn"
OUTFILE=$(mktemp --dry-run "$OUTFILE_TEMPLATE")
echo "PGN File: $OUTFILE"
python -c 'from engine import dump_parameters; dump_parameters()'

$CUTECHESS_PATH/build/cutechess-cli \
-engine cmd="$CMD_DIR"/../uci.sh name="$1"  ponder  option.Threads=2   \
-engine cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo="$2" name=stockfish"$2" \
-each proto=uci tc=40/120+1 timemargin=9999 -draw movenumber=40 movecount=5 score=50 \
-resign movecount=3 score=500 twosided=false -maxmoves 100 -recover -games "$3"  -pgnout "$OUTFILE" # -debug

echo "PGN File: $OUTFILE"
