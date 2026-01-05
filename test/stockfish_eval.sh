#!/bin/bash

SF_ENGINE="stockfish"
DEPTH=1

echo "Stockfish Position Evaluator (depth $DEPTH)"
echo "Enter FEN positions to evaluate, or 'quit'/'exit' to end."
echo ""

while true; do
    read -p "FEN> " FEN

    # Check for exit conditions
    case "${FEN,,}" in
        quit|exit)
            echo "Goodbye!"
            exit 0
            ;;
        "")
            echo "Please enter a FEN or 'quit' to exit."
            continue
            ;;
    esac

    echo "Evaluating at depth $DEPTH..."

    # Use a temp file to capture output
    TMPFILE=$(mktemp)

    # Run stockfish and capture output - use process substitution to avoid subshell issue
    {
        echo "uci"
        echo "isready"
        echo "position fen $FEN"
        echo "go depth $DEPTH"
        # Give stockfish time to process
        sleep 2
        echo "quit"
    } | "$SF_ENGINE" 2>/dev/null > "$TMPFILE"

    SF_OUTPUT=$(cat "$TMPFILE")
    rm -f "$TMPFILE"

    LAST_INFO_LINE=$(echo "$SF_OUTPUT" | grep "^info " | grep " score " | tail -n 1)
    SCORE_TYPE=$(echo "$LAST_INFO_LINE" | grep -oP 'score \K(cp|mate)')
    SCORE_VALUE=$(echo "$LAST_INFO_LINE" | grep -oP "score $SCORE_TYPE \K-?\d+")
    BEST_MOVE=$(echo "$SF_OUTPUT" | grep "^bestmove" | awk '{print $2}')

    if [ -n "$SCORE_VALUE" ]; then
        echo "Evaluation ($SCORE_TYPE): $SCORE_VALUE"
        echo "Best Move: $BEST_MOVE"
    else
        echo "Could not find a valid evaluation. Check your FEN."
        echo "Debug output:"
        echo "$SF_OUTPUT"
    fi
    echo ""
done