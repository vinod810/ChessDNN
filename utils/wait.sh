#!/bin/bash

# Wait for prepare_data script to finish, then run defunc_build_model.py

while true; do
    count=$(ps -ef | grep prepare_data | grep -v grep | wc -l)
    
    if [ "$count" -le 0 ]; then
        echo "prepare_data script has finished. Running build_model.py..."
        python defunc_build_model.py
        break
    else
        echo "prepare_data still running ($count processes). Waiting..."
        sleep 600
    fi
done
