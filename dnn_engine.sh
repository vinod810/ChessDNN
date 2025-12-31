#!/bin/bash
# Suppress TensorFlow logging
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=-1  # Optional: disable GPU if not needed

cd ~/Documents/Projects/ChessDNN
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ChessDNN
exec python uci_engine.py
