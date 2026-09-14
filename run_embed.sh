#!/bin/bash
# Usage: ./run_embed.sh <huggingface|pytorch> --model <model_name> [extra options]

if [ $# -lt 2 ]; then
    echo "Usage: ./run_embed.sh <huggingface|pytorch> --model <model_name> [extra options]"
    exit 1
fi

FRAMEWORK=$1
shift
DATETIME=$(date '+%Y%m%d_%H%M')

mkdir -p logs_embed

nohup nice -n 10 uv run python embed.py --framework "$FRAMEWORK" "$@" > "logs_embed/embed_${FRAMEWORK}_${DATETIME}.log" 2>&1 &

echo "Background embedding started with PID $!"
