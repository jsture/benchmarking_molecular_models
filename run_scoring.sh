#!/bin/bash
# Usage: ./run_scoring.sh --model <model_name> [extra options]

if [ $# -lt 1 ]; then
    echo "Usage: ./run_scoring.sh --model <model_name> [extra options]"
    exit 1
fi

DATETIME=$(date '+%Y%m%d_%H%M')

mkdir -p logs_scoring

nohup nice -n 10 uv run python score.py "$@" > "logs_scoring/score_${DATETIME}.log" 2>&1 &

echo "Background scoring started with PID $!"
