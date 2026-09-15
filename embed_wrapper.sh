#!/bin/bash
set -e

# Usage: ./embed_wrapper.sh model_wrappers/<huggingface|pytorch> --model <model_name> [extra options]

WRAPPER_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--delete)
            rm -rf .venv
            echo "Cleaned up .venv"
            exit 0
            ;;
        *)
            if [ -z "$WRAPPER_PATH" ] && [[ "$1" == model_wrappers/* || "$1" == "huggingface" || "$1" == "pytorch" ]]; then
                WRAPPER_PATH=$1
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

FRAMEWORK="huggingface"
if [ -n "$WRAPPER_PATH" ]; then
    FRAMEWORK=$(basename "$WRAPPER_PATH")
fi

uv run python embed.py --framework "$FRAMEWORK" "${EXTRA_ARGS[@]}"
