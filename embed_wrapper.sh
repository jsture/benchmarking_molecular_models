#!/bin/bash

# Parse options
DELETE=false
CACHE=true
CMD=embed.py
REINSTALL=""
WRAPPER_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--delete)
            DELETE=true
            shift
            ;;
        -r|--reinstall)
            REINSTALL=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        --embed)
            CMD=embed.py
            shift
            ;;
        *)
            WRAPPER_PATH=$1
            shift
            ;;
        -*|--*)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$WRAPPER_PATH" ]; then
    echo "Usage: ./embed_wrapper.sh model_wrappers/<huggingface|pytorch>"
    exit 1
fi

run() {
    # Ensure virtual environment exists using uv
    if [ ! -d ".venv" ] || [ -n "$REINSTALL" ]; then
        echo "Setting up virtual environment with uv..."
        uv venv --python 3.11 .venv
        source .venv/bin/activate
        uv pip install -r requirements.txt
    else
        source .venv/bin/activate
    fi

    # Source wrapper init if available
    if [ -f "$WRAPPER_PATH/init.sh" ]; then
        source "$WRAPPER_PATH/init.sh"
    fi

    if [ -z "$HYDRA_EXPERIMENT" ]; then
        HYDRA_EXPERIMENT=$(basename "$WRAPPER_PATH")
    fi

    export PYTHONPATH=$PYTHONPATH:.:$WRAPPER_PATH

    uv run python -u $CMD --multirun +experiment=$HYDRA_EXPERIMENT ++cache=$CACHE

    echo "Embedding completed."
}

delete() {
    rm -rf .venv
    echo "Cleaned up .venv"
}

if $DELETE; then
    delete
else
    run
fi
