#!/bin/bash
set -e

echo "Setting up virtual environment with uv (Python 3.11)..."
uv venv --python 3.11 .venv
source .venv/bin/activate

echo "Installing project dependencies with uv..."
uv pip install -r requirements.txt

echo "Setup complete! Activate the environment with:"
echo "  source .venv/bin/activate"
