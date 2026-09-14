# Molecular Model Benchmarking

A streamlined framework for evaluating and benchmarking local **PyTorch** and **HuggingFace** pretrained molecular models on TDC ADMET and MoleculeNet (OGB) benchmarks.

---

## Prerequisites

- **Python**: `>=3.10, <3.12`
- **uv**: Fast Python package installer and dependency manager ([uv installation guide](https://docs.astral.sh/uv/))

```sh
# On macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or via Homebrew on macOS
brew install uv
```

## Installation

Create the virtual environment and install all dependencies in seconds:

```sh
./install_deps.sh
```

Or manually:

```sh
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Quickstart

### 1. Download Benchmark Datasets
Download TDC ADMET and OGB MoleculeNet benchmark datasets:

```sh
uv run python download.py
```

### 2. Generate Embeddings

#### HuggingFace Model (Local or Hub)
To embed datasets with a HuggingFace model (e.g. ChemBERTa, MoLFormer, or your local directory):

```sh
# Using a local model directory or HuggingFace Hub model:
uv run python embed.py +experiment=huggingface +model=huggingface_chemberta10m_mlm

# Or point to a local model checkpoint directory:
uv run python embed.py +experiment=huggingface model_name=/path/to/my_hf_model
```

#### PyTorch Model (Local Checkpoint)
To embed datasets using a local PyTorch model checkpoint (`.pt`, TorchScript, or pickled `nn.Module`):

```sh
uv run python embed.py +experiment=pytorch model_name=/path/to/my_pytorch_model.pt
```

Or run using the wrapper script:

```sh
./embed_wrapper.sh model_wrappers/huggingface
# or in the background:
./run_embed.sh huggingface
```

### 3. Evaluate and Score Embeddings

Train and evaluate supervised learning heads (Ridge, Logistic Regression, Random Forest, KNN) across benchmark datasets:

```sh
uv run python score.py --multirun +experiment=huggingface
# or for your custom model:
uv run python score.py +experiment=huggingface model_name=my_model_name
```

To run scoring in the background:

```sh
./run_scoring.sh huggingface
```

All evaluation metrics and best-performing hyperparameter heads are automatically saved to SQLite at `data/meta.db`.

---

## Adding Your Own Model

See [docs/custom_model.md](docs/custom_model.md) for detailed instructions on configuring and benchmarking custom PyTorch architectures or HuggingFace transformers.

---

## Repository Structure

```
├── config/              # Hydra configurations
│   ├── dataset/         # TDC & OGB benchmark dataset definitions
│   ├── experiment/      # Experiment sweeps (huggingface, pytorch)
│   ├── model/           # Model-specific parameters
│   ├── embed.yaml       # Embedding pipeline settings
│   └── score.yaml       # Scoring pipeline settings
├── model_wrappers/      # Model integration wrappers
│   ├── huggingface/     # General HuggingFace & transformer models
│   └── pytorch/         # General local PyTorch models
├── src/
│   ├── common/          # Dataset types, serialization, database schemas
│   ├── embedding/       # Embedding generation orchestration
│   └── eval/            # Supervised evaluation heads & metrics
├── download.py          # Entrypoint to download and prepare datasets
├── embed.py             # Entrypoint to generate embeddings
├── score.py             # Entrypoint to train heads and evaluate scores
├── pyproject.toml       # uv / Python project configuration
└── requirements.txt     # Clean dependency specification
```
