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

Create the virtual environment and install all dependencies:

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
# Download and prepare all datasets:
uv run python download.py --dataset all

# Or download specific dataset(s):
uv run python download.py --dataset DILI CYP2C9_Veith

# List all 26 available datasets:
uv run python download.py --list
```

### 2. Generate Embeddings

#### HuggingFace Model (Hub or Local Directory)
To embed datasets with a HuggingFace model (e.g. ChemBERTa, MoLFormer, ChemGPT, or your local directory):

```sh
# Using a HuggingFace Hub model:
uv run python embed.py --model DeepChem/ChemBERTa-10M-MLM --dataset DILI

# Using a local model directory:
uv run python embed.py --model /path/to/my_hf_model --dataset all

# Specify batch size, pooling (mean/cls/pooler), or device:
uv run python embed.py --model DeepChem/ChemBERTa-10M-MLM --dataset all --batch-size 64 --pooling mean --device cuda
```

#### PyTorch Model (Local Checkpoint / TorchScript)
To embed datasets using a local PyTorch model (`.pt`, `.pth`, TorchScript, or pickled `nn.Module`):

```sh
uv run python embed.py --framework pytorch --model /path/to/my_pytorch_model.pt --dataset DILI
```

Or run using the wrapper script:

```sh
./embed_wrapper.sh model_wrappers/huggingface --model DeepChem/ChemBERTa-10M-MLM --dataset DILI
# or in the background:
./run_embed.sh huggingface --model DeepChem/ChemBERTa-10M-MLM --dataset all
```

### 3. Evaluate and Score Embeddings

Train and evaluate supervised learning heads (`rf`, `ridge`, `knn`) across benchmark datasets:

```sh
# Evaluate on a single dataset:
uv run python score.py --model ChemBERTa-10M-MLM --dataset DILI

# Evaluate across all datasets with 8 parallel worker jobs:
uv run python score.py --model ChemBERTa-10M-MLM --dataset all --n-jobs 8

# Select specific heads (e.g. Ridge and Random Forest only):
uv run python score.py --model ChemBERTa-10M-MLM --dataset all --heads ridge rf
```

To run scoring in the background:

```sh
./run_scoring.sh --model ChemBERTa-10M-MLM --dataset all --n-jobs 8
```

All evaluation metrics and best-performing hyperparameter heads are automatically saved to human-readable YAML files at `data/results/{dataset}/{model}/{head}.yaml` and aggregated in `data/results.csv` for easy analysis with pandas or spreadsheets.

---

## Adding Your Own Model

See [docs/custom_model.md](docs/custom_model.md) for detailed instructions on configuring and benchmarking custom PyTorch architectures or HuggingFace transformers.

---

## Repository Structure

```
├── model_wrappers/      # Model integration wrappers
│   ├── huggingface/     # General HuggingFace & transformer models
│   └── pytorch/         # General local PyTorch models
├── src/
│   ├── common/          # Dataset definitions, types, serialization, and storage
│   │   ├── datasets.py  # Typed Python dataset registry (26 TDC ADMET & OGB benchmarks)
│   │   ├── config.py    # Path configuration and EmbeddingConfig dataclass
│   │   ├── types.py     # SmilesEmbedder, Dataset, EmbeddedDataset abstractions
│   │   ├── data_v2.py   # Dataset downloading, preprocessing, and SMILES canonicalization
│   │   └── results.py   # YAML & CSV benchmark result storage and synchronization
│   ├── embedding/       # Embedding generation orchestration
│   └── eval/            # Supervised evaluation heads & metrics
├── download.py          # CLI entrypoint to download and prepare datasets
├── embed.py             # CLI entrypoint to generate embeddings
├── score.py             # CLI entrypoint to train heads and evaluate scores
├── pyproject.toml       # uv / Python project configuration
└── requirements.txt     # Clean dependency specification
```
