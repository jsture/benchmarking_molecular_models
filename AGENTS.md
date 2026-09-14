# Repository Guidelines

## Project Structure & Module Organization

The benchmark follows download → embed → score, with root entry points `download.py`, `embed.py`, and `score.py`. Shared dataset types, serialization, and database utilities live in `src/common/`; embedding orchestration lives in `src/embedding/`; supervised evaluation lives in `src/eval/`.

Hydra configurations are organized under `config/dataset/`, `config/model/`, `config/experiment/`, and `config/embedding/`. Model implementations live in `model_wrappers/` (`huggingface` and `pytorch`). `docs/custom_model.md` explains custom model extensions.

## Build, Test, and Development Commands

Use Python 3.11 (supported range: 3.10–3.11) and `uv` for dependency and environment management. Run commands from the repository root:

- `uv venv --python 3.11 .venv && source .venv/bin/activate`: create and activate the benchmark virtual environment.
- `bash install_deps.sh` or `uv pip install -r requirements.txt`: install benchmark requirements.
- `uv run python download.py`: download configured datasets.
- `uv run python embed.py +experiment=huggingface +model=huggingface_chemberta10m_mlm`: generate embeddings for datasets.
- `bash embed_wrapper.sh model_wrappers/huggingface`: wrapper script for embedding.
- `uv run python score.py --multirun +experiment=huggingface`: train and evaluate configured supervised heads.

Check Hydra sweeper settings before running: `config/embed.yaml` selects multiple datasets, while `config/score.yaml` configures scoring sweeps. Concurrency is controlled via Hydra joblib settings.

## Coding Style & Naming Conventions

Follow four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Use two-space YAML indentation.

New wrappers implement `SmilesEmbedder`, expose `get_embedder()`, and return an `(N, D)` NumPy array. Preserve input order and represent failed samples with NaN rows. Keep model names unique and consistent with configuration.

## Testing Guidelines

Validate changes with a small configured dataset through download, embedding, and scoring (e.g. `clf_DILI` or `clf_ogbg-molhiv`). Check embedding dimensions, sample alignment, failed-sample handling, and results stored in `data/meta.db`.

## Commit & Pull Request Guidelines

Keep commits concise and descriptive (e.g. `Simplify repo for PyTorch/HuggingFace`). Keep credentials, virtual environments, downloaded datasets, and large model weights out of commits.
