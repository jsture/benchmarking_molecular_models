import joblib
import os
import sys
import importlib
import hydra
import logging as log

from os.path import join
from hydra.utils import get_original_cwd
from src.common.types import EmbeddingConfig, Dataset
from src.embedding.embedding import embed, is_already_embedded


def resolve_get_embedder(cfg, model_name: str):
    try:
        from wrapper import get_embedder
        return get_embedder
    except ImportError:
        pass

    # Check if an experiment is specified or infer from model_name
    wrapper_type = "huggingface"
    if hasattr(cfg, "experiment") and cfg.experiment:
        wrapper_type = str(cfg.experiment).lower()
    elif "pytorch" in str(model_name).lower():
        wrapper_type = "pytorch"

    try:
        mod = importlib.import_module(f"model_wrappers.{wrapper_type}.wrapper")
        return mod.get_embedder
    except ModuleNotFoundError:
        mod = importlib.import_module("model_wrappers.huggingface.wrapper")
        return mod.get_embedder


@hydra.main(config_path="./config", config_name="embed")
def main(cfg):
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        pass

    embed_config = EmbeddingConfig(**cfg.embedding)
    if "model_name" in cfg:
        model_name = cfg.model_name
    else:
        model_name = cfg.model.model_name

    kwargs = cfg.model.kwargs if "model" in cfg and "kwargs" in cfg.model else {}

    get_embedder = resolve_get_embedder(cfg, model_name)
    model = get_embedder(model_name, task=cfg.dataset.task, **kwargs)
    log.info(f"Embedding model: {model.name}")
    dataset_name = cfg.dataset.name

    if is_already_embedded(embed_config, dataset_name, model) and cfg.cache:
        log.info(f"Embedding already exists for {model.name} on {dataset_name}")
        return

    dataset_path = os.path.join(get_original_cwd(), embed_config.prepared_directory, f"{dataset_name}.joblib")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at {dataset_path}. "
            f"Please run 'uv run python download.py' first."
        )
    dataset = joblib.load(dataset_path)

    illegal_smiles_path = cfg.illegal_smiles if 'illegal_smiles' in cfg else None
    if illegal_smiles_path is not None:
        full_illegal_path = join(get_original_cwd(), illegal_smiles_path)
        if os.path.exists(full_illegal_path):
            with open(full_illegal_path, 'r') as f:
                illegal_smiles = [line.strip() for line in f.readlines()]
            log.info(f"Filtering out illegal SMILES: {len(illegal_smiles)} molecules")
            dataset.filter_out_problematic_molecules(illegal_smiles)

    embed(embed_config, dataset, model, cache=cfg.cache)


if __name__ == '__main__':
    main()
