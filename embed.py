#!/usr/bin/env python
"""
Generate embeddings for benchmark datasets using HuggingFace or PyTorch molecular models.

Usage:
    python embed.py --model DeepChem/ChemBERTa-10M-MLM --dataset DILI
    python embed.py --model DeepChem/ChemBERTa-10M-MLM --dataset all
    python embed.py --framework pytorch --model path/to/model.pt --dataset DILI
"""
import argparse
import importlib
import logging as log
import os
import sys
from pathlib import Path
import joblib

from src.common.config import BASE_DIR, EmbeddingConfig
from src.common.data_v2 import download_and_prep
from src.common.datasets import resolve_datasets, list_dataset_names
from src.embedding.embedding import embed, is_already_embedded

logging_format = "%(asctime)s - %(levelname)s - %(message)s"
log.basicConfig(level=log.INFO, format=logging_format)


def resolve_get_embedder(framework: str):
    framework = framework.lower().strip()
    try:
        from wrapper import get_embedder
        return get_embedder
    except ImportError:
        pass

    if framework == "pytorch":
        from model_wrappers.pytorch.wrapper import get_embedder
        return get_embedder
    elif framework == "huggingface":
        from model_wrappers.huggingface.wrapper import get_embedder
        return get_embedder
    else:
        try:
            mod = importlib.import_module(f"model_wrappers.{framework}.wrapper")
            return mod.get_embedder
        except ModuleNotFoundError:
            raise ValueError(
                f"Unknown framework or wrapper '{framework}'. "
                f"Available built-in frameworks: 'huggingface', 'pytorch'."
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate molecular embeddings for benchmark datasets."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model identifier: HuggingFace Hub model ID (e.g. 'DeepChem/ChemBERTa-10M-MLM'), "
             "or path to local model weights/checkpoint/directory.",
    )
    parser.add_argument(
        "--framework", "-f",
        type=str,
        default="huggingface",
        choices=["huggingface", "pytorch"],
        help="Model framework wrapper to use (default: 'huggingface').",
    )
    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        default=["all"],
        help="Dataset name(s) to embed (e.g. 'DILI', 'clf_DILI', 'ogbg-molhiv'), or 'all'. "
             "Accepts multiple arguments or comma-separated names. Default: all.",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--pooling", "-p",
        type=str,
        default=None,
        choices=["mean", "cls", "pooler"],
        help="Token pooling strategy for HuggingFace models: 'mean', 'cls', or 'pooler' (default: model default).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'auto', 'cpu', 'cuda', or 'mps' (default: 'auto').",
    )
    parser.add_argument(
        "--prep-dir",
        type=str,
        default="data/prepared",
        help="Directory where prepared datasets are stored (default: 'data/prepared').",
    )
    parser.add_argument(
        "--embedded-dir",
        type=str,
        default="data/embedded",
        help="Directory to save generated embeddings (default: 'data/embedded').",
    )
    parser.add_argument(
        "--max-invalid",
        type=int,
        default=50,
        help="Maximum allowed failed/invalid embeddings before erroring (default: 50).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputing embeddings even if already cached.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available benchmark datasets and exit.",
    )
    return parser.parse_args()


def main():
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass

    args = parse_args()

    if args.list_datasets:
        print("Available benchmark datasets:")
        for name in list_dataset_names():
            print(f"  - {name}")
        sys.exit(0)

    # Flatten comma-separated values in dataset args
    queries = []
    for item in args.dataset:
        queries.extend([x.strip() for x in item.split(",") if x.strip()])

    try:
        datasets_to_run = resolve_datasets(queries)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    embed_config = EmbeddingConfig(
        prepared_directory=args.prep_dir,
        embedded_directory=args.embedded_dir,
        max_invalid_embeddings=args.max_invalid,
        cache=not args.no_cache,
    )

    get_embedder = resolve_get_embedder(args.framework)

    # Instantiate embedder
    model_kwargs = {"batch_size": args.batch_size, "device": args.device}
    if args.framework == "huggingface" and args.pooling is not None:
        model_kwargs["pooling"] = args.pooling

    log.info(f"Loading embedder for model '{args.model}' (framework: {args.framework})...")
    embedder = get_embedder(args.model, **model_kwargs)
    log.info(f"Loaded model embedder: {embedder.name}")

    # Load illegal SMILES filter if available
    illegal_smiles_file = BASE_DIR / "src/common/illegal_smiles.txt"
    illegal_smiles = []
    if illegal_smiles_file.exists():
        with open(illegal_smiles_file, "r") as f:
            illegal_smiles = [line.strip() for line in f.readlines() if line.strip()]

    prep_base = Path(embed_config.resolve(embed_config.prepared_directory))

    for cfg in datasets_to_run:
        dataset_prep_path = prep_base / f"{cfg.name}.joblib"

        if is_already_embedded(embed_config, cfg.name, embedder) and not args.no_cache:
            log.info(f"Embedding already exists for model '{embedder.name}' on dataset '{cfg.name}', skipping.")
            continue

        # Auto-download/prepare if not found
        if not dataset_prep_path.exists():
            log.info(f"Dataset '{cfg.name}' not found at {dataset_prep_path}, preparing now...")
            download_and_prep(
                dataset_cfg=cfg,
                raw_dir="data/raw",
                prep_dir=embed_config.prepared_directory,
                cache=True,
            )

        log.info(f"Loading prepared dataset: {cfg.name}")
        dataset = joblib.load(str(dataset_prep_path))

        if illegal_smiles:
            dataset.filter_out_problematic_molecules(illegal_smiles)

        log.info(f"Embedding dataset: {cfg.name} ({len(dataset.data)} samples)...")
        embed(embed_config, dataset, embedder, cache=not args.no_cache)

    log.info("Embedding complete for all requested datasets.")


if __name__ == "__main__":
    main()
