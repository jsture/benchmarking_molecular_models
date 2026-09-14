#!/usr/bin/env python
"""
Train and evaluate supervised heads on generated molecular embeddings.

Usage:
    python score.py --model ChemBERTa-10M-MLM --dataset DILI
    python score.py --model ChemBERTa-10M-MLM --dataset all --heads rf ridge knn
    python score.py --model ChemBERTa-10M-MLM --dataset all --n-jobs 8
"""
import argparse
import logging as log
import sys
import traceback
from pathlib import Path
from joblib import Parallel, delayed

from src.common.config import BASE_DIR, EmbeddingConfig
from src.common.datasets import resolve_datasets, list_dataset_names
from src.common.db import DbContex, init_db
from src.eval import eval_procedure, AVAILABLE_HEADS

logging_format = "%(asctime)s - %(levelname)s - %(message)s"
log.basicConfig(level=log.INFO, format=logging_format)


def resolve_model_name(model_arg: str, embedded_dir: str, dataset_names: list[str]) -> str:
    candidates = [
        model_arg,
        model_arg.split("/")[-1],
        model_arg.split("/")[-1].split(".")[0],
    ]
    emb_path = Path(embedded_dir)
    if not emb_path.is_absolute():
        emb_path = BASE_DIR / emb_path

    for ds in dataset_names:
        for cand in candidates:
            if (emb_path / ds / f"{cand}.joblib").exists():
                return cand

    return candidates[1]


def evaluate_task(
    dataset_cfg,
    head: str,
    model_name: str,
    embed_config: EmbeddingConfig,
    override: bool,
    safe: bool,
):
    log.info(f"Evaluating {model_name} on {dataset_cfg.name} (task: {dataset_cfg.task}, metric: {dataset_cfg.metric}) with head: {head}")
    try:
        with DbContex(embed_config):
            eval_procedure(
                dataset_info=dataset_cfg,
                embedded_dir=embed_config.embedded_directory,
                predictions_dir=embed_config.predictions_directory,
                model_name=model_name,
                model_head=head,
                override=override,
            )
    except Exception as e:
        if safe:
            log.error(f"Error evaluating {model_name} on {dataset_cfg.name} with head {head}: {e}")
            log.error(traceback.format_exc())
        else:
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate supervised heads on molecular embeddings."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name (e.g. 'ChemBERTa-10M-MLM' or 'DeepChem/ChemBERTa-10M-MLM').",
    )
    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        default=["all"],
        help="Dataset name(s) to score (e.g. 'DILI', 'clf_DILI', 'ogbg-molhiv'), or 'all'. "
             "Accepts multiple arguments or comma-separated names. Default: all.",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        default=AVAILABLE_HEADS,
        choices=AVAILABLE_HEADS,
        help=f"Supervised head(s) to train. Available: {AVAILABLE_HEADS} (default: all).",
    )
    parser.add_argument(
        "--n-jobs", "-j",
        type=int,
        default=4,
        help="Number of parallel evaluation workers across datasets and heads (default: 4). "
             "Use 1 for sequential execution.",
    )
    parser.add_argument(
        "--override", "--no-cache",
        action="store_true",
        dest="override",
        help="Re-evaluate and overwrite existing results in database.",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        default=True,
        help="Run in safe mode: continue evaluating remaining tasks if one fails (default: True).",
    )
    parser.add_argument(
        "--no-safe",
        action="store_false",
        dest="safe",
        help="Fail immediately on the first evaluation error.",
    )
    parser.add_argument(
        "--embedded-dir",
        type=str,
        default="data/embedded",
        help="Directory where embeddings are stored (default: 'data/embedded').",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="data/predictions",
        help="Directory to save test predictions (default: 'data/predictions').",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="data/meta.db",
        help="Path to SQLite database (default: 'data/meta.db').",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available benchmark datasets and exit.",
    )
    return parser.parse_args()


def main():
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
        embedded_directory=args.embedded_dir,
        predictions_directory=args.predictions_dir,
        database=args.database,
    )
    init_db(embed_config)

    resolved_model = resolve_model_name(
        args.model,
        args.embedded_dir,
        [d.name for d in datasets_to_run],
    )
    log.info(f"Target model for scoring: '{resolved_model}' (input: '{args.model}')")

    tasks = [
        (cfg, head)
        for cfg in datasets_to_run
        for head in args.heads
    ]
    log.info(f"Queued {len(tasks)} scoring task(s) across {len(datasets_to_run)} dataset(s) and {len(args.heads)} head(s).")

    if args.n_jobs == 1 or len(tasks) == 1:
        for cfg, head in tasks:
            evaluate_task(cfg, head, resolved_model, embed_config, args.override, args.safe)
    else:
        Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(evaluate_task)(
                cfg, head, resolved_model, embed_config, args.override, args.safe
            )
            for cfg, head in tasks
        )

    log.info(f"Scoring complete. Results saved to {args.database}")


if __name__ == "__main__":
    main()
