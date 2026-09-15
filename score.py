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
from src.common.results import sync_csv_from_yaml, load_results
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
    cv_verbosity: int = 0,
    seed: int | None = 42,
):
    if not log.getLogger().handlers:
        log.basicConfig(level=log.INFO, format=logging_format)

    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    print(
        f"[{dataset_cfg.name}] [{head}] Starting evaluation for model '{model_name}' "
        f"(task: {dataset_cfg.task}, metric: {dataset_cfg.metric})...",
        flush=True,
    )
    try:
        eval_procedure(
            dataset_info=dataset_cfg,
            embedded_dir=embed_config.embedded_directory,
            predictions_dir=embed_config.predictions_directory,
            model_name=model_name,
            model_head=head,
            override=override,
            results_dir=embed_config.results_directory,
            cv_verbosity=cv_verbosity,
            random_state=seed,
        )
    except Exception as e:
        if safe:
            print(f"[{dataset_cfg.name}] [{head}] ERROR evaluating {model_name}: {e}", flush=True)
            log.error(f"[{dataset_cfg.name}] [{head}] Error evaluating {model_name}: {e}")
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
        help="Re-evaluate and overwrite existing results.",
    )
    parser.add_argument(
        "--cv-verbose",
        type=int,
        default=0,
        help="Verbosity level for scikit-learn cross-validation (default: 0). Use >0 for fold-by-fold logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model heads and cross-validation (default: 42). Use -1 for unseeded.",
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
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory to save per-evaluation YAML result files (default: 'data/results').",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="data/results.csv",
        help="Path to aggregated benchmark results CSV file (default: 'data/results.csv').",
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
        results_directory=args.results_dir,
        results_file=args.results_file,
    )

    resolved_model = resolve_model_name(
        args.model,
        args.embedded_dir,
        [d.name for d in datasets_to_run],
    )

    seed = args.seed if args.seed >= 0 else None
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    tasks = [
        (cfg, head)
        for cfg in datasets_to_run
        for head in args.heads
    ]

    print("\n==================================================", flush=True)
    print(f"Scoring model: '{resolved_model}'", flush=True)
    print(f"Datasets ({len(datasets_to_run)}): {', '.join(d.name for d in datasets_to_run)}", flush=True)
    print(f"Heads ({len(args.heads)}): {', '.join(args.heads)}", flush=True)
    print(f"Seed: {seed if seed is not None else 'None (unseeded)'}", flush=True)
    print(f"Total tasks: {len(tasks)} | Concurrency (n_jobs): {args.n_jobs}", flush=True)
    print("==================================================\n", flush=True)

    if args.n_jobs == 1 or len(tasks) == 1:
        for cfg, head in tasks:
            evaluate_task(cfg, head, resolved_model, embed_config, args.override, args.safe, args.cv_verbose, seed)
    else:
        Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(evaluate_task)(
                cfg, head, resolved_model, embed_config, args.override, args.safe, args.cv_verbose, seed
            )
            for cfg, head in tasks
        )

    # Synchronize all YAML results into the master CSV file
    df = sync_csv_from_yaml(args.results_dir, args.results_file)

    if not df.empty:
        current_model_df = df[df["embedder"] == resolved_model]
        if not current_model_df.empty:
            cols = ["dataset", "head", "cv_metric_name", "cv_metric", "test_metric_name", "test_metric"]
            available_cols = [c for c in cols if c in current_model_df.columns]
            print("\n=== Benchmark Evaluation Summary ===")
            print(current_model_df[available_cols].to_string(index=False))
            print("=====================================\n")

    log.info(f"Scoring complete. Results saved to {args.results_dir} and {args.results_file}")


if __name__ == "__main__":
    main()
