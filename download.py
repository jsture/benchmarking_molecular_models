#!/usr/bin/env python
"""
Download and prepare molecular benchmark datasets.

Usage:
    python download.py --dataset DILI
    python download.py --dataset DILI,CYP2C9_Veith
    python download.py --dataset all
    python download.py --list
"""
import argparse
import logging as log
import sys
from src.common.data_v2 import download_and_prep
from src.common.datasets import resolve_datasets, list_dataset_names

logging_format = "%(asctime)s - %(levelname)s - %(message)s"
log.basicConfig(level=log.INFO, format=logging_format)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare molecular benchmark datasets (TDC ADMET and OGB MoleculeNet)."
    )
    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        default=["all"],
        help="Dataset name(s) to download (e.g. 'DILI', 'clf_DILI', 'ogbg-molhiv'), or 'all'. "
             "Accepts multiple arguments or comma-separated names. Default: all.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory to cache raw downloaded files (default: 'data/raw').",
    )
    parser.add_argument(
        "--prep-dir",
        type=str,
        default="data/prepared",
        help="Directory to save prepared .joblib datasets (default: 'data/prepared').",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching and force re-download/re-prep.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available benchmark datasets and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
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

    log.info(f"Downloading/preparing {len(datasets_to_run)} dataset(s)...")
    success_count = 0
    for cfg in datasets_to_run:
        log.info(f"Processing dataset: {cfg.name}")
        try:
            out_file = download_and_prep(
                dataset_cfg=cfg,
                raw_dir=args.raw_dir,
                prep_dir=args.prep_dir,
                cache=not args.no_cache,
            )
            log.info(f"Dataset {cfg.name} ready at: {out_file}")
            success_count += 1
        except Exception as e:
            log.error(f"Failed to process dataset {cfg.name}: {e}")
            raise

    log.info(f"Successfully prepared {success_count}/{len(datasets_to_run)} datasets.")


if __name__ == "__main__":
    main()
