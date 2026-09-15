"""
Storage and retrieval of benchmark evaluation results using YAML and CSV files.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import logging as log
import os
import yaml
import pandas as pd

import numpy as np

from .config import BASE_DIR


def to_serializable(val: Any) -> Any:
    """Recursively converts numpy and non-standard types to standard Python types for YAML/JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, (np.bool_,)):
        return bool(val)
    elif isinstance(val, np.ndarray):
        return [to_serializable(x) for x in val.tolist()]
    elif isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_serializable(x) for x in val]
    return val


@dataclass
class ResultRecord:
    dataset: str
    task: str
    embedder: str
    head: str
    cv_metric_name: str
    cv_metric: float
    test_metric_name: str
    test_metric: float
    hyperparams: Dict[str, Any]
    library_hash: str
    timestamp: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        self.hyperparams = to_serializable(self.hyperparams)

    def to_csv_dict(self) -> Dict[str, Any]:
        """Flattens record for tabular CSV storage."""
        return {
            "dataset": self.dataset,
            "task": self.task,
            "embedder": self.embedder,
            "head": self.head,
            "cv_metric_name": self.cv_metric_name,
            "cv_metric": self.cv_metric,
            "test_metric_name": self.test_metric_name,
            "test_metric": self.test_metric,
            "hyperparams": json.dumps(self.hyperparams, sort_keys=True),
            "library_hash": self.library_hash,
            "timestamp": self.timestamp,
            "seed": self.seed,
        }


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return BASE_DIR / p
    return p


def get_yaml_path(results_dir: str | Path, dataset: str, embedder: str, head: str) -> Path:
    dir_path = resolve_path(results_dir)
    return dir_path / dataset / embedder / f"{head}.yaml"


def is_already_evaluated(results_dir: str | Path, dataset: str, embedder: str, head: str) -> bool:
    """Checks whether an evaluation result YAML already exists and is non-empty."""
    yaml_path = get_yaml_path(results_dir, dataset, embedder, head)
    return yaml_path.exists() and yaml_path.stat().st_size > 0


def delete_result(results_dir: str | Path, dataset: str, embedder: str, head: str, results_csv: Optional[str | Path] = None) -> None:
    """Deletes the result YAML file and removes its row from results.csv if present."""
    yaml_path = get_yaml_path(results_dir, dataset, embedder, head)
    if yaml_path.exists():
        try:
            yaml_path.unlink()
            log.info(f"Deleted previous evaluation file: {yaml_path}")
        except OSError as e:
            log.warning(f"Could not delete {yaml_path}: {e}")

    if results_csv is not None:
        csv_path = resolve_path(results_csv)
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                match = (df["dataset"] == dataset) & (df["embedder"] == embedder) & (df["head"] == head)
                if match.any():
                    df = df[~match]
                    df.to_csv(csv_path, index=False)
            except Exception as e:
                log.warning(f"Could not remove record from {csv_path}: {e}")


def save_result_yaml(record: ResultRecord, results_dir: str | Path) -> Path:
    """Saves an evaluation record as a human-readable YAML file atomically."""
    yaml_path = get_yaml_path(results_dir, record.dataset, record.embedder, record.head)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(record)
    content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    with open(yaml_path, "w") as f:
        f.write(content)

    return yaml_path



def sync_csv_from_yaml(results_dir: str | Path, results_csv: str | Path) -> pd.DataFrame:
    """
    Scans all result YAML files in results_dir and synchronizes results.csv.
    This guarantees zero write contention during parallel scoring runs.
    """
    dir_path = resolve_path(results_dir)
    csv_path = resolve_path(results_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    if dir_path.exists():
        for yaml_file in dir_path.glob("**/*.yaml"):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "dataset" in data and "embedder" in data and "head" in data:
                    record = ResultRecord(**data)
                    records.append(record.to_csv_dict())
            except Exception as e:
                log.warning(f"Error reading {yaml_file}: {e}")

    if records:
        df = pd.DataFrame(records)
        df.sort_values(by=["dataset", "embedder", "head"], inplace=True)
    else:
        df = pd.DataFrame(columns=[
            "dataset", "task", "embedder", "head",
            "cv_metric_name", "cv_metric", "test_metric_name", "test_metric",
            "hyperparams", "library_hash", "timestamp", "seed"
        ])

    df.to_csv(csv_path, index=False)
    log.info(f"Synchronized {len(df)} evaluation result(s) to {csv_path}")
    return df


def load_results(results_csv: str | Path) -> pd.DataFrame:
    """Loads all benchmark results from CSV."""
    csv_path = resolve_path(results_csv)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)
