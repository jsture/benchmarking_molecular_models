import os
import joblib
import json
import torch
import logging as log
from pathlib import Path
from typing import Any

from .const import DEFAULT_MEMORY_WEIGHT
from .train import fit_and_eval_embedding
from .eval_metrics import evaluate
from .utils import get_model_version_hash
from ...common.config import BASE_DIR
from ...common.types import EvaluationResult, EmbeddedDataset
from ...common.results import ResultRecord, is_already_evaluated, delete_result, save_result_yaml


def eval_embedding(
    data: EmbeddedDataset,
    pred_directory: str,
    dataset_config: Any,
    metric_name: str,
    model_head: str,
    cv_verbosity: int = 0,
) -> EvaluationResult:
    log.info(f"[{data.name}] [{model_head}] Training head model (metric: {metric_name})...")
    mem_wt = getattr(dataset_config, 'memory_weight', None)
    if mem_wt is None and isinstance(dataset_config, dict):
        mem_wt = dataset_config.get('memory_weight', DEFAULT_MEMORY_WEIGHT)
    elif mem_wt is None:
        mem_wt = DEFAULT_MEMORY_WEIGHT

    head_result = fit_and_eval_embedding(
        dataset=data, 
        metric_name=metric_name, 
        model_head=model_head,
        memory_weight=mem_wt,
        cv_verbosity=cv_verbosity,
    )
    log.info(f"[{data.name}] [{model_head}] CV training complete, best CV {metric_name}: {head_result.cv_score:.4f}")
    return evaluate(head_result, dataset_config, pred_directory)


def eval_procedure(
    dataset_info: Any,
    embedded_dir: str,
    predictions_dir: str,
    model_name: str,
    model_head: str,
    override: bool = False,
    results_dir: str = "data/results",
    cv_verbosity: int = 0,
) -> ResultRecord | None:
    model_version_hash = get_model_version_hash()
    dataset_name = dataset_info.name if hasattr(dataset_info, "name") else str(dataset_info)
    metric = dataset_info.metric if hasattr(dataset_info, "metric") else "roc_auc"

    if is_already_evaluated(results_dir, dataset_name, model_name, model_head):
        if not override:
            print(f"[{dataset_name}] [{model_head}] Already evaluated for '{model_name}', skipping.", flush=True)
            log.info(f"[{dataset_name}] [{model_head}] Already evaluated for '{model_name}', skipping.")
            return None
        print(f"[{dataset_name}] [{model_head}] Overriding previous evaluation for '{model_name}'.", flush=True)
        log.warning(f"[{dataset_name}] [{model_head}] Overriding previous evaluation for '{model_name}'.")
        delete_result(results_dir, dataset_name, model_name, model_head)
        
    if model_head == 'knn' and 'muv' in dataset_name:
        print(f"[{dataset_name}] [{model_head}] Skipping KNN evaluation for MUV (not supported).", flush=True)
        log.warning(f"[{dataset_name}] [{model_head}] Skipping KNN evaluation for MUV (not supported).")
        return None

    emb_path = Path(embedded_dir)
    if not emb_path.is_absolute():
        emb_path = BASE_DIR / emb_path

    embedded_filename = str(emb_path / dataset_name / f"{model_name}.joblib")
    legacy_filename = str(emb_path / dataset_name / f"{model_name}.json")
    
    if os.path.exists(legacy_filename):
        log.info(f"[{dataset_name}] Legacy embedded dataset found, converting to new format")
        embedded_data = EmbeddedDataset.deserialize_legacy(legacy_filename)
    elif not os.path.exists(embedded_filename):
        print(f"[{dataset_name}] [{model_head}] ERROR: Embedded dataset not found at {embedded_filename}", flush=True)
        log.error(f"[{dataset_name}] Embedded dataset not found: {embedded_filename}")
        return None
    else:
        embedded_data: EmbeddedDataset = joblib.load(embedded_filename)

    if embedded_data.X is None:
        log.error(f"[{dataset_name}] Embedded dataset is empty")
        raise RuntimeError(f"[{dataset_name}] Embedded dataset is empty")
    
    if isinstance(embedded_data.X, torch.Tensor):
        log.info(f"[{dataset_name}] Converting torch.Tensor to numpy array")
        embedded_data.X = embedded_data.X.detach().cpu().numpy()
    
    if len(embedded_data.X.shape) == 1:
        log.warning(f"[{dataset_name}] Invalid X shape (1 dim), assuming invalid concatenation")
        desired_samples = embedded_data.y.shape[0]
        embedded_data.X = embedded_data.X.reshape(desired_samples, -1)
    log.info(f"[{dataset_name}] [{model_head}] Shape {embedded_data.X.shape} for dataset {embedded_data.name}, task {embedded_data.task}")

    print(
        f"[{dataset_name}] [{model_head}] Training head on {embedded_data.X.shape[0]} samples "
        f"(features: {embedded_data.X.shape[1]}, task: {embedded_data.task}, metric: {metric})...",
        flush=True,
    )

    result = eval_embedding(
        embedded_data,
        predictions_dir,
        dataset_info,
        metric,
        model_head,
        cv_verbosity=cv_verbosity,
    )
    log.info(f"[{dataset_name}] [{model_head}] Evaluation complete, test result: {result.metric_value}")
    
    record = ResultRecord(
        dataset=embedded_data.name,
        task=embedded_data.task,
        embedder=embedded_data.embedder,
        head=result.model,
        cv_metric_name=metric,
        cv_metric=float(result.cv_metric_value),
        test_metric_name=result.metric_name,
        test_metric=float(result.metric_value),
        hyperparams=result.hyperparams,
        library_hash=model_version_hash,
    )
    save_result_yaml(record, results_dir)

    print(
        f"[{dataset_name}] [{model_head}] Finished: "
        f"CV {metric} = {record.cv_metric:.4f} | "
        f"Test {record.test_metric_name} = {record.test_metric:.4f}",
        flush=True,
    )
    return record