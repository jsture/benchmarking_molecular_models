import os
from pathlib import Path
import pandas as pd
import numpy as np

from typing import Tuple, List, Dict, Optional
from .types import Dataset
from .datasets import DatasetConfig
from .config import BASE_DIR
from rdkit import Chem

from tdc.benchmark_group import admet_group
from tdc.single_pred import ADME, Tox, HTS

# Tuple of train, val, test indices
Splits = Dict[str, List[int]]


def ogb_solver(name: str, root: str) -> Tuple[pd.DataFrame, Splits]:
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name=name, root=root)
    smiles = pd.read_csv(f"{root}/{name.replace('-', '_')}/mapping/mol.csv.gz").drop(columns=["mol_id"])
    return smiles, dataset.get_idx_split()


def load_tdc_module_dataset(module, name: str, root: str, label: Optional[str] = None) -> Tuple[pd.DataFrame, Splits]:
    kwargs = {'label_name': label} if label is not None else {}
    data = module(name=name, path=root, **kwargs)
    splits = data.get_split(method='scaffold')
    splits['train']['split'] = 'train'
    splits['valid']['split'] = 'train'
    splits['test']['split'] = 'test'

    dataset = pd.concat([splits['train'], splits['valid'], splits['test']]).reset_index(drop=True)

    splits = {
        'train': dataset[dataset['split'] == 'train'].index.tolist(),
        'valid': [],
        'test': dataset[dataset['split'] == 'test'].index.tolist()
    }

    return dataset.rename(columns={"Drug": "smiles"}).drop(
        ["Drug_ID"], axis=1, errors="ignore"
    ), splits


def tdc_admet_solver(module, name: str, root: str) -> Tuple[pd.DataFrame, Splits]:
    data = module(name=name, path=root)
    split = data.get_split()

    train, valid, test = split["train"], split["valid"], split["test"]
    dataset = pd.concat([train, valid, test]).reset_index(drop=True)

    cache_path = os.path.join(root, "tdc_benchmark")
    group = admet_group(path=cache_path)
    benchmark = group.get(name)

    return (
        dataset.rename(columns={"Drug": "smiles"}).drop(
            ["Drug_ID"], axis=1, errors="ignore"
        ),
        {
            "train": list(
                benchmark["train_val"].merge(
                    dataset.reset_index().groupby(["Drug_ID", "Drug"]).first(),
                    on=["Drug_ID", "Drug"],
                )["index"]
            ),
            "valid": [],
            "test": list(
                benchmark["test"].merge(
                    dataset.reset_index().groupby(["Drug_ID", "Drug"]).first(),
                    on=["Drug_ID", "Drug"],
                )["index"]
            ),
        },
    )


def get_tdc_group(group_name: str):
    return {
        'ADME': ADME,
        'TOX': Tox,
        'HTS': HTS,
    }[group_name]


def get_tdc_solver(benchmark: str):
    return {
        'admet': tdc_admet_solver,
    }[benchmark]


def build_dataset(name: str, task: str, raw_data: pd.DataFrame, splits: Splits) -> Dataset:
    raw_data['smiles'] = raw_data['smiles'].map(Chem.CanonSmiles)
    return Dataset(
        name=name,
        data=raw_data,
        splits=splits,
        task=task
    )


def load(dataset_config: DatasetConfig, raw_dir: str = "data/raw") -> Dataset:
    abs_raw_dir = str(Path(raw_dir) if Path(raw_dir).is_absolute() else BASE_DIR / raw_dir)

    if dataset_config.source_name == 'TDC' and dataset_config.source_benchmark is not None:
        module = get_tdc_group(dataset_config.source_group)
        solver = get_tdc_solver(dataset_config.source_benchmark)
        raw_data, splits = solver(module, dataset_config.name, abs_raw_dir)
    elif dataset_config.source_name == 'TDC':
        module = get_tdc_group(dataset_config.source_group)
        collection = dataset_config.collection_name if dataset_config.collection_name else dataset_config.name
        raw_data, splits = load_tdc_module_dataset(
            module=module,
            name=collection,
            root=abs_raw_dir,
            label=dataset_config.labels,
        )
    elif dataset_config.source_name == 'OGB':
        raw_data, splits = ogb_solver(dataset_config.name, abs_raw_dir)
    else:
        raise ValueError(f"Unknown dataset source: {dataset_config.source_name}")

    return build_dataset(
        name=dataset_config.name,
        task=dataset_config.task,
        raw_data=raw_data,
        splits=splits
    )


def download_and_prep(
    dataset_cfg: DatasetConfig,
    raw_dir: str = "data/raw",
    prep_dir: str = "data/prepared",
    cache: bool = True,
) -> Path:
    import joblib
    prep_path = Path(prep_dir) if Path(prep_dir).is_absolute() else BASE_DIR / prep_dir
    prep_path.mkdir(parents=True, exist_ok=True)
    out_file = prep_path / f"{dataset_cfg.name}.joblib"

    if cache and out_file.exists():
        return out_file

    dataset = load(dataset_cfg, raw_dir=raw_dir)
    joblib.dump(dataset, str(out_file))
    return out_file