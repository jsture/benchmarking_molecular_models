import pandas as pd
import numpy as np

from omegaconf import DictConfig
from os.path import join
from hydra.utils import get_original_cwd
from typing import Tuple, List, Dict, Optional
from .types import Dataset
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

    group = admet_group(path="data/")
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


def load(dataset_config: DictConfig, raw_dir: str, use_hydra: bool = True) -> Dataset:
    if use_hydra:
        raw_dir = join(get_original_cwd(), raw_dir)

    if dataset_config.source.name == 'TDC' and 'benchmark' in dataset_config.source:
        module = get_tdc_group(dataset_config.source.group)
        solver = get_tdc_solver(dataset_config.source.benchmark)
        raw_data, splits = solver(module, dataset_config.name, raw_dir)
    elif dataset_config.source.name == 'TDC':
        raw_data, splits = load_tdc_module_dataset(
            module=get_tdc_group(dataset_config.source.group),
            name=dataset_config.source.collection_name if 'labels' in dataset_config.source else dataset_config.name,
            root=raw_dir,
            label=dataset_config.source.labels if 'labels' in dataset_config.source else None,
        )
    elif dataset_config.source.name == 'OGB':
        raw_data, splits = ogb_solver(dataset_config.name, raw_dir)
    else:
        raise ValueError(f"Unknown dataset source: {dataset_config.source.name}")

    return build_dataset(
        name=dataset_config.name,
        task=dataset_config.task,
        raw_data=raw_data,
        splits=splits
    )