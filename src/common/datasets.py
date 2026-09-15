from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class DatasetConfig:
    name: str
    task: str           # 'classification' | 'regression'
    metric: str         # 'roc_auc' | 'pr_auc_score' | 'mae' | 'spearmancorr'
    source_name: str    # 'TDC' | 'OGB'
    source_group: Optional[str] = None       # 'ADME', 'TOX', 'HTS'
    source_benchmark: Optional[str] = None   # 'admet'
    collection_name: Optional[str] = None
    labels: Optional[str] = None
    memory_weight: int = 1


# Complete benchmark registry of molecular datasets
_DATASET_LIST: List[DatasetConfig] = [
    # TDC ADMET Benchmark Group
    DatasetConfig(name="AMES", task="classification", metric="roc_auc", source_name="TDC", source_group="TOX", source_benchmark="admet"),
    DatasetConfig(name="Bioavailability_Ma", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="CYP2C9_Substrate_CarbonMangels", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="CYP2C9_Veith", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="CYP2D6_Substrate_CarbonMangels", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="CYP2D6_Veith", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="CYP3A4_Substrate_CarbonMangels", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="CYP3A4_Veith", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="DILI", task="classification", metric="roc_auc", source_name="TDC", source_group="TOX", source_benchmark="admet"),
    DatasetConfig(name="HIA_Hou", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="Pgp_Broccatelli", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME", source_benchmark="admet"),
    DatasetConfig(name="hERG", task="classification", metric="roc_auc", source_name="TDC", source_group="TOX", source_benchmark="admet"),

    # TDC Single Datasets
    DatasetConfig(name="CYP1A2_Veith", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME"),
    DatasetConfig(name="CYP2C19_Veith", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME"),
    DatasetConfig(name="PAMPA_NCATS", task="classification", metric="roc_auc", source_name="TDC", source_group="ADME"),
    DatasetConfig(name="SARSCoV2_3CLPro_Diamond", task="classification", metric="roc_auc", source_name="TDC", source_group="HTS"),
    DatasetConfig(name="SARSCoV2_Vitro_Touret", task="classification", metric="roc_auc", source_name="TDC", source_group="HTS"),
    DatasetConfig(name="hERG_Karim", task="classification", metric="roc_auc", source_name="TDC", source_group="TOX"),

    # OGB / MoleculeNet Datasets
    DatasetConfig(name="ogbg-molbace", task="classification", metric="roc_auc", source_name="OGB"),
    DatasetConfig(name="ogbg-molbbbp", task="classification", metric="roc_auc", source_name="OGB"),
    DatasetConfig(name="ogbg-molclintox", task="classification", metric="roc_auc", source_name="OGB"),
    DatasetConfig(name="ogbg-molhiv", task="classification", metric="roc_auc", source_name="OGB"),
    DatasetConfig(name="ogbg-molmuv", task="classification", metric="roc_auc", source_name="OGB", memory_weight=16),
    DatasetConfig(name="ogbg-molsider", task="classification", metric="roc_auc", source_name="OGB"),
    DatasetConfig(name="ogbg-moltox21", task="classification", metric="roc_auc", source_name="OGB", memory_weight=4),
    DatasetConfig(name="ogbg-moltoxcast", task="classification", metric="roc_auc", source_name="OGB", memory_weight=16),
]

DATASETS: Dict[str, DatasetConfig] = {d.name: d for d in _DATASET_LIST}

# Support aliases with clf_ prefix
for d in _DATASET_LIST:
    DATASETS[f"clf_{d.name}"] = d


def get_dataset_config(name: str) -> DatasetConfig:
    """Returns the DatasetConfig for a given dataset name or alias."""
    if name in DATASETS:
        return DATASETS[name]
    # Check case-insensitive
    for k, v in DATASETS.items():
        if k.lower() == name.lower():
            return v
    raise KeyError(f"Dataset '{name}' not found. Available datasets: {list_dataset_names()}")


def list_dataset_names(include_aliases: bool = False) -> List[str]:
    """Lists standard dataset names."""
    if include_aliases:
        return list(DATASETS.keys())
    return [d.name for d in _DATASET_LIST]


def resolve_datasets(requested: Optional[List[str]]) -> List[DatasetConfig]:
    """
    Resolves a list of requested dataset names/patterns.
    If None, empty, or ['all'], returns all benchmark datasets.
    """
    if not requested or "all" in [r.lower() for r in requested]:
        return list(_DATASET_LIST)

    configs = []
    seen = set()
    for item in requested:
        # Support comma-separated strings like "DILI,hERG"
        for sub in item.split(","):
            sub = sub.strip()
            if not sub:
                continue
            cfg = get_dataset_config(sub)
            if cfg.name not in seen:
                seen.add(cfg.name)
                configs.append(cfg)
    return configs
