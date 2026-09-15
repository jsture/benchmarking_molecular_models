import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Repository root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class EmbeddingConfig:
    raw_directory: str = "data/raw"
    embedded_directory: str = "data/embedded"
    prepared_directory: str = "data/prepared"
    predictions_directory: str = "data/predictions"
    results_directory: str = "data/results"
    results_file: str = "data/results.csv"
    illegal_smiles: str = "src/common/illegal_smiles.txt"
    max_invalid_embeddings: int = 50
    data_directory: str = "data/downloaded"
    clock_directory: str = "data/clock"
    svd_directory: str = "data/svd"
    max_samples: Optional[int] = None
    cache: bool = True

    def resolve(self, path: str) -> str:
        """Resolves a relative path against the repository base directory."""
        if os.path.isabs(path):
            return path
        return str(BASE_DIR / path)


DEFAULT_CONFIG = EmbeddingConfig()
