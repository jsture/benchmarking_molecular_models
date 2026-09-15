# Benchmarking Local PyTorch & HuggingFace Models

This guide describes how to benchmark your own local PyTorch or HuggingFace model on the molecular representation learning benchmarks.

---

## 1. Benchmarking a Local HuggingFace Model

The `huggingface` wrapper supports any model compatible with HuggingFace `transformers` (either a local directory path with model weights/tokenizer or a HuggingFace Hub model identifier).

### Direct CLI Usage

```bash
# Embed a single dataset or all datasets:
uv run python embed.py --model /path/to/your/checkpoint_or_directory --dataset DILI

# Configure batch size, pooling strategy ('mean', 'cls', 'pooler'), and device:
uv run python embed.py --model /path/to/your/checkpoint_or_directory --dataset all --batch-size 64 --pooling mean --device cuda

# Evaluate supervised heads on the generated embeddings:
uv run python score.py --model my_hf_model --dataset all --n-jobs 8
```

---

## 2. Benchmarking a Local PyTorch Model

The `pytorch` wrapper (`model_wrappers/pytorch/wrapper.py`) supports TorchScript models, pickled `torch.nn.Module` files, or custom PyTorch classes.

### Option A: Using a TorchScript or PyTorch Checkpoint

```bash
uv run python embed.py --framework pytorch --model /path/to/your/model.pt --dataset DILI
uv run python score.py --model model --dataset DILI
```

### Option B: Defining a Custom PyTorch Embedder

If your model requires custom featurization (e.g. custom tokenization, SMILES graphs to tensors, etc.), implement your wrapper by subclassing `SmilesEmbedder`:

```python
import torch
import numpy as np
from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device

class MyCustomEmbedder(SmilesEmbedder):
    def __init__(self, model_path: str = "./checkpoints/my_model.pt", device=None):
        self._device = get_device(device)
        self._model = torch.load(model_path).to(self._device)
        self._model.eval()

    def process_batch(self, smiles_list):
        # Featurize SMILES and run forward pass
        ...
        return embeddings_tensor  # (B, D)

    def forward(self, smiles):
        outputs = []
        with torch.no_grad():
            for b in batch(smiles, n=128):
                outputs.append(self.process_batch(b).detach().cpu())
        return torch.cat(outputs, dim=0).numpy()

    @property
    def name(self):
        return "my_custom_model"
```

Implement `get_embedder(name, **kwargs)` in `model_wrappers/pytorch/wrapper.py` to instantiate your class:

```python
def get_embedder(name: str, **kwargs):
    if name == "my_custom_model":
        return MyCustomEmbedder(**kwargs)
    return PyTorchSmilesEmbedder(model_path=name, **kwargs)
```

---

## 3. Workflow Summary

1. **Download Datasets**:
   ```bash
   uv run python download.py --dataset all
   ```
2. **Generate Embeddings**:
   ```bash
   uv run python embed.py --model DeepChem/ChemBERTa-10M-MLM --dataset all
   # Or using the background runner:
   ./run_embed.sh huggingface --model DeepChem/ChemBERTa-10M-MLM --dataset all
   ```
3. **Score Embeddings**:
   ```bash
   uv run python score.py --model ChemBERTa-10M-MLM --dataset all --n-jobs 8
   # Or using the background runner:
   ./run_scoring.sh --model ChemBERTa-10M-MLM --dataset all --n-jobs 8
   ```
4. Results are stored in `data/results/{dataset}/{model}/{head}.yaml` and aggregated in `data/results.csv`.
