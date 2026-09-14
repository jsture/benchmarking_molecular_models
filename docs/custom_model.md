# Benchmarking Local PyTorch & HuggingFace Models

This guide describes how to benchmark your own local PyTorch or HuggingFace model on the molecular representation learning benchmarks.

---

## 1. Benchmarking a Local HuggingFace Model

The `huggingface` wrapper supports any model compatible with HuggingFace `transformers` (either a local directory path with model weights/tokenizer or a HuggingFace Hub model identifier).

### Quickstart with Config

Create a config file in `config/model/my_hf_model.yaml`:

```yaml
model_name: /path/to/your/checkpoint_or_directory
kwargs:
  pooling: mean   # 'mean' (default), 'cls', or 'pooler'
  batch_size: 128
  max_length: 512
```

Then run embedding and scoring:

```bash
# Generate embeddings on benchmark datasets
uv run python embed.py +experiment=huggingface +model=my_hf_model

# Evaluate supervised heads on the generated embeddings
uv run python score.py +experiment=huggingface model_name=my_hf_model
```

---

## 2. Benchmarking a Local PyTorch Model

The `pytorch` wrapper (`model_wrappers/pytorch/wrapper.py`) supports TorchScript models, pickled `torch.nn.Module` files, or custom PyTorch classes.

### Option A: Using a TorchScript or PyTorch Checkpoint

Create a config in `config/model/my_pytorch_model.yaml`:

```yaml
model_name: my_model
kwargs:
  model_path: /path/to/your/model.pt
  batch_size: 128
```

Run embedding and scoring:

```bash
uv run python embed.py +experiment=pytorch +model=my_pytorch_model
uv run python score.py +experiment=pytorch model_name=my_model
```

### Option B: Defining a Custom PyTorch Embedder

If your model requires custom featurization (e.g. custom tokenization), implement your wrapper by subclassing `SmilesEmbedder`:

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
   uv run python download.py
   ```
2. **Generate Embeddings**:
   ```bash
   uv run python embed.py +experiment=huggingface +model=my_hf_model
   # Or using the wrapper script:
   ./embed_wrapper.sh model_wrappers/huggingface
   ```
3. **Score Embeddings**:
   ```bash
   uv run python score.py +experiment=huggingface model_name=my_hf_model
   # Or using the background runner:
   ./run_scoring.sh huggingface
   ```
4. Results are stored in the SQLite database at `data/meta.db`.
