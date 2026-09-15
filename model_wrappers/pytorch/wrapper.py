import os
import torch
import numpy as np
import logging as log
from typing import Optional, Callable
from src.common.types import SmilesEmbedder
from src.common.utils import get_device, batch


class PyTorchSmilesEmbedder(SmilesEmbedder):
    """
    General embedder for benchmarking local PyTorch molecular models.

    Supports:
      - TorchScript models (.pt, .pth) loaded via torch.jit.load
      - Serialized torch.nn.Module instances loaded via torch.load
      - Directly passing an instantiated torch.nn.Module
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        featurizer: Optional[Callable] = None,
        batch_size: int = 128,
        device: Optional[str] = None,
        output_dim: Optional[int] = None,
        **_kwargs
    ):
        self._device = get_device(device)
        self._batch_size = batch_size
        self._featurizer = featurizer
        self._output_dim = output_dim

        if model is not None:
            self._model = model.to(self._device)
            self._model_name = getattr(model, "name", model.__class__.__name__)
        elif model_path is not None:
            self._model_name = os.path.basename(os.path.normpath(model_path)).split('.')[0]
            log.info(f"Loading local PyTorch model from: {model_path} on {self._device}")
            try:
                self._model = torch.jit.load(model_path, map_location=self._device)
            except Exception:
                self._model = torch.load(model_path, map_location=self._device)
        else:
            raise ValueError("Either 'model_path' or 'model' must be provided.")

        self._model.eval()

    def process_batch(self, smiles_batch):
        """Processes a batch of SMILES. Override this method if custom featurization is required."""
        if self._featurizer is not None:
            inputs = self._featurizer(smiles_batch)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self._device)
            with torch.no_grad():
                out = self._model(inputs)
        else:
            with torch.no_grad():
                out = self._model(smiles_batch)

        if isinstance(out, torch.Tensor):
            return out.detach().cpu()
        elif isinstance(out, np.ndarray):
            return torch.from_numpy(out)
        return torch.tensor(out)

    def forward(self, smiles):
        outputs = []
        for b in batch(smiles, n=self._batch_size):
            try:
                out = self.process_batch(b)
            except Exception as e:
                log.warning(f"Batch processing failed ({e}), falling back to sample-by-sample")
                batch_outs = []
                for s in b:
                    try:
                        batch_outs.append(self.process_batch([s]))
                    except Exception as s_err:
                        log.error(f"Failed to embed molecule '{s}': {s_err}")
                        dim = self._output_dim or (outputs[0].shape[-1] if outputs else 512)
                        batch_outs.append(torch.full((1, dim), float('nan')))
                out = torch.cat(batch_outs, dim=0)
            outputs.append(out)

        return torch.cat(outputs, dim=0).numpy()

    @property
    def name(self):
        return self._model_name

    @property
    def device_used(self):
        return self._device.type.split(':')[0]


def get_embedder(name: str, **kwargs):
    """
    Returns a PyTorch embedder for the given model name or checkpoint path.
    """
    return PyTorchSmilesEmbedder(model_path=name, **kwargs)
