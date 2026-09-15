import os
import torch
import logging as log
from typing import Optional, Literal
from transformers import AutoModel, AutoTokenizer
from src.common.utils import get_device, batch
from src.common.types import SmilesEmbedder


def smiles_to_selfies(smiles: str) -> Optional[str]:
    try:
        import selfies as sf
        from rdkit import Chem
        default_constraints = sf.get_semantic_constraints()
        default_constraints['P-1'] = 6
        default_constraints['Fe'] = 10
        default_constraints['Fe+3'] = 10
        default_constraints['Fe+2'] = 9
        sf.set_semantic_constraints(default_constraints)
        try:
            return sf.encoder(Chem.CanonSmiles(smiles))
        except Exception:
            return sf.encoder(smiles)
    except Exception as e:
        log.warning(f"Error converting SMILES to SELFIES for '{smiles}': {e}")
        return None


class HuggingFaceSmilesEmbedder(SmilesEmbedder):
    """
    General embedder for any local or HuggingFace Hub molecular model.
    Accepts either a local path (e.g. './checkpoints/my_model') or a HuggingFace Hub identifier.
    """
    def __init__(
        self,
        model_path: str,
        pooling: Literal['mean', 'cls', 'pooler'] = 'mean',
        batch_size: int = 128,
        max_length: int = 512,
        device: Optional[str] = None,
        use_selfies: bool = False,
        **_kwargs
    ):
        self._model_path = model_path
        self._model_name = os.path.basename(os.path.normpath(model_path))
        self._pooling = pooling
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = get_device(device)
        self._use_selfies = use_selfies

        log.info(f"Loading HuggingFace model from: {model_path} on {self._device} (pooling={pooling})")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self._device)
        self._model.eval()

    def _pool_outputs(self, outputs, attention_mask):
        if self._pooling == 'cls':
            return outputs.last_hidden_state[:, 0, :]
        elif self._pooling == 'pooler' and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:  # 'mean' pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

    def _embed_batch(self, batch_smiles):
        if self._use_selfies:
            batch_inputs = [smiles_to_selfies(s) or "" for s in batch_smiles]
        else:
            batch_inputs = list(batch_smiles)

        encoded = self._tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt"
        ).to(self._device)

        outputs = self._model(**encoded)
        return self._pool_outputs(outputs, encoded['attention_mask'])

    def forward(self, smiles):
        outputs = []
        with torch.no_grad():
            for b in batch(smiles, n=self._batch_size):
                out = self._embed_batch(b).to('cpu')
                outputs.append(out)
        return torch.cat(outputs, dim=0).numpy()

    @property
    def name(self):
        return self._model_name

    @property
    def device_used(self):
        return self._device.type.split(':')[0]


class MolFormerEmbedder(HuggingFaceSmilesEmbedder):
    def __init__(self, model_path: str = "ibm/MoLFormer-XL-both-10pct", **kwargs):
        pooling = kwargs.pop('pooling', 'pooler')
        super().__init__(model_path=model_path, pooling=pooling, **kwargs)


class ChemBERTaEmbedder(HuggingFaceSmilesEmbedder):
    def __init__(self, model_path: str = "DeepChem/ChemBERTa-10M-MTR", **kwargs):
        pooling = kwargs.pop('pooling', 'cls')
        super().__init__(model_path=model_path, pooling=pooling, **kwargs)


class ChemGPTEmbedder(HuggingFaceSmilesEmbedder):
    def __init__(self, model_path: str = "ncfrey/ChemGPT-4.7M", **kwargs):
        pooling = kwargs.pop('pooling', 'mean')
        use_selfies = kwargs.pop('use_selfies', True)
        super().__init__(model_path=model_path, pooling=pooling, use_selfies=use_selfies, **kwargs)



def get_embedder(name: str, **kwargs) -> SmilesEmbedder:
    name_lower = name.lower()
    if 'molformer' in name_lower:
        return MolFormerEmbedder(model_path=name, **kwargs)
    elif 'chemberta' in name_lower:
        return ChemBERTaEmbedder(model_path=name, **kwargs)
    elif 'chemgpt' in name_lower:
        return ChemGPTEmbedder(model_path=name, **kwargs)
    else:
        return HuggingFaceSmilesEmbedder(model_path=name, **kwargs)
