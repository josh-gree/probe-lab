"""Extraction: PromptSet -> residual-stream activations.

Runs a PromptSet's prompts through a causal LM and pulls the residual stream
(every layer's hidden state, via `output_hidden_states`), reduced over tokens
into one vector per (example, layer). The result is row-aligned to the source
PromptSet: row i of `acts` corresponds to prompt i.

Only `mean` pooling is implemented so far. Right-padding is used (the
transformers default); it's correct for the rotary-position models we target
(SmolLM2 / Qwen / Pythia) because RoPE encodes *relative* position, so a uniform
pad shift leaves real-token activations unchanged given the attention mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_lab.prompts import PromptSet

# short name -> HF id; a raw HF id may also be passed through directly
MODELS = {"SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M"}
DEFAULT_MODEL = "SmolLM2-135M"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(model: str):
    """Load (model, tokenizer, device) for a registry name or raw HF id."""
    hf_id = MODELS.get(model, model)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = get_device()
    net = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=torch.float32,  # float32 is safest/most accurate on MPS
        output_hidden_states=True,
    ).to(device)
    net.eval()
    return net, tokenizer, device


@dataclass
class Activations:
    """Residual-stream activations, row-aligned to `promptset`.

    `acts` has shape (N, n_layers, H) where n_layers == num blocks + 1
    (index 0 is the embedding layer).
    """

    acts: np.ndarray            # (N, L+1, H) float32
    promptset: PromptSet        # source PromptSet (prompts, ids, metadata)
    model: str
    pooling: str

    @property
    def meta(self) -> pd.DataFrame:
        """Convenience accessor for the source metadata table."""
        return self.promptset.df

    @property
    def n_layers(self) -> int:
        return self.acts.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.acts.shape[2]

    def layer(self, layer: int) -> np.ndarray:
        """Activations at a single layer: shape (N, H)."""
        return self.acts[:, layer, :]

    def __len__(self) -> int:
        return self.acts.shape[0]

    def __repr__(self) -> str:
        return (
            f"Activations(n={len(self)}, n_layers={self.n_layers}, "
            f"H={self.hidden_size}, model={self.model!r}, pooling={self.pooling!r})"
        )


@torch.no_grad()
def extract(ps: PromptSet, *, model: str = DEFAULT_MODEL,
            pooling: str = "mean", batch_size: int = 16) -> Activations:
    """Extract per-layer residual activations for every prompt in `ps`.

    Args:
        ps: the prompts to run.
        model: registry name (see MODELS) or a raw HF id.
        pooling: token reduction. Only "mean" is implemented so far.
        batch_size: prompts per forward pass.

    Returns:
        Activations with `acts` of shape (N, n_layers, H), aligned to `ps`.
    """
    if pooling != "mean":
        raise NotImplementedError(
            f"only 'mean' pooling is implemented so far, got {pooling!r}"
        )

    net, tokenizer, device = _load_model(model)
    prompts = ps.prompts

    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = net(**enc)

        mask = enc["attention_mask"].unsqueeze(-1).float()      # (B, T, 1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = [((h * mask).sum(dim=1) / counts).float().cpu()
                  for h in out.hidden_states]                    # each (B, H)
        chunks.append(torch.stack(pooled).permute(1, 0, 2))      # (B, L+1, H)

    acts = torch.cat(chunks).numpy()
    # alignment: exactly one row per prompt, in prompt order
    assert acts.shape[0] == len(ps), (acts.shape[0], len(ps))
    return Activations(acts=acts, promptset=ps, model=model, pooling=pooling)
