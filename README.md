# probe-lab

A small personal toolkit for activation probing of language models.

**Pipeline:** externally-generated parquet (a `prompt` column + arbitrary
metadata) → `PromptSet` → extract residual activations → per-layer linear probes.

Prompt *construction* is intentionally **not** part of the framework: you emit a
parquet however you like, and the framework consumes it. See `DESIGN.md` for the
full spec and design decisions.

## Status

Early. Implemented so far:

- `PromptSet.from_parquet` — load prompts + metadata from a parquet file.

Next: residual-activation extraction and the linear-probe analysis.

## Usage

```python
from probe_lab import PromptSet

ps = PromptSet.from_parquet("dataset.parquet")  # needs a `prompt` column
ps.prompts            # list of prompt strings, in id order
ps.metadata_columns   # everything except `prompt`
ps.ids                # 0..N-1 (row i == example i)
```

## Develop

```bash
uv run pytest
```
