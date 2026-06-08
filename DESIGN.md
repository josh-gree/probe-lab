# interp — API spec (draft)

A small personal framework for activation probing.

**Scope (exactly this, nothing more):**

> Make prompts with metadata → run through a model → extract residual
> activations → fit linear probes to each layer wrt some metadata.

The creative part — *constructing* the prompts — is **not** in the framework.
You write whatever script you like (arithmetic generator, corpus sampler, …)
and its only job is to emit a **parquet file**. The framework starts there.

Core mental model: a **DataFrame (N rows, one `prompt` column + any metadata
columns) joined to an aligned activation tensor (N, L, H) by row index.**

---

## 1. PromptSet — just a parquet

A PromptSet is a thin wrapper over a DataFrame that has a `prompt` column.
Every other column is metadata. The row index is the `id`.

```python
class PromptSet:
    df: pd.DataFrame                 # must contain a "prompt" column

    @classmethod
    def from_parquet(cls, path) -> "PromptSet"
    def hash(self) -> str            # content hash of df, for caching
```

```
# example parquet contents
prompt                  | source       | label | a  | b  | sum
"43 + 17 = 60"          | arithmetic   | 0     | 43 | 17 | 60
" peer. Humphrey of..." | shakespeare  | 1     |    |    |
```

That's the whole input layer. Our existing `build_dataset.py` just becomes a
script that writes `data/dataset.parquet` instead of jsonl — nothing about
arithmetic or Shakespeare leaks into the framework.

*(No template/grid/corpus builders in the framework — those are your prompt
scripts' problem, and an extension if we ever want them.)*

---

## 2. extract — residual activations, aligned & cached

```python
class Pooling(StrEnum):   # LAST="last", MEAN="mean"
    ...

def extract(ps: PromptSet, *, model: str,
            pooling: Pooling | str = Pooling.LAST,   # accepts the enum or its value
            layers="all") -> Activations
```

- Pulls the **residual stream** per layer via `output_hidden_states=True`
  (embedding + every block) — exactly what we already have working.
- `pooling` is a `Pooling` enum (default `LAST`); `extract` coerces a passed
  string so `"last"`/`"mean"` also work. We **right-pad** (pads on the end).
  `Pooling.MEAN` averages over the real tokens (`(h*mask).sum(1) / mask.sum(1)`).
  `Pooling.LAST` takes the last real token, which with right-padding is *not* the
  final column, so we index it via the mask: `h[arange(B), attn.sum(1) - 1]`.
  Just the two.
- Why padding side is a non-issue here: our models (SmolLM2 / Qwen / Pythia) use
  **rotary (RoPE)** position embeddings, which encode *relative* position — the
  QK score depends only on `(m - n)`. Padding uniformly shifts all real-token
  positions by the same constant, leaving relative offsets (and therefore
  attention, and therefore the residual stream) unchanged, provided pads are
  masked out of attention. So no `position_ids` correction is needed. Right-pad
  is chosen only because it keeps positions `0..L-1` by default (zero caveats)
  and matches the code already verified by `verify_alignment` at ~1e-4.
  (This relative-invariance does *not* hold for absolute/learned positional
  embeddings like GPT-2, where padding side would matter.)
- Guarantees alignment: row `i` of `acts` ↔ row `i` of `ps.df`. Runs the
  existing `verify_alignment` self-check on a sample automatically.

```python
class Activations:
    acts: np.ndarray        # (N, n_layers, H)
    promptset: PromptSet    # the source PromptSet (prompts, ids, metadata)
    model: str
    pooling: Pooling

    @property
    def meta(self) -> pd.DataFrame:      # convenience -> promptset.df
    def layer(self, l) -> np.ndarray     # (N, H)
```

`Activations` holds the source `PromptSet` (not a bare DataFrame), so it stays
tied to the object guaranteeing the prompt/id invariants; `.meta` exposes
`promptset.df` for the probe stage. `acts` and `meta` are always the same N in
the same order — that single invariant is the framework.

---

## 3. probe — linear probe per layer vs a metadata column

```python
def probe(acts: Activations, target: str, *,
          control: str = "shuffled",     # label-permuted baseline
          test_size=0.3, seed=0) -> ProbeResult

class ProbeResult:
    scores: pd.DataFrame    # layer, score, control_score
    def plot_by_layer(self): ...
```

- `target` names a metadata column (`"label"`, `"is_correct"`, …).
- classification vs regression chosen from the column dtype.
- `control="shuffled"` always reported, so a score means something.

This is exactly our current `run_probe.py`, with the target generalized from
"label" to "any metadata column."

---

## End-to-end (what an experiment looks like)

```python
ps   = PromptSet.from_parquet("data/dataset.parquet")
acts = extract(ps, model="SmolLM2-135M", pooling="last")
res  = probe(acts, target="label")
res.plot_by_layer()
```

Three lines. The only thing that changes between experiments is the parquet.

---

## Caching (modest, but worth it)

The slow step is `extract`; probing is instant. So cache extraction keyed on
`hash(ps) + model + pooling + layers`. The concrete payoff is the one we already
felt: **iterate on probes without re-running the model every time.** Adding new
prompts → new hash → re-extract (a secondary nicety, not the main point).

```
data/cache/<key>.npz     # acts + row index;  meta re-read from the parquet
```

Keep it dead simple; no premature cleverness.

---

## Decision: backend — stick with vanilla `transformers` for now

You asked whether we should just use **TransformerLens / nnsight**. Honest read:

- For *this* scope — residual stream per layer + linear probes — vanilla
  `transformers` (`output_hidden_states=True`) already does the whole job, and
  we've built and **verified** it. No extra dependency, full control, any HF
  model loads.
- TransformerLens / nnsight earn their weight when you want things we are
  explicitly **not** doing yet: sub-layer hooks (MLP / attention / individual
  heads), standardized hook naming, and especially **interventions** (activation
  patching, steering, ablation). They also have their own model-loading path and
  support lists (newer models like SmolLM2 may need extra config).

**Recommendation:** stay vanilla now. **Tripwire to switch:** the day we want
head-level activations or to *intervene* on the residual stream rather than just
read it — that's when TransformerLens pays for itself. Cheap to adopt later
because nothing in this API exposes the backend.

---

## Dependencies to add

- `pandas` + `pyarrow` (parquet I/O). `torch`, `transformers`, `scikit-learn`,
  `numpy` already in.
