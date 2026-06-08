import numpy as np
import pandas as pd
import pytest

from probe_lab import PROMPT_COLUMN, Activations, Pooling, PromptSet, extract

# SmolLM2-135M dimensions: 30 transformer blocks + 1 embedding layer, hidden 576.
N_LAYERS = 31
HIDDEN = 576


@pytest.fixture(scope="module")
def promptset():
    # varying lengths, to exercise right-padding within a batch
    df = pd.DataFrame(
        {
            PROMPT_COLUMN: [
                "12 + 5 = 17",
                "To be, or not to be, that is the question.",
                "hello",
                "The quick brown fox jumps over the lazy dog.",
            ],
            "label": [0, 1, 1, 1],
        }
    )
    return PromptSet(df)


@pytest.fixture(scope="module", params=[Pooling.MEAN, Pooling.LAST])
def pooling(request):
    return request.param


@pytest.fixture(scope="module")
def acts(promptset, pooling):
    # one model load + forward per pooling mode
    return extract(promptset, model="SmolLM2-135M", pooling=pooling)


@pytest.mark.model
def test_shape(acts, promptset):
    assert isinstance(acts, Activations)
    assert acts.acts.shape == (len(promptset), N_LAYERS, HIDDEN)
    assert acts.n_layers == N_LAYERS
    assert acts.hidden_size == HIDDEN
    assert len(acts) == len(promptset)


@pytest.mark.model
def test_layer_accessor(acts, promptset):
    assert acts.layer(0).shape == (len(promptset), HIDDEN)       # embeddings
    assert acts.layer(N_LAYERS - 1).shape == (len(promptset), HIDDEN)  # last block


@pytest.mark.model
def test_source_aligned(acts, promptset, pooling):
    assert acts.promptset is promptset
    assert list(acts.meta.columns) == list(promptset.df.columns)
    assert len(acts.meta) == len(promptset)
    assert acts.pooling == pooling
    assert acts.model == "SmolLM2-135M"


@pytest.mark.model
def test_finite(acts):
    assert np.isfinite(acts.acts).all()


@pytest.mark.model
def test_residual_alignment(acts, promptset, pooling):
    # "hello" (index 2) is the shortest prompt, so inside the batch it is
    # right-padded. Extracting it ALONE (batch of 1, no padding) must reproduce
    # its batched row to float precision -> proves (a) row i maps to prompt i and
    # (b) padding does not leak into the kept activations. Holds for both poolings.
    i = 2
    single = extract(
        PromptSet(pd.DataFrame({PROMPT_COLUMN: [promptset.prompts[i]]})),
        model="SmolLM2-135M",
        pooling=pooling,
    )
    single_row = single.acts[0]              # (n_layers, H), unpadded
    # residual-stream magnitude varies hugely across layers (RMS ~0.1 to ~850),
    # so compare *relative* to scale, not with an absolute tolerance.
    scale = np.sqrt((single_row ** 2).mean())
    self_rel = np.abs(acts.acts[i] - single_row).max() / scale
    # contrast: the same single activation vs a DIFFERENT prompt's row
    other_rel = np.abs(single_row - acts.acts[(i + 1) % len(acts)]).max() / scale

    # embedding layer is position-free, so padding can't touch it at all
    assert np.array_equal(acts.acts[i, 0], single_row[0])
    assert self_rel < 1e-3, f"row {i} does not match its own prompt: {self_rel}"
    assert other_rel > 1e-1, f"row {i} matches a different prompt too: {other_rel}"


@pytest.mark.model
def test_mean_and_last_differ(promptset):
    # guard against `last` silently computing the mean
    mean = extract(promptset, model="SmolLM2-135M", pooling=Pooling.MEAN)
    last = extract(promptset, model="SmolLM2-135M", pooling=Pooling.LAST)
    assert not np.allclose(mean.acts, last.acts)


def test_invalid_pooling_raises(promptset):
    # coercion to Pooling rejects unknown values before any model load (fast)
    with pytest.raises(ValueError):
        extract(promptset, pooling="sum")
