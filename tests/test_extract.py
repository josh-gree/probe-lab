import numpy as np
import pandas as pd
import pytest

from probe_lab import PROMPT_COLUMN, Activations, PromptSet, extract

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


@pytest.fixture(scope="module")
def acts(promptset):
    # model load + forward happens once for the whole module
    return extract(promptset, model="SmolLM2-135M", pooling="mean")


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
def test_source_aligned(acts, promptset):
    assert acts.promptset is promptset
    assert list(acts.meta.columns) == list(promptset.df.columns)
    assert len(acts.meta) == len(promptset)
    assert acts.pooling == "mean"
    assert acts.model == "SmolLM2-135M"


@pytest.mark.model
def test_finite(acts):
    assert np.isfinite(acts.acts).all()


def test_unimplemented_pooling_raises(promptset):
    # checked before any model load, so this is fast
    with pytest.raises(NotImplementedError, match="mean"):
        extract(promptset, pooling="last")
