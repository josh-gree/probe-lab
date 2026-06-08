import numpy as np
import pandas as pd
import pytest

from probe_lab import PROMPT_COLUMN, PromptSet


@pytest.fixture
def parquet_path(tmp_path):
    """Write a small prompt+metadata parquet and return its path."""
    df = pd.DataFrame(
        {
            PROMPT_COLUMN: ["43 + 17 = 60", "to be or not to be", "12 + 5 = 17"],
            "label": [0, 1, 0],
            "source": ["arithmetic", "shakespeare", "arithmetic"],
        }
    )
    path = tmp_path / "dataset.parquet"
    df.to_parquet(path)
    return path


def test_from_parquet_loads(parquet_path):
    ps = PromptSet.from_parquet(parquet_path)
    assert len(ps) == 3
    assert ps.prompts == ["43 + 17 = 60", "to be or not to be", "12 + 5 = 17"]


def test_metadata_survives(parquet_path):
    ps = PromptSet.from_parquet(parquet_path)
    assert ps.metadata_columns == ["label", "source"]
    assert ps.df["label"].tolist() == [0, 1, 0]
    assert ps.df["source"].tolist() == ["arithmetic", "shakespeare", "arithmetic"]


def test_ids_are_range(parquet_path):
    ps = PromptSet.from_parquet(parquet_path)
    assert np.array_equal(ps.ids, np.arange(3))


def test_index_is_reset():
    # a non-default index must be reset so ids are 0..N-1
    df = pd.DataFrame({PROMPT_COLUMN: ["a", "b"]}, index=[7, 99])
    ps = PromptSet(df)
    assert np.array_equal(ps.ids, np.arange(2))
    assert ps.prompts == ["a", "b"]


def test_missing_prompt_column_raises():
    df = pd.DataFrame({"text": ["a"], "label": [0]})
    with pytest.raises(ValueError, match="prompt"):
        PromptSet(df)


def test_empty_raises():
    df = pd.DataFrame({PROMPT_COLUMN: []})
    with pytest.raises(ValueError, match="empty"):
        PromptSet(df)


def test_repr(parquet_path):
    ps = PromptSet.from_parquet(parquet_path)
    assert repr(ps) == "PromptSet(n=3, metadata=['label', 'source'])"
