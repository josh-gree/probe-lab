"""PromptSet: the framework's input layer.

A PromptSet is a thin wrapper over a pandas DataFrame that must contain a
`prompt` column. Every other column is metadata. The (reset) row index is the
`id`: row i is example i. Prompt *construction* lives outside the framework —
users emit a parquet however they like, and `PromptSet.from_parquet` ingests it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROMPT_COLUMN = "prompt"


class PromptSet:
    """Prompts + arbitrary metadata, backed by a DataFrame.

    Args:
        df: must contain a `prompt` column; other columns are metadata.

    Raises:
        ValueError: if the `prompt` column is missing, or the frame is empty.
    """

    def __init__(self, df: pd.DataFrame):
        if PROMPT_COLUMN not in df.columns:
            raise ValueError(
                f"DataFrame must have a {PROMPT_COLUMN!r} column; "
                f"got columns {list(df.columns)}"
            )
        if len(df) == 0:
            raise ValueError("PromptSet is empty: the DataFrame has no rows")
        # reset so id == row index 0..N-1, regardless of the input's index
        self.df = df.reset_index(drop=True)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "PromptSet":
        """Load a PromptSet from a parquet file with a `prompt` column."""
        return cls(pd.read_parquet(path))

    @property
    def prompts(self) -> list[str]:
        """The prompt strings, in id order."""
        return self.df[PROMPT_COLUMN].tolist()

    @property
    def metadata_columns(self) -> list[str]:
        """Names of the metadata columns (everything but `prompt`)."""
        return [c for c in self.df.columns if c != PROMPT_COLUMN]

    @property
    def ids(self) -> np.ndarray:
        """The ids `0..N-1` (== row index)."""
        return self.df.index.to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"PromptSet(n={len(self)}, metadata={self.metadata_columns})"
