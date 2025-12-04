from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .samples import ABSASample, _df_to_absa_samples_from_aspects_json


def load_sentfin_df(
    path: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load SentFin CSV in its native format and normalize columns.

    Original columns:
      - "S No."      → id
      - "Title"      → text
      - "Decisions"  → aspects_json  (JSON mapping aspect -> sentiment)
      - "Words"      → word_count

    Returns a DataFrame with at least:
      - id
      - text
      - aspects_json
      - word_count
    """
    df = pd.read_csv(path)

    rename_map = {
        "S No.": "id",
        "Title": "text",
        "Decisions": "aspects_json",
        "Words": "word_count",
    }
    df = df.rename(columns=rename_map)

    # Basic sanity: ensure required cols exist
    required = {"id", "text", "aspects_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SentFin file is missing required columns: {missing}")

    if start is not None or end is not None:
        df = df.iloc[start:end]

    if max_rows is not None:
        df = df.head(max_rows)

    return df


def sentfin_df_to_absa_samples(df: pd.DataFrame) -> List[ABSASample]:
    """
    Convert a normalized SentFin DataFrame into generic ABSASample objects
    using the ground-truth 'aspects_json' column.

    This is for ABSA evaluation (not for NER-driven aspect discovery).
    """
    return _df_to_absa_samples_from_aspects_json(df, text_col="text", aspects_col="aspects_json")


def load_sentfin_absa_samples(
    path: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> List[ABSASample]:
    """
    Convenience loader: path → normalized df → ABSASample list.
    """
    df = load_sentfin_df(path, start=start, end=end, max_rows=max_rows)
    return sentfin_df_to_absa_samples(df)
