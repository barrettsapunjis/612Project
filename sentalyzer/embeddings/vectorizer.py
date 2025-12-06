from __future__ import annotations

"""
TF-IDF vectorizer utilities.

These helpers centralize how we build and apply a TfidfVectorizer so that
multiple components (SVM training, inference, analysis) can share consistent
settings.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class VectorizerConfig:
    """
    Configuration for a TF-IDF text vectorizer.
    """

    ngram_range: Tuple[int, int] = (1, 2)
    max_features: Optional[int] = 50_000
    min_df: int = 1
    sublinear_tf: bool = True
    max_df: float = 0.95


def build_vectorizer(cfg: VectorizerConfig) -> TfidfVectorizer:
    """
    Create an unfitted TfidfVectorizer from a VectorizerConfig.
    """
    return TfidfVectorizer(
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        sublinear_tf=cfg.sublinear_tf,
        max_df=cfg.max_df,
    )

def default_vectorizer() -> TfidfVectorizer:
    """
    Convenience helper returning a TfidfVectorizer with default settings.
    """
    cfg = VectorizerConfig(
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=1,
        sublinear_tf=True,
        max_df=0.95,
    )
    return build_vectorizer(cfg)


def transform_texts(
    vectorizer: TfidfVectorizer,
    texts: Iterable[str],
) -> object:
    """
    Apply a fitted TfidfVectorizer to raw texts.
    """
    texts = list(texts)
    return vectorizer.transform(texts)



