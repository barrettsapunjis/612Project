from __future__ import annotations

"""
Transformer-based text embedding utilities.

These helpers build on `sentence-transformers` so that multiple parts of the
project (e.g., alternative classifiers, retrieval, analysis) can share the
same embedding model and encoding logic.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    """
    Configuration for a transformer-based sentence embedding model.
    """

    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None  # "cpu", "cuda", etc. If None, auto-detect.
    batch_size: int = 32
    normalize: bool = True


def load_embedding_model(cfg: EmbeddingConfig) -> SentenceTransformer:
    """
    Load a SentenceTransformer model according to the provided config.
    """
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(cfg.model_id, device=device)
    return model


def encode_texts(
    model: SentenceTransformer,
    texts: Iterable[str],
    batch_size: Optional[int] = None,
    normalize: Optional[bool] = None,
) -> np.ndarray:
    """
    Encode a collection of texts into dense embeddings.

    Returns a NumPy array of shape (n_texts, dim).
    """
    texts = list(texts)
    if not texts:
        # Return an empty embedding matrix with the correct second dimension.
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    bs = batch_size or 32
    norm = normalize if normalize is not None else True

    embeddings = model.encode(
        texts,
        batch_size=bs,
        convert_to_numpy=True,
        normalize_embeddings=norm,
    )
    return embeddings



