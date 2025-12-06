from __future__ import annotations

"""
Embedding utilities for sentalyzer.

This package provides helpers for transformer-based text embeddings that can
be shared between training and inference code.
"""

from .transformer import EmbeddingConfig, load_embedding_model, encode_texts

__all__ = [
    "EmbeddingConfig",
    "load_embedding_model",
    "encode_texts",
]

