"""
Public, typed API for the `sentalyzer` package.

This module re-exports the small set of types and functions that pipeline
scripts (and external callers) are expected to use, so that importing is
simple and stable:

    from sentalyzer import (
        ABSASample,
        load_sentfin_df,
        load_sentfin_absa_samples,
        extract_all_org_aspects_batch,
        SVMABSAModel,
        SVMABSAConfig,
        train_svm_absa,
        SetFitABSAModel,
        SetFitABSAConfig,
        train_setfit_absa,
    )

The internal submodules remain available for more advanced usage, but most
callers should not need to import from deep paths.
"""

from __future__ import annotations

from typing import List, Sequence

# Data layer
from .data.samples import (
    ABSASample,
    AspectCombineFn,
    concat_aspect,
    default_setfit_combiner,
    inline_marker_combiner,
    sep_token_combiner,
    print_predictions,
    validate_predictions,
    mask_aspect,
)
from .data.sentfin import (
    load_sentfin_df,
    load_sentfin_absa_samples,
    sentfin_df_to_absa_samples,
)

# Extraction layer
from .extraction.api import (
    get_extractor,
    extract_all_org_aspects,
    extract_all_org_aspects_batch,
    extract_targeted_org_aspects,
)
from .extraction.types import EntityMention, AspectCandidate, Span

# ABSA – SVM
from .absa.SVMUtil import (
    SVMABSAConfig,
    SVMABSAModel,
    train_svm_absa,
    prepare_training_data,
)

# ABSA – SetFit
from .absa.setfitUtil import (
    SetFitABSAModel,
    SetFitABSAConfig,
    train_setfit_absa,
    aspects_to_setfit_df,
    samples_to_setfit_dataset,
    split_dataset,
)

from .embeddings import (
    EmbeddingConfig,
    load_embedding_model,
    encode_texts,
)

# Shared reporting utilities
from .absa.utils.reporting import write_eval_report


__all__ = [
    # Data / samples
    "ABSASample",
    "AspectCombineFn",
    "concat_aspect",
    "default_setfit_combiner",
    "inline_marker_combiner",
    "sep_token_combiner",
    "print_predictions",
    "validate_predictions",
    "load_sentfin_df",
    "load_sentfin_absa_samples",
    "sentfin_df_to_absa_samples",
    "mask_aspect",
    # Extraction
    "get_extractor",
    "extract_all_org_aspects",
    "extract_all_org_aspects_batch",
    "extract_targeted_org_aspects",
    "EntityMention",
    "AspectCandidate",
    "Span",
    # ABSA – SVM
    "SVMABSAConfig",
    "SVMABSAModel",
    "train_svm_absa",
    "prepare_training_data",
    # ABSA – SetFit
    "SetFitABSAModel",
    "SetFitABSAConfig",
    "train_setfit_absa",
    "aspects_to_setfit_df",
    "samples_to_setfit_dataset",
    "split_dataset",
    # Reporting
    "write_eval_report",
]


