from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import os
import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator

from sentalyzer.data.samples import (
    ABSASample,
    AspectCombineFn,
    default_setfit_combiner,
    validate_predictions,
)
from sentalyzer.data.text_cleaning import build_default_cleaner
from sentalyzer.embeddings import encode_texts, EmbeddingConfig, load_embedding_model
from sentalyzer.embeddings.vectorizer import VectorizerConfig, build_vectorizer, default_vectorizer
from sentence_transformers import SentenceTransformer

def prepare_sparse_input(
    vectorizer: TfidfVectorizer,
    texts: Iterable[str],
) -> object:
    """
    Build a sparse TF-IDF feature matrix for the given input texts.

    Assumes the vectorizer has already been fitted.
    """
    texts = list(texts)
    return vectorizer.transform(texts)


def prepare_embeddings(
    texts: Iterable[str],
    embedding_config: EmbeddingConfig | None = None,
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """
    Encode a collection of texts into dense transformer-based embeddings.
    """
    texts = list(texts)

    
    if model is None:
        model = load_embedding_model(embedding_config)

    embeddings = encode_texts(
        model=model,
        texts=texts,
        batch_size=embedding_config.batch_size,
        normalize=embedding_config.normalize,
    )
    return embeddings


def prepare_dense_input(
    vectorizer: TfidfVectorizer,
    texts: Iterable[str],
    embedding_config: EmbeddingConfig | None = None,
    embedding_model: SentenceTransformer | None = None,
) -> np.ndarray:
    """
    Concatenate TF-IDF features and dense transformer embeddings for each text.
    """
    texts = list(texts)
    X_sparse = prepare_sparse_input(vectorizer, texts).toarray()
    X_emb = prepare_embeddings(texts, embedding_config=embedding_config, model=embedding_model)
    return np.concatenate([X_sparse, X_emb], axis=1)


@dataclass
class SVMABSAConfig:
    model_dir: str
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int | None = 50_000
    min_df: int = 2
    C: float = 1.0
    class_weight: str | None = "balanced"
    random_state: int = 42
    test_size: float = 0.1
    use_tfidf: bool = True
    use_dense: bool = False
    embedding_config: EmbeddingConfig | None = None
    combine_fn: AspectCombineFn = default_setfit_combiner
    vectorizer_config: VectorizerConfig = default_vectorizer()


@dataclass
class SVMABSAModel:
    """
    ABSA sentiment classifier using a classical SVM.

    This is designed as a drop-in sibling to SetFitABSAModel:
    - Same combine_fn contract (text, aspect) -> combined string
    - Same public methods: from_dir, predict_samples, predict_labels, evaluate
    """

    model_dir: str
    vectorizer: TfidfVectorizer | None
    classifier: BaseEstimator
    combine_fn: AspectCombineFn = default_setfit_combiner
    # Feature configuration used at training time
    use_tfidf: bool = True
    use_dense: bool = False
    embedding_config: EmbeddingConfig | None = None
    embedding_model: SentenceTransformer | None = None
    labels: List[str] | None = None  # optional metadata


    @classmethod
    def from_dir(
        cls,
        model_dir: str,
        combine_fn: AspectCombineFn = default_setfit_combiner,
    ) -> "SVMABSAModel":
        """
        Load persisted vectorizer + classifier from model_dir.

        Expected files:
        - vectorizer.joblib
        - classifier.joblib
        - config.json (optional, e.g. {"labels": ["negative", "neutral", "positive"]})
        """
        vec_path = os.path.join(model_dir, "vectorizer.joblib")
        clf_path = os.path.join(model_dir, "classifier.joblib")
        cfg_path = os.path.join(model_dir, "config.json")

        try:
            vectorizer = joblib.load(vec_path)
        except FileNotFoundError:
            vectorizer = None
        classifier = joblib.load(clf_path)

        labels: List[str] | None = None
        use_tfidf = True
        use_dense = False
        embedding_config: EmbeddingConfig | None = None
        embedding_model: SentenceTransformer | None = None

        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            labels = cfg.get("labels")
            use_tfidf = cfg.get("use_tfidf", True)
            use_dense = cfg.get("use_dense", False)
            ec = cfg.get("embedding_config")
            if ec is not None:
                embedding_config = EmbeddingConfig(
                    model_id=ec.get("model_id", EmbeddingConfig().model_id),
                    device=ec.get("device"),
                    batch_size=ec.get("batch_size", EmbeddingConfig().batch_size),
                    normalize=ec.get("normalize", EmbeddingConfig().normalize),
                )
                embedding_model = load_embedding_model(embedding_config)

        return cls(
            model_dir=model_dir,
            vectorizer=vectorizer,
            classifier=classifier,
            combine_fn=combine_fn,
            use_tfidf=use_tfidf,
            use_dense=use_dense,
            embedding_config=embedding_config,
            embedding_model=embedding_model,
            labels=labels,
        )

    def predict_samples(
        self,
        samples: Iterable[ABSASample],
        embedding_config: EmbeddingConfig | None = None,
    ) -> List[str]:
        """
        Predict labels for ABSASample inputs.

        - If embedding_config is provided and a vectorizer is available, use
          TF-IDF + transformer embeddings.
        - If only a vectorizer is available, use TF-IDF features.
        - If no vectorizer is available, fall back to transformer embeddings
          only (dense features).

        Assumes the vectorizer (if present) was already fitted at training
        time and loaded via from_dir.
        """
        # Fall back to the embedding config used at training time when one
        # isn't explicitly provided.
        cfg = embedding_config or self.embedding_config

        samples = list(samples)
        texts = [self.combine_fn(s.text, s.aspect) for s in samples]

        if self.use_dense and self.use_tfidf and cfg is not None and self.vectorizer is not None:
            # TF-IDF + transformer embeddings
            X = prepare_dense_input(self.vectorizer, texts, embedding_config=cfg, embedding_model=self.embedding_model)
        elif self.use_tfidf and self.vectorizer is not None:
            # TF-IDF only
            X = prepare_sparse_input(self.vectorizer, texts)
        else:
            # Dense-only (no vectorizer persisted)
            X = prepare_embeddings(texts, embedding_config=cfg, model=self.embedding_model)

        preds = self.classifier.predict(X)
        return [str(p) for p in preds]
    

    def predict_labels(self, texts: Sequence[str], aspects: Sequence[str]) -> List[str]:
        assert len(texts) == len(aspects)
        samples = [ABSASample(text=t, aspect=a, label=None) for t, a in zip(texts, aspects)]
        return self.predict_samples(samples)

    def predict_scores(
        self,
        samples: Iterable[ABSASample],
        embedding_config: EmbeddingConfig | None = None,
    ) -> List[dict[str, float]]:
        """
        Return per-label scores for each sample, using the classifier's
        decision function or probabilities when available.

        For LinearSVC and similar models, this will be the raw decision
        margins (higher -> more confident). For probabilistic classifiers,
        it will be the predicted probabilities.

        Uses the same feature preparation logic as predict_samples to ensure
        consistency with the training configuration.
        """
        # Fall back to the embedding config used at training time when one
        # isn't explicitly provided.
        cfg = embedding_config or self.embedding_config

        samples = list(samples)
        texts = [self.combine_fn(s.text, s.aspect) for s in samples]

        # Use the same feature preparation as predict_samples
        if self.use_dense and self.use_tfidf and cfg is not None and self.vectorizer is not None:
            # TF-IDF + transformer embeddings
            X = prepare_dense_input(self.vectorizer, texts, embedding_config=cfg, embedding_model=self.embedding_model)
        elif self.use_tfidf and self.vectorizer is not None:
            # TF-IDF only
            X = prepare_sparse_input(self.vectorizer, texts)
        else:
            # Dense-only (no vectorizer persisted)
            X = prepare_embeddings(texts, embedding_config=cfg, model=self.embedding_model)

        if hasattr(self.classifier, "decision_function"):
            raw_scores = self.classifier.decision_function(X)
        elif hasattr(self.classifier, "predict_proba"):
            raw_scores = self.classifier.predict_proba(X)
        else:
            raise RuntimeError("Classifier does not support decision_function or predict_proba.")

        raw_scores = np.asarray(raw_scores)

        # Ensure 2D shape (n_samples, n_classes)
        if raw_scores.ndim == 1:
            raw_scores = raw_scores.reshape(-1, 1)

        # Derive label order from saved metadata or classifier classes_
        label_order: List[str]
        if self.labels:
            label_order = list(self.labels)
        elif hasattr(self.classifier, "classes_"):
            label_order = [str(c) for c in self.classifier.classes_]
        else:
            label_order = [str(i) for i in range(raw_scores.shape[1])]

        # If we have a binary decision_function with only one column but two labels,
        # mirror scores as [-margin, +margin] to produce per-class scores.
        if raw_scores.shape[1] == 1 and len(label_order) == 2:
            margins = raw_scores[:, 0]
            raw_scores = np.vstack([-margins, margins]).T

        scores: List[dict[str, float]] = []
        for row in raw_scores:
            scores.append({label: float(score) for label, score in zip(label_order, row)})
        return scores

    def evaluate(self, samples: Iterable[ABSASample]) -> None:
        samples = list(samples)
        preds = self.predict_samples(samples)
        validate_predictions(samples, preds)
