from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import os
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator

from sentalyzer.data.samples import (
    ABSASample,
    AspectCombineFn,
    default_setfit_combiner,
    validate_predictions,
)


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


@dataclass
class SVMABSAModel:
    """
    ABSA sentiment classifier using a classical SVM.

    This is designed as a drop-in sibling to SetFitABSAModel:
    - Same combine_fn contract (text, aspect) -> combined string
    - Same public methods: from_dir, predict_samples, predict_labels, evaluate
    """

    model_dir: str
    vectorizer: TfidfVectorizer
    classifier: BaseEstimator
    combine_fn: AspectCombineFn = default_setfit_combiner
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

        vectorizer = joblib.load(vec_path)
        classifier = joblib.load(clf_path)

        labels = None
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            labels = cfg.get("labels")

        return cls(
            model_dir=model_dir,
            vectorizer=vectorizer,
            classifier=classifier,
            combine_fn=combine_fn,
            labels=labels,
        )

    def _combine_texts(self, samples: Iterable[ABSASample]) -> List[str]:
        return [self.combine_fn(s.text, s.aspect) for s in samples]

    def predict_samples(self, samples: Iterable[ABSASample]) -> List[str]:
        samples = list(samples)
        texts = self._combine_texts(samples)

        X = self.vectorizer.transform(texts)
        preds = self.classifier.predict(X)

        # Ensure list[str]
        return [str(p) for p in preds]

    def predict_labels(self, texts: Sequence[str], aspects: Sequence[str]) -> List[str]:
        assert len(texts) == len(aspects)
        samples = [ABSASample(text=t, aspect=a, label=None) for t, a in zip(texts, aspects)]
        return self.predict_samples(samples)

    def evaluate(self, samples: Iterable[ABSASample]) -> None:
        samples = list(samples)
        preds = self.predict_samples(samples)
        validate_predictions(samples, preds)
