from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sentalyzer.data.samples import (
    ABSASample,
    AspectCombineFn,
    default_setfit_combiner,
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


def prepare_training_data(
    samples: Iterable[ABSASample],
    combine_fn: AspectCombineFn,
) -> tuple[List[str], List[str]]:
    samples = list(samples)
    texts = [combine_fn(s.text, s.aspect) for s in samples]
    labels = [str(s.label) for s in samples]
    return texts, labels


def train_svm_absa(
    samples: Iterable[ABSASample],
    cfg: SVMABSAConfig,
    combine_fn: AspectCombineFn = default_setfit_combiner,
) -> None:
    os.makedirs(cfg.model_dir, exist_ok=True)

    texts, labels = prepare_training_data(samples, combine_fn)

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )


    vectorizer = TfidfVectorizer(
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
    )
    classifier = LinearSVC(
        C=cfg.C,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_vec, y_train)

    X_val_vec = vectorizer.transform(X_val)
    y_pred = classifier.predict(X_val_vec)

    print("config paramerters: ")
    print(f"\ntest size: {cfg.test_size}")
    print(f"\ntrain count: {len(X_train)}")
    print("Validation accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    vec_path = os.path.join(cfg.model_dir, "vectorizer.joblib")
    clf_path = os.path.join(cfg.model_dir, "classifier.joblib")
    cfg_path = os.path.join(cfg.model_dir, "config.json")

    joblib.dump(vectorizer, vec_path)
    joblib.dump(classifier, clf_path)

    meta = {
        "labels": sorted(list(set(labels))),
        "ngram_range": cfg.ngram_range,
        "max_features": cfg.max_features,
        "min_df": cfg.min_df,
        "C": cfg.C,
        "class_weight": cfg.class_weight,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved vectorizer to {vec_path}")
    print(f"Saved classifier to {clf_path}")
    print(f"Saved config to {cfg_path}")
