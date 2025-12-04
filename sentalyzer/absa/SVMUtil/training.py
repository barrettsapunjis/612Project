from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

from sentalyzer.data.samples import (
    ABSASample,
    AspectCombineFn,
    default_setfit_combiner,
)
from sentalyzer.data.text_cleaning import build_default_cleaner
from sentalyzer.absa.SVMUtil.svm_absa_model import SVMABSAModel, SVMABSAConfig


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
    # Feature flags
    use_tfidf: bool = True
    use_dense: bool = False



def prepare_training_data(
    samples: Iterable[ABSASample],
    combine_fn: AspectCombineFn,
) -> tuple[List[str], List[str], List[np.ndarray]]:
    samples = list(samples)
    cleaner = build_default_cleaner()
    texts = [combine_fn(cleaner(s.text), s.aspect) for s in samples]
    labels = [str(s.label) for s in samples]
    return texts, labels



def train_svm_absa(
    samples: Iterable[ABSASample],
    cfg: SVMABSAConfig,
    combine_fn: AspectCombineFn = default_setfit_combiner,
) -> None:
    os.makedirs(cfg.model_dir, exist_ok=True)

    if not cfg.use_tfidf and not cfg.use_dense:
        raise ValueError("At least one of use_tfidf or use_dense must be True.")

    texts, labels = prepare_training_data(samples, combine_fn)

    # --- Hyperparameter search (currently disabled) ---
    # best_params = search_svm_hyperparams(texts, labels, cfg)
    # cfg.ngram_range = best_params.get("tfidf__ngram_range", cfg.ngram_range)
    # cfg.min_df = best_params.get("tfidf__min_df", cfg.min_df)
    # cfg.max_features = best_params.get("tfidf__max_features", cfg.max_features)
    # cfg.C = best_params.get("clf__C", cfg.C)
    # cfg.class_weight = best_params.get("clf__class_weight", cfg.class_weight)

    X_train_texts, X_val_texts, y_train, y_val, X_train_emb, X_val_emb = train_test_split(
        texts,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    # ----- Build feature matrices -----
    vectorizer: TfidfVectorizer | None = None

    svm_model = SVMABSAModel(
        model_dir=cfg.model_dir,
        vectorizer=vectorizer,
        classifier=classifier,
        combine_fn=combine_fn,
    )

    # Optional TF-IDF features
    if cfg.use_dense and cfg.use_tfidf:
        X_train_vec = svm_model.prepare_dense_input(X_train_texts)
        X_val_vec = svm_model.prepare_dense_input(X_val_texts)
    elif cfg.use_dense:
        X_train_vec = svm_model.prepare_embeddings(X_train_texts)
        X_val_vec = svm_model.prepare_embeddings(X_val_texts)
    elif cfg.use_tfidf:
        X_train_vec = svm_model.prepare_sparse_input(X_train_texts)
        X_val_vec = svm_model.prepare_sparse_input(X_val_texts)
    else:
        X_train_tfidf = None
        X_val_tfidf = None

 

    classifier = LinearSVC(
        C=cfg.C,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
    )
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_val_vec)

    print("config paramerters: ")
    print(f"\ntest size: {cfg.test_size}")
    print(f"\ntrain count: {len(X_train_texts)}")
    print("Validation accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    vec_path = os.path.join(cfg.model_dir, "vectorizer.joblib")
    clf_path = os.path.join(cfg.model_dir, "classifier.joblib")
    cfg_path = os.path.join(cfg.model_dir, "config.json")

    if vectorizer is not None:
        joblib.dump(vectorizer, vec_path)
    joblib.dump(classifier, clf_path)

    meta = {
        "labels": sorted(list(set(labels))),
        "ngram_range": cfg.ngram_range,
        "max_features": cfg.max_features,
        "min_df": cfg.min_df,
        "C": cfg.C,
        "class_weight": cfg.class_weight,
        "use_tfidf": cfg.use_tfidf,
        "use_dense": cfg.use_dense,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved vectorizer to {vec_path}")
    print(f"Saved classifier to {clf_path}")
    print(f"Saved config to {cfg_path}")
