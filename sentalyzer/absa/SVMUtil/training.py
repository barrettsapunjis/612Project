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
from sentalyzer.absa.SVMUtil.svm_absa_model import (
    SVMABSAModel,
    SVMABSAConfig,
    prepare_dense_input,
    prepare_embeddings,
    prepare_sparse_input,
)
from sentalyzer.embeddings import EmbeddingConfig
from sentalyzer.data.text_cleaning import build_default_cleaner
from sentalyzer.embeddings.vectorizer import build_vectorizer, VectorizerConfig, default_vectorizer



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
    embedding_cfg: EmbeddingConfig | None = None,
) -> None:
    os.makedirs(cfg.model_dir, exist_ok=True)

    if not cfg.use_tfidf and not cfg.use_dense:
        raise ValueError("At least one of use_tfidf or use_dense must be True.")
    
    print("-- STARTING SVM TRAINING --")
    print("config parameters: ")
    print(f"\ntest size: {cfg.test_size}")
    print(f"\ntrain count: {len(samples)}")
    print(f"\nusing tfidf: {cfg.use_tfidf}")
    print(f"\nusing dense: {cfg.use_dense}")
    print(f"\nembedding config: {embedding_cfg}")
    print(f"\ncombine function: {cfg.combine_fn}")
    print(f"\nmodel directory: {cfg.model_dir}")
    print(f"\nrandom state: {cfg.random_state}")
    print("\nsamples count: ", len(samples))
    print("\nusing tfidf: ", cfg.use_tfidf)
    print("\nusing dense: ", cfg.use_dense)

    print("\npreparing training data")
    texts, labels = prepare_training_data(samples, cfg.combine_fn)
    print("training data prepared")
    print("\ntexts count: ", len(texts))
    print("\nlabels count: ", len(labels))
    print("\nexample text: ", texts[0])
    print("\nexample label: ", labels[0])

    # --- Hyperparameter search (currently disabled) ---
    # best_params = search_svm_hyperparams(texts, labels, cfg)
    # cfg.ngram_range = best_params.get("tfidf__ngram_range", cfg.ngram_range)
    # cfg.min_df = best_params.get("tfidf__min_df", cfg.min_df)
    # cfg.max_features = best_params.get("tfidf__max_features", cfg.max_features)
    # cfg.C = best_params.get("clf__C", cfg.C)
    # cfg.class_weight = best_params.get("clf__class_weight", cfg.class_weight)

    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    # ----- Build feature matrices -----
    if cfg.vectorizer_config is None:
        vectorizer = default_vectorizer()
    else:
        vectorizer = build_vectorizer(cfg.vectorizer_config)
    cleaner = build_default_cleaner()

    X_train_texts = [cleaner(text) for text in X_train_texts]
    X_val_texts = [cleaner(text) for text in X_val_texts]
    print("\nX_train_texts count: ", len(X_train_texts))
    print("\nX_val_texts count: ", len(X_val_texts))
    print("\nexample X_train_text: ", X_train_texts[0])
    print("\nexample X_val_text: ", X_val_texts[0])

    # Fit TF-IDF once on training texts if it's enabled
    if cfg.use_tfidf:
        vectorizer.fit(X_train_texts)

    # Optional TF-IDF / dense features
    if cfg.use_dense and cfg.use_tfidf:
        print("using dense and tfidf")
        X_train_vec = prepare_dense_input(
            vectorizer=vectorizer,
            texts=X_train_texts,
            embedding_config=embedding_cfg,
        )
        X_val_vec = prepare_dense_input(
            vectorizer=vectorizer,
            texts=X_val_texts,
            embedding_config=embedding_cfg,
        )
    elif cfg.use_dense:
        print("using dense")
        X_train_vec = prepare_embeddings(texts=X_train_texts, embedding_config=embedding_cfg)
        X_val_vec = prepare_embeddings(texts=X_val_texts, embedding_config=embedding_cfg)
    elif cfg.use_tfidf:
        print("using tfidf")
        X_train_vec = prepare_sparse_input(vectorizer=vectorizer, texts=X_train_texts)
        X_val_vec = prepare_sparse_input(vectorizer=vectorizer, texts=X_val_texts)

    print("\nexample X_train_vec: ", X_train_vec[0])
    print("\nexample X_val_vec: ", X_val_vec[0])

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


    meta: dict = {
        "labels": sorted(list(set(labels))),
        "ngram_range": cfg.ngram_range,
        "max_features": cfg.max_features,
        "min_df": cfg.min_df,
        "C": cfg.C,
        "class_weight": cfg.class_weight,
        "use_tfidf": cfg.use_tfidf,
        "use_dense": cfg.use_dense,
    }
    # Persist embedding config (if used) so inference can rebuild the embedding model.
    if embedding_cfg is not None:
        meta["embedding_config"] = {
            "model_id": embedding_cfg.model_id,
            "device": embedding_cfg.device,
            "batch_size": embedding_cfg.batch_size,
            "normalize": embedding_cfg.normalize,
        }
        print(f"Saving embedding config to {cfg_path}")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved vectorizer to {vec_path}")
    print(f"Saved classifier to {clf_path}")
    print(f"Saved config to {cfg_path}")
