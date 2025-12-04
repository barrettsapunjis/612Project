from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sentalyzer.absa.SVMUtil.training import SVMABSAConfig
from sentalyzer.data.samples import ABSASample, AspectCombineFn, default_setfit_combiner
from sentalyzer.absa.utils.reporting import write_eval_report


def write_svm_eval_report(
    samples: Iterable[ABSASample],
    cfg: SVMABSAConfig,
    report_dir: str,
    data_path: str,
    combine_fn: AspectCombineFn = default_setfit_combiner,
    prefix: str = "svm",
) -> None:
    """
    Train/val split with the same seed as training and persist metrics.

    This mirrors the SVM training pipeline: TF-IDF + LinearSVC.
    """
    samples = list(samples)
    texts = [combine_fn(s.text, s.aspect) for s in samples]
    labels = [str(s.label) for s in samples]

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
    clf = LinearSVC(
        C=cfg.C,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    clf.fit(X_train_vec, y_train)
    X_val_vec = vectorizer.transform(X_val)
    y_pred = clf.predict(X_val_vec)

    # Write report using gold y_val (aligned to y_pred)
    aligned_samples = [ABSASample(text="", aspect="", label=lbl) for lbl in y_val]

    write_eval_report(
        samples=aligned_samples,
        preds=y_pred,
        report_dir=report_dir,
        prefix=f"{prefix}_train",
        data_path=data_path,
        model_path=cfg.model_dir,
        extra={
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "ngram_range": cfg.ngram_range,
            "max_features": cfg.max_features,
            "min_df": cfg.min_df,
            "C": cfg.C,
            "class_weight": cfg.class_weight,
        },
    )
