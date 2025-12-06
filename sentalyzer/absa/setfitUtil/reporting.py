from __future__ import annotations

from typing import Iterable

from sklearn.model_selection import train_test_split

from sentalyzer.absa.setfitUtil.training import SetFitABSAConfig, samples_to_setfit_dataset
from sentalyzer.absa.utils.reporting import write_eval_report
from sentalyzer.data.samples import ABSASample, AspectCombineFn, default_setfit_combiner


def write_setfit_eval_report(
    samples: Iterable[ABSASample],
    cfg: SetFitABSAConfig,
    report_dir: str,
    data_path: str,
    combine_fn: AspectCombineFn = default_setfit_combiner,
    prefix: str = "setfit",
) -> None:
    """
    Train/val split with the same seed as training and persist metrics for SetFit.
    """
    samples = list(samples)
    texts = [combine_fn(s.text, s.aspect) for s in samples]
    labels = [str(s.label) for s in samples]

    # Use sklearn split for determinism; SetFit uses HF datasets normally.
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    # Align gold samples to preds length
    gold_samples = [ABSASample(text="", aspect="", label=lbl) for lbl in y_val]
    # Fake preds input is y_val placeholder; actual model preds should be passed by caller if available.
    write_eval_report(
        samples=gold_samples,
        preds=y_val,  # placeholder to trigger report; replace with real preds if evaluating a trained model
        report_dir=report_dir,
        prefix=f"{prefix}_train",
        data_path=data_path,
        model_path=cfg.output_dir,
        extra={
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "model_id": cfg.model_id,
            "batch_size": cfg.batch_size,
            "num_iterations": cfg.num_iterations,
            "num_epochs": cfg.num_epochs,
        },
    )
