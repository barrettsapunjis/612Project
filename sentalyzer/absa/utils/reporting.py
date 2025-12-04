from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sentalyzer.data.samples import ABSASample


def write_eval_report(
    *,
    samples: Sequence[ABSASample],
    preds: Iterable[str],
    report_dir: str = "./reports",
    prefix: str = "eval",
    data_path: str | None = None,
    model_path: str | None = None,
    extra: dict | None = None,
) -> Path:
    """
    Persist evaluation metrics (accuracy, classification report, confusion matrix) to a timestamped JSON.

    Args:
        samples: gold-labeled samples
        preds: model predictions (aligned with samples)
        report_dir: output directory
        prefix: filename prefix (e.g., 'svm_train', 'inference')
        data_path: optional data source path for traceability
        model_path: optional model path for traceability
        extra: optional additional metadata
    """
    samples = list(samples)
    preds = list(preds)
    gold = [s.label for s in samples if s.label is not None]
    preds = preds[: len(gold)]

    labels = sorted(set(gold + preds)) if gold else []
    acc = accuracy_score(gold, preds) if gold else None
    report = classification_report(gold, preds, output_dict=True, zero_division=0) if gold else {}
    cm = confusion_matrix(gold, preds, labels=labels).tolist() if gold and preds else []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_eval_{timestamp}.json"

    # Ensure any Path-like inputs are converted to plain strings for JSON.
    data_path_str = str(data_path) if data_path is not None else None
    model_path_str = str(model_path) if model_path is not None else None

    payload = {
        "timestamp": timestamp,
        "data_path": data_path_str,
        "model_path": model_path_str,
        "labels": labels,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    if extra:
        payload.update(extra)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[report] Wrote eval report to {out_path}")
    return out_path
