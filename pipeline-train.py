"""
Pipeline training script:
  - Load SentFin ABSA samples
  - Train SVM ABSA model (SetFit option provided but commented)
  - Persist artifacts to models/svm_sentfin
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

# Public sentalyzer API imports for a simpler, stable surface.
from sentalyzer import (
    SVMABSAConfig,
    default_setfit_combiner,
    load_sentfin_absa_samples,
    train_svm_absa,
    mask_aspect,
)
from sentalyzer.absa.SVMUtil.reporting import write_svm_eval_report
from sentalyzer.embeddings import EmbeddingConfig
from sentalyzer.data.samples import concat_aspect

# Base paths (resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"

# Configurable paths and limits
DATA_CSV = DATA_PATH / "data_42_8-2" / "train.csv"
SVM_MODEL_DIR = MODEL_PATH / "svm_sentfin"
# SETFIT_MODEL_DIR = MODEL_PATH / "setfit-absa-sentfin"
MAX_ROWS = None  # set to an int to limit rows
REPORT_DIR = BASE_DIR / "reports"


def main() -> None:
    samples = load_sentfin_absa_samples(
        path=str(DATA_CSV),
        max_rows=MAX_ROWS,
    )
    if not samples:
        raise SystemExit(f"[error] No training samples loaded from {DATA_CSV}")

    cfg = SVMABSAConfig(
        model_dir=str(SVM_MODEL_DIR),
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=1,
        C=1.0,
        class_weight="balanced",
        random_state=42,
        test_size=0.8,
        use_tfidf=True,
        combine_fn=concat_aspect,
    )

    # Train SVM ABSA
    train_svm_absa(samples=samples, cfg=cfg)

    # Evaluate on a held-out split using the same random seed for reproducibility

    write_svm_eval_report(
        samples=samples,
        cfg=cfg,
        report_dir=REPORT_DIR,
        data_path=str(DATA_CSV),
        combine_fn=default_setfit_combiner,
        prefix="svm",
    )


if __name__ == "__main__":
    main()
