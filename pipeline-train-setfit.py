"""
Pipeline training script for SetFit ABSA:
  - Load SentFin ABSA samples
  - Train SetFit ABSA model
  - Persist artifacts to setfit-absa-sentfin
  - Write a timestamped eval report
"""

from __future__ import annotations

from pathlib import Path

# Public sentalyzer API imports for a simpler, stable surface.
from sentalyzer import (
    SetFitABSAConfig,
    default_setfit_combiner,
    load_sentfin_absa_samples,
    train_setfit_absa,
)
from sentalyzer.absa.setfitUtil.reporting import write_setfit_eval_report

# Base paths (resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data"


# Configurable paths and limits
DATA_CSV = DATA_PATH / "data_42_1-9" / "train.csv"
SETFIT_MODEL_DIR = MODEL_PATH / "setfit-absa-sentfin"
MAX_ROWS = None  # set to an int to limit rows
REPORT_DIR = BASE_DIR / "reports"


def main() -> None:
    samples = load_sentfin_absa_samples(
        path=str(DATA_CSV),
        max_rows=MAX_ROWS,
    )
    if not samples:
        raise SystemExit(f"[error] No training samples loaded from {DATA_CSV}")

    cfg = SetFitABSAConfig(
        model_id="sentence-transformers/paraphrase-mpnet-base-v2",
        output_dir=str(SETFIT_MODEL_DIR),
        batch_size=16,
        num_iterations=5,
        num_epochs=1,
        test_size=0.8,
        random_state=42,
    )

    metrics = train_setfit_absa(
        samples=samples,
        cfg=cfg,
        combine_fn=default_setfit_combiner,
    )
    print("[eval] Training metrics:", metrics)

    write_setfit_eval_report(
        samples=samples,
        cfg=cfg,
        report_dir=REPORT_DIR,
        data_path=str(DATA_CSV),
        combine_fn=default_setfit_combiner,
        prefix="setfit",
    )


if __name__ == "__main__":
    main()
