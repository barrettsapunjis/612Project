"""
Pipeline inference script for SetFit ABSA:
  - Load text data via the SentFin dataloader
  - Run NER to extract candidate ORG aspects
  - Run SetFit ABSA model to score sentiments
  - If labels exist, report validation metrics + confusion matrix (timestamped report)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

# Use the public sentalyzer API for a simpler, stable interface.
from sentalyzer import (
    ABSASample,
    extract_all_org_aspects,
    load_sentfin_absa_samples,
    load_sentfin_df,
    SetFitABSAModel,
    validate_predictions,
    write_eval_report,
)

# Base paths (script executed from repo root)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data"

# Configurable paths and limits
DATA_CSV = DATA_PATH / "data_42_1-9" / "test.csv"
SETFIT_MODEL_DIR = MODEL_PATH / "setfit-absa-sentfin-bester"
MAX_TEXT_ROWS = None # Set None to use all rows
MIN_NER_SCORE = 0.7
REPORT_DIR = "./reports"


def load_texts(path: str, max_rows: int | None) -> List[str]:
    """Load raw texts from SentFin (no labels) for inference."""
    df = load_sentfin_df(path, max_rows=max_rows)
    return df["text"].tolist()


def extract_aspect_samples(texts: Iterable[str]) -> List[ABSASample]:
    """Run NER to get aspects and form ABSASample objects without labels."""
    samples: List[ABSASample] = []
    for text in texts:
        aspects = extract_all_org_aspects(text, min_score=MIN_NER_SCORE)
        for ac in aspects:
            samples.append(ABSASample(text=text, aspect=ac.aspect, label=None))
    return samples


def print_inference(samples: List[ABSASample], preds: List[str]) -> None:
    print(f"[info] Inference complete on {len(samples)} aspect mentions (NER-derived):")
    for sample, pred in zip(samples, preds):
        print("---")
        print("Text:", sample.text)
        print("Aspect:", sample.aspect)
        print("Predicted sentiment:", pred)


def evaluate_labeled(model: SetFitABSAModel) -> None:
    """If labels exist in the test CSV, run evaluation + confusion matrix."""
    gold_samples = load_sentfin_absa_samples(DATA_CSV, max_rows=MAX_TEXT_ROWS)
    if not gold_samples:
        print("[validate] No labeled samples available for evaluation.")
        return

    gold_preds = model.predict_samples(gold_samples)
    print("\n[validate] Evaluation on labeled test split:")
    validate_predictions(gold_samples, gold_preds)
    write_eval_report(
        samples=gold_samples,
        preds=gold_preds,
        report_dir=str(REPORT_DIR),
        prefix="inference_setfit",
        data_path=str(DATA_CSV),
        model_path=str(SETFIT_MODEL_DIR),
        extra={"max_rows": MAX_TEXT_ROWS},
    )


def run_inference() -> None:
    texts = load_texts(DATA_CSV, MAX_TEXT_ROWS)
    if not texts:
        print(f"[error] No texts loaded from {DATA_CSV}")
        return

    samples = extract_aspect_samples(texts)
    if not samples:
        print("[warn] No aspects extracted; nothing to score.")
        return

    model = SetFitABSAModel.from_dir(SETFIT_MODEL_DIR)

    #preds = model.predict_samples(samples)
    #print_inference(samples, preds)
    evaluate_labeled(model)


if __name__ == "__main__":
    run_inference()
