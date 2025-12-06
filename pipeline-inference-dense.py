"""
Generic plug-and-play inference pipeline:
  - Load text data via the SentFin dataloader
  - Run NER to extract candidate ORG aspects
  - Run ABSA model to score sentiments
  - If labels exist (e.g., SentFin test split), report validation metrics + confusion matrix

Default ABSA model: SVM. A SetFit option is provided (commented out).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

# Public sentalyzer API imports: keep pipeline dependencies shallow and typed.
from sentalyzer import (
    ABSASample,
    default_setfit_combiner,
    extract_all_org_aspects_batch,
    load_sentfin_absa_samples,
    load_sentfin_df,
    SVMABSAModel,
    validate_predictions,
    write_eval_report,
    mask_aspect,
    concat_aspect,
)
from sentalyzer.embeddings import EmbeddingConfig
# Base paths (resolved relative to this file)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data"

# Configurable paths and limits
DATA_CSV = DATA_PATH / "data_42_8-2" / "test.csv"
SVM_MODEL_DIR = MODEL_PATH / "svm_sentfin-dense-idf"
# SETFIT_MODEL_DIR = MODEL_PATH / "setfit-absa-sentfin"
MAX_TEXT_ROWS = None  # Set None to use all rows
MIN_NER_SCORE = 0.7
REPORT_DIR = BASE_DIR / "reports"


def load_texts(path: str, max_rows: int | None) -> List[str]:
    """Load raw texts from SentFin (no labels) for inference."""
    df = load_sentfin_df(str(path), max_rows=max_rows)
    return df["text"].tolist()


def extract_aspect_samples(texts: Iterable[str]) -> List[ABSASample]:
    """Run NER (batched) to get aspects and form ABSASample objects without labels."""
    texts = list(texts)
    all_aspects = extract_all_org_aspects_batch(texts, min_score=MIN_NER_SCORE)

    samples: List[ABSASample] = []
    for text, aspects in zip(texts, all_aspects):
        for ac in aspects:
            print("extracted aspect: ", ac.aspect)
            samples.append(ABSASample(text=text, aspect=ac.aspect, label=None))
    return samples


def print_inference(samples: List[ABSASample], preds: List[str]) -> None:
    print(f"[info] Inference complete on {len(samples)} aspect mentions (NER-derived):")
    for sample, pred in zip(samples, preds):
        print("---")
        print("Text:", sample.text)
        print("Aspect:", sample.aspect)
        print("Predicted sentiment:", pred)


def evaluate_labeled(model: SVMABSAModel) -> None:
    """If labels exist in the test CSV, run evaluation + confusion matrix."""
    gold_samples = load_sentfin_absa_samples(str(DATA_CSV), max_rows=MAX_TEXT_ROWS)
    if not gold_samples:
        print("[validate] No labeled samples available for evaluation.")
        return

    gold_preds = model.predict_samples(gold_samples)
    print("\n[validate] Evaluation on labeled test split:")
    validate_predictions(gold_samples, gold_preds)
    write_eval_report(
        samples=gold_samples,
        preds=gold_preds,
        report_dir=REPORT_DIR,
        prefix="inference",
        data_path=str(DATA_CSV),
        model_path=str(SVM_MODEL_DIR),
        extra={"max_rows": MAX_TEXT_ROWS},
    )



def run_inference() -> None:
    print("\nloading texts")
    texts = load_texts(DATA_CSV, MAX_TEXT_ROWS)
    if not texts:
        print(f"[error] No texts loaded from {DATA_CSV}")
        return

    samples = extract_aspect_samples(texts)
    if not samples:
        print("[warn] No aspects extracted; nothing to score.")
        return
    print("\nLoading SVM model")
    # Choose ABSA model: default SVM (SetFit alternative commented)
    model = SVMABSAModel.from_dir(str(SVM_MODEL_DIR), combine_fn=concat_aspect)
    # model = SetFitABSAModel.from_dir(str(SETFIT_MODEL_DIR))

    print("Model loaded, running predicitons\n")
    preds = model.predict_samples(samples)
    #print_inference(samples, preds)
    evaluate_labeled(model)


if __name__ == "__main__":
    run_inference()

