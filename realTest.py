"""
Real-world test evaluation script for comparing SVM and SetFit ABSA models.

This script:
1. Loads test data from CSV
2. Extracts aspects using NER
3. Runs predictions with both models
4. Generates comprehensive report with statistics and detailed results
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from sentalyzer import (
    SVMABSAModel,
    SetFitABSAModel,
    concat_aspect,
    extract_all_org_aspects_batch,
    load_absa_samples_from_csv,
    ABSASample,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_CSV = "data/test_set_edited.csv"
SVM_MODEL_DIR = "models/svm_sentfin-dense-idf"
SETFIT_MODEL_DIR = "models/setfit-absa-sentfin-bester"
MIN_NER_SCORE = 0.7
OUTPUT_DIR = "reports"
REPORT_PREFIX = "realtest"


def main():
    print("=" * 80)
    print("REAL-WORLD TEST EVALUATION")
    print("=" * 80)
    print(f"Data: {DATA_CSV}")
    print(f"SVM Model: {SVM_MODEL_DIR}")
    print(f"SetFit Model: {SETFIT_MODEL_DIR}")
    print(f"Min NER Score: {MIN_NER_SCORE}")
    print()

    # Load and prepare data
    print("[1/5] Loading test data...")
    samples = load_absa_samples_from_csv(DATA_CSV)
    print(f"  Loaded {len(samples)} samples from CSV")

    print("[2/5] Extracting aspects using NER...")
    texts = [sample.text for sample in samples]
    aspects = extract_all_org_aspects_batch(texts, min_score=MIN_NER_SCORE)
    
    new_samples = []
    for text, aspect_list in zip(texts, aspects):
        if aspect_list:
            for ac in aspect_list:
                new_samples.append(ABSASample(text=text, aspect=ac.aspect, label=None))
        else:
            # If no aspects found, still include the sample with empty aspect
            new_samples.append(ABSASample(text=text, aspect="", label=None))
    
    print(f"  Created {len(new_samples)} aspect samples ({len([s for s in new_samples if s.aspect])} with aspects)")

    # Load models
    print("[3/5] Loading models...")
    svm_model = SVMABSAModel.from_dir(str(SVM_MODEL_DIR), combine_fn=concat_aspect)
    setfit_model = SetFitABSAModel.from_dir(str(SETFIT_MODEL_DIR), combine_fn=concat_aspect)
    print("  Models loaded successfully")

    # Run predictions
    print("[4/5] Running predictions...")
    svm_preds = svm_model.predict_samples(new_samples)
    svm_scores = svm_model.predict_scores(new_samples)
    setfit_preds = setfit_model.predict_samples(new_samples)
    setfit_scores = setfit_model.predict_scores(new_samples)
    print("  Predictions complete")

    # Generate statistics
    print("[5/5] Generating report...")
    
    # Prediction distributions
    svm_dist = Counter(svm_preds)
    setfit_dist = Counter(setfit_preds)
    
    # Agreement analysis
    agreements = sum(1 for s, f in zip(svm_preds, setfit_preds) if s == f)
    agreement_rate = agreements / len(new_samples) if new_samples else 0
    
    # Confidence analysis (average max confidence)
    svm_confidences = [max(score.values()) for score in svm_scores]
    setfit_confidences = [max(score.values()) for score in setfit_scores]
    avg_svm_conf = sum(svm_confidences) / len(svm_confidences) if svm_confidences else 0
    avg_setfit_conf = sum(setfit_confidences) / len(setfit_confidences) if setfit_confidences else 0

    # Create detailed results DataFrame
    results_data = []
    for i, (sample, svm_pred, svm_score, setfit_pred, setfit_score) in enumerate(
        zip(new_samples, svm_preds, svm_scores, setfit_preds, setfit_scores)
    ):
        results_data.append({
            "id": i + 1,
            "text": sample.text[:100] + "..." if len(sample.text) > 100 else sample.text,
            "aspect": sample.aspect,
            "svm_prediction": svm_pred,
            "svm_confidence": max(svm_score.values()),
            "svm_scores": json.dumps(svm_score),
            "setfit_prediction": setfit_pred,
            "setfit_confidence": max(setfit_score.values()),
            "setfit_scores": json.dumps(setfit_score),
            "agreement": "Yes" if svm_pred == setfit_pred else "No",
        })
    
    results_df = pd.DataFrame(results_data)

    # Save detailed results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{REPORT_PREFIX}_detailed_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Create summary report
    summary = {
        "timestamp": timestamp,
        "data_path": DATA_CSV,
        "svm_model": SVM_MODEL_DIR,
        "setfit_model": SETFIT_MODEL_DIR,
        "min_ner_score": MIN_NER_SCORE,
        "statistics": {
            "total_samples": len(new_samples),
            "samples_with_aspects": len([s for s in new_samples if s.aspect]),
            "svm_predictions": dict(svm_dist),
            "setfit_predictions": dict(setfit_dist),
            "agreement": {
                "count": agreements,
                "rate": round(agreement_rate, 4),
            },
            "confidence": {
                "svm_avg": round(avg_svm_conf, 4),
                "setfit_avg": round(avg_setfit_conf, 4),
            },
        },
    }
    
    json_path = output_dir / f"{REPORT_PREFIX}_summary_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print formatted report
    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(new_samples)}")
    print(f"  Samples with aspects: {len([s for s in new_samples if s.aspect])}")
    
    print(f"\nSVM Model Predictions:")
    for label, count in sorted(svm_dist.items()):
        pct = (count / len(new_samples)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\nSetFit Model Predictions:")
    for label, count in sorted(setfit_dist.items()):
        pct = (count / len(new_samples)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\nModel Agreement:")
    print(f"  Agreement: {agreements}/{len(new_samples)} ({agreement_rate*100:.1f}%)")
    print(f"  Disagreement: {len(new_samples) - agreements}/{len(new_samples)} ({(1-agreement_rate)*100:.1f}%)")
    
    print(f"\nAverage Confidence:")
    print(f"  SVM: {avg_svm_conf:.4f}")
    print(f"  SetFit: {avg_setfit_conf:.4f}")
    
    print(f"\nOutput Files:")
    print(f"  Detailed results: {csv_path}")
    print(f"  Summary report: {json_path}")
    print("=" * 80)
    
    # Show sample predictions
    print("\nSample Predictions (first 10):")
    print("-" * 80)
    for i in range(min(10, len(results_data))):
        r = results_data[i]
        print(f"\n[{i+1}] Text: {r['text']}")
        print(f"    Aspect: {r['aspect'] or '(none)'}")
        print(f"    SVM: {r['svm_prediction']} (conf: {r['svm_confidence']:.3f})")
        print(f"    SetFit: {r['setfit_prediction']} (conf: {r['setfit_confidence']:.3f})")
        print(f"    Agreement: {r['agreement']}")


if __name__ == "__main__":
    main()