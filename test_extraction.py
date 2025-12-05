"""
Extraction test script for evaluating NER aspect extraction performance.

This script:
1. Loads SentFin dataset with ground truth aspects
2. Runs NER extraction on the texts
3. Compares extracted aspects with ground truth
4. Generates comprehensive metrics and reports
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

from sentalyzer import (
    extract_all_org_aspects_batch,
    load_sentfin_df,
    load_sentfin_absa_samples,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_CSV = "data/data_42_1-9/train.csv"
MIN_NER_SCORE = 0.7
MAX_ROWS = None  # Set to None to use all rows, or specify a number
OUTPUT_DIR = "reports"
REPORT_PREFIX = "extraction_test"


def normalize_aspect(aspect: str) -> str:
    """Normalize aspect string for comparison (lowercase, strip)."""
    return aspect.strip().lower()


def extract_ground_truth_aspects(df: pd.DataFrame) -> List[Tuple[str, Set[str]]]:
    """
    Extract ground truth aspects from SentFin DataFrame.
    
    Returns list of (text, set of ground truth aspects) tuples.
    """
    results = []
    for _, row in df.iterrows():
        text = row["text"]
        aspects_json = row.get("aspects_json", "{}")
        
        try:
            aspects_dict = json.loads(aspects_json) if isinstance(aspects_json, str) else {}
            gt_aspects = {normalize_aspect(str(asp)) for asp in aspects_dict.keys()}
        except (json.JSONDecodeError, AttributeError):
            gt_aspects = set()
        
        results.append((text, gt_aspects))
    
    return results


def calculate_extraction_metrics(
    ground_truth: List[Set[str]],
    extracted: List[List[str]],
) -> dict:
    """
    Calculate precision, recall, and F1 for aspect extraction.
    
    Args:
        ground_truth: List of sets of ground truth aspects per text
        extracted: List of lists of extracted aspect strings per text
    
    Returns:
        Dictionary with metrics
    """
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    
    per_text_results = []
    
    for gt_set, ext_list in zip(ground_truth, extracted):
        ext_set = {normalize_aspect(asp) for asp in ext_list}
        
        # Calculate matches
        tp = len(gt_set & ext_set)  # Intersection
        fp = len(ext_set - gt_set)   # Extracted but not in ground truth
        fn = len(gt_set - ext_set)   # In ground truth but not extracted
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        per_text_results.append({
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_count": len(gt_set),
            "extracted_count": len(ext_set),
        })
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "per_text": per_text_results,
    }


def main():
    print("=" * 80)
    print("EXTRACTION TEST EVALUATION")
    print("=" * 80)
    print(f"Data: {DATA_CSV}")
    print(f"Min NER Score: {MIN_NER_SCORE}")
    print(f"Max Rows: {MAX_ROWS if MAX_ROWS else 'All'}")
    print()

    # Load SentFin data
    print("[1/4] Loading SentFin dataset...")
    df = load_sentfin_df(DATA_CSV, max_rows=MAX_ROWS)
    print(f"  Loaded {len(df)} texts from SentFin dataset")

    # Extract ground truth aspects
    print("[2/4] Extracting ground truth aspects...")
    text_gt_pairs = extract_ground_truth_aspects(df)
    texts = [pair[0] for pair in text_gt_pairs]
    ground_truth_aspects = [pair[1] for pair in text_gt_pairs]
    
    total_gt_aspects = sum(len(gt) for gt in ground_truth_aspects)
    texts_with_gt = sum(1 for gt in ground_truth_aspects if len(gt) > 0)
    print(f"  Found {total_gt_aspects} ground truth aspects across {texts_with_gt} texts")

    # Run NER extraction
    print("[3/4] Running NER extraction...")
    extracted_aspects_batch = extract_all_org_aspects_batch(texts, min_score=MIN_NER_SCORE)
    extracted_aspects = [[ac.aspect for ac in aspects] for aspects in extracted_aspects_batch]
    
    total_extracted = sum(len(ext) for ext in extracted_aspects)
    texts_with_extracted = sum(1 for ext in extracted_aspects if len(ext) > 0)
    print(f"  Extracted {total_extracted} aspects across {texts_with_extracted} texts")

    # Calculate metrics
    print("[4/4] Calculating metrics...")
    metrics = calculate_extraction_metrics(ground_truth_aspects, extracted_aspects)
    
    # Create detailed results DataFrame
    results_data = []
    for i, (text, gt_set, ext_list, per_text_metrics) in enumerate(
        zip(texts, ground_truth_aspects, extracted_aspects, metrics["per_text"])
    ):
        ext_set = {normalize_aspect(asp) for asp in ext_list}
        
        # Find matches, false positives, false negatives
        matches = gt_set & ext_set
        false_positives = ext_set - gt_set
        false_negatives = gt_set - ext_set
        
        results_data.append({
            "id": i + 1,
            "text": text[:150] + "..." if len(text) > 150 else text,
            "gt_aspects": ", ".join(sorted(gt_set)) if gt_set else "(none)",
            "extracted_aspects": ", ".join(sorted(ext_set)) if ext_set else "(none)",
            "matches": ", ".join(sorted(matches)) if matches else "(none)",
            "false_positives": ", ".join(sorted(false_positives)) if false_positives else "(none)",
            "false_negatives": ", ".join(sorted(false_negatives)) if false_negatives else "(none)",
            "tp": per_text_metrics["tp"],
            "fp": per_text_metrics["fp"],
            "fn": per_text_metrics["fn"],
            "gt_count": per_text_metrics["gt_count"],
            "extracted_count": per_text_metrics["extracted_count"],
        })
    
    results_df = pd.DataFrame(results_data)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{REPORT_PREFIX}_detailed_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Create summary report
    summary = {
        "timestamp": timestamp,
        "data_path": DATA_CSV,
        "min_ner_score": MIN_NER_SCORE,
        "max_rows": MAX_ROWS,
        "statistics": {
            "total_texts": len(texts),
            "texts_with_ground_truth": texts_with_gt,
            "texts_with_extracted": texts_with_extracted,
            "total_ground_truth_aspects": total_gt_aspects,
            "total_extracted_aspects": total_extracted,
            "metrics": {
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
            },
        },
    }
    
    json_path = output_dir / f"{REPORT_PREFIX}_summary_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print formatted report
    print()
    print("=" * 80)
    print("EXTRACTION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  Total texts: {len(texts)}")
    print(f"  Texts with ground truth aspects: {texts_with_gt}")
    print(f"  Texts with extracted aspects: {texts_with_extracted}")
    print(f"  Total ground truth aspects: {total_gt_aspects}")
    print(f"  Total extracted aspects: {total_extracted}")
    
    print(f"\nExtraction Metrics:")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['true_positives']}/{metrics['true_positives'] + metrics['false_positives']})")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['true_positives']}/{metrics['true_positives'] + metrics['false_negatives']})")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\n  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    # Show examples
    print(f"\nExample Results:")
    print("-" * 80)
    
    # Show some perfect matches
    perfect_matches = [i for i, m in enumerate(metrics["per_text"]) 
                      if m["tp"] > 0 and m["fp"] == 0 and m["fn"] == 0]
    if perfect_matches:
        print(f"\nPerfect Matches (showing first 3):")
        for idx in perfect_matches[:3]:
            r = results_data[idx]
            print(f"\n  [{r['id']}] Text: {r['text']}")
            print(f"      Ground Truth: {r['gt_aspects']}")
            print(f"      Extracted:    {r['extracted_aspects']}")
    
    # Show some false negatives
    false_negatives = [i for i, m in enumerate(metrics["per_text"]) if m["fn"] > 0]
    if false_negatives:
        print(f"\nFalse Negatives (missed aspects, showing first 3):")
        for idx in false_negatives[:3]:
            r = results_data[idx]
            print(f"\n  [{r['id']}] Text: {r['text']}")
            print(f"      Ground Truth: {r['gt_aspects']}")
            print(f"      Extracted:    {r['extracted_aspects']}")
            print(f"      Missed:       {r['false_negatives']}")
    
    # Show some false positives
    false_positives = [i for i, m in enumerate(metrics["per_text"]) if m["fp"] > 0]
    if false_positives:
        print(f"\nFalse Positives (incorrectly extracted, showing first 3):")
        for idx in false_positives[:3]:
            r = results_data[idx]
            print(f"\n  [{r['id']}] Text: {r['text']}")
            print(f"      Ground Truth: {r['gt_aspects']}")
            print(f"      Extracted:    {r['extracted_aspects']}")
            print(f"      Incorrect:    {r['false_positives']}")
    
    print(f"\nOutput Files:")
    print(f"  Detailed results: {csv_path}")
    print(f"  Summary report: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

