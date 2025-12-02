import argparse
import json
import pandas as pd
from setfit import SetFitModel
from sklearn.metrics import accuracy_score, classification_report

CSV_PATH = "sentfin.csv"
MODEL_DIR = "setfit-absa-sentfin-BEST"

def load_samples(path: str, start: int | None, end: int | None, sample_count: int | None, print_all: bool):
    df = pd.read_csv(path, header=None, names=["id", "text", "aspects_json", "label_id"])
    if start is not None or end is not None:
        df = df.iloc[start:end]
    rows = []
    for _, row in df.iterrows():
        try:
            aspects = json.loads(row["aspects_json"])
        except json.JSONDecodeError:
            continue
        for aspect, sent in aspects.items():
            rows.append({
                "text": row["text"],
                "aspect": aspect,
                "label": sent.lower(),
            })
    if not rows:
        return []
    # Deterministic truncation if sample_count is provided and not printing all.
    if not print_all and sample_count:
        rows = rows[:sample_count]
    for s in rows:
        s["combined"] = s["text"].strip() + " [ASPECT] " + s["aspect"].strip()
    return rows

def print_predictions(samples, preds, label="sampled"):
    print(f"[print] Showing {len(samples)} {label} predictions")
    for sample, pred in zip(samples, preds):
        print("\n---")
        print("Text:", sample["text"])
        print("Aspect:", sample["aspect"])
        print("Gold label:", sample["label"])
        print("Predicted:", pred)

def validate_predictions(samples, preds):
    gold = [s["label"] for s in samples]
    acc = accuracy_score(gold, preds)
    print(f"[validate] Accuracy: {acc:.4f}")
    print("[validate] Classification report:")
    print(classification_report(gold, preds, zero_division=0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-all", action="store_true", help="Print all predictions after slicing.")
    parser.add_argument("--validate", action="store_true", help="Compute accuracy and a classification report.")
    parser.add_argument("--start-row", type=int, default=None, help="Start row (0-based, inclusive) to slice the CSV.")
    parser.add_argument("--end-row", type=int, default=None, help="End row (0-based, exclusive) to slice the CSV.")
    parser.add_argument("--sample-count", type=int, default=None, help="If set, take the first N rows after slicing (ignored with --print-all).")
    args = parser.parse_args()

    samples = load_samples(CSV_PATH, args.start_row, args.end_row, args.sample_count, args.print_all)
    if not samples:
        raise SystemExit("No samples to test.")

    print(f"[load] Loaded {len(samples)} samples for testing (print_all={args.print_all}).")
    model = SetFitModel.from_pretrained(MODEL_DIR)
    print("[model] Loaded trained model.")

    texts = [s["combined"] for s in samples]
    print(texts)
    preds = model.predict(texts)

    label = "all" if args.print_all else "sampled"
    print_predictions(samples, preds, label=label)
    if args.validate:
        validate_predictions(samples, preds)

if __name__ == "__main__":
    main()


