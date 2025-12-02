import argparse
import json
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = "sentfin.csv"
SENTIMENT_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"

def load_data(path: str, start: int | None, end: int | None, random_count: int | None, seed: int) -> pd.DataFrame:
    """Read the raw CSV and expand aspect labels without altering the text."""
    print(f"[load] Reading CSV from {path} ...")
    df = pd.read_csv(path, header=None, names=["id", "text", "aspects_json", "label_id"])
    if start is not None or end is not None:
        df = df.iloc[start:end]
    if random_count:
        df = df.sample(n=min(random_count, len(df)), random_state=seed)
    rows = []
    for _, row in df.iterrows():
        try:
            aspects = json.loads(row["aspects_json"])
        except json.JSONDecodeError:
            continue
        for _, sent in aspects.items():
            rows.append({"text": row["text"], "label": sent.lower()})
    print(f"[load] Parsed aspect rows: {len(rows)}")
    return pd.DataFrame(rows)

def build_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-row", type=int, default=None, help="Start row (0-based, inclusive) to slice the CSV.")
    parser.add_argument("--end-row", type=int, default=None, help="End row (0-based, exclusive) to slice the CSV.")
    parser.add_argument("--random-count", type=int, default=None, help="If set, randomly sample this many rows after slicing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    df = load_data(CSV_PATH, args.start_row, args.end_row, args.random_count, args.seed)
    if df.empty:
        raise SystemExit("No valid rows found in CSV.")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print(f"[split] Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    train_ds = build_dataset(train_df)
    test_ds = build_dataset(test_df)
    print("[datasets] Built HuggingFace datasets.")
    print("[datasets] Train example:", train_ds[0])

    print("[model] Loading SetFit encoder ...")
    model = SetFitModel.from_pretrained(SENTIMENT_MODEL, multi_target_strategy=None)

    print("[train] Starting training ...")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        batch_size=4,
        num_iterations=10,
        num_epochs=3,
        column_mapping={"text": "text", "label": "label"},
    )

    trainer.train()
    print("[train] Done.")

    print("[eval] Evaluating ...")
    metrics = trainer.evaluate()
    print("[eval] Metrics:", metrics)

    print("[save] Saving model to setfit-absa-sentfin ...")
    model.save_pretrained("setfit-absa-sentfin")
    print("[save] Complete.")

if __name__ == "__main__":
    main()
