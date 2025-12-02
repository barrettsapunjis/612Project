import json
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = "sentfin.csv"
SENTIMENT_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
MAX_ROWS = 1000  # cap for quick runs; set None to use all

def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV and expand aspect labels without altering the text."""
    print(f"[load] Reading CSV from {path} ...")
    df = pd.read_csv(path, header=None, names=["id", "text", "aspects_json", "label_id"])
    rows = []
    for _, row in df.iterrows():
        try:
            aspects = json.loads(row["aspects_json"])
        except json.JSONDecodeError:
            continue
        for _, sent in aspects.items():
            rows.append({"text": row["text"], "label": sent.lower()})
        if MAX_ROWS and len(rows) >= MAX_ROWS:
            break
    print(f"[load] Parsed aspect rows: {len(rows)}")
    return pd.DataFrame(rows)

def build_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]])

def main():
    df = load_data(CSV_PATH)
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
        batch_size=32,
        num_iterations=5,
        num_epochs=1,
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
