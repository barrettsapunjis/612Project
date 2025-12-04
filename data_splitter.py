#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(
    csv_path: str,
    seed: int,
    test_size: float = 0.9,
    shuffle: bool = True,
) -> None:
    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=shuffle,
        stratify=None,  # change if you want stratification
    )

    size = int(test_size*10)
    out_dir = f"data_{seed}_{10-size}-{size}"
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"saved: {train_path}")
    print(f"saved: {test_path}")
    print(f"train: {len(train_df)} rows  test: {len(test_df)} rows")


if __name__ == "__main__":
    # hardcode or replace with argparse
    # split_csv(
    #     csv_path="./data/sentfin.csv",
    #     seed=42,
    #     test_size=0.9,
    # )

    # split_csv(
    #     csv_path="./data/sentfin.csv",
    #     seed=42,
    #     test_size=0.2,
    # )

    split_csv(
        csv_path="./data/sentfin.csv",
        seed=42,
        test_size=0.75,
    )
