import os
import pandas as pd
import requests
from pathlib import Path

BASE = Path("data/raw")
BASE.mkdir(parents=True, exist_ok=True)

URL_TRAIN = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_train.csv"
URL_VALID = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_valid.csv"

PATH_TRAIN = BASE / "sent_train.csv"
PATH_VALID = BASE / "sent_valid.csv"


def ensure_file(path: Path, url: str):
    if not path.exists():
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)


def get_huggingface_train():
    ensure_file(PATH_TRAIN, URL_TRAIN)
    df = pd.read_csv(PATH_TRAIN)
    print("\nTrain head:")
    print(df.head())
    print("Train shape:", df.shape)
    print(df["text"][1])
    return df


def get_huggingface_test():
    ensure_file(PATH_VALID, URL_VALID)
    df = pd.read_csv(PATH_VALID)
    print("\nTest head:")
    print(df.head())
    print("Test shape:", df.shape)
    return df


get_huggingface_test()
get_huggingface_train()
