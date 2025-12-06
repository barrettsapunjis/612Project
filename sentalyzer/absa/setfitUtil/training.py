from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence
import os
import datetime

from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
import torch

from sentalyzer.data.samples import (
    ABSASample,
    AspectCombineFn,
    default_setfit_combiner,
)


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class SetFitABSAConfig:
    model_id: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    output_dir: str = field(default_factory=lambda: f"setfit-absa-sentfin_{_ts()}")
    batch_size: int = 32
    num_iterations: int = 5
    num_epochs: int = 1
    test_size: float | int = 0.9  # allow ratio or absolute count
    random_state: int = 42
    device: str | None = "cuda"

def samples_to_setfit_dataset(
    samples: Iterable[ABSASample],
    combine_fn: AspectCombineFn = default_setfit_combiner,
) -> Dataset:
    samples = list(samples)
    texts = [combine_fn(s.text, s.aspect) for s in samples]
    labels = [str(s.label) for s in samples]
    return Dataset.from_dict({"text": texts, "label": labels})


def split_dataset(
    ds: Dataset,
    test_size: float | int,
    seed: int,
) -> dict[str, Dataset]:
    # Stratified split in `datasets` requires ClassLabel dtype; here labels are strings.
    # Support ratio (<1.0) or absolute count (>=1) with a reproducible split.
    total = len(ds)
    if isinstance(test_size, float):
        ts_value = test_size
    else:
        ts_value = min(int(test_size), total)
        if ts_value >= total:
            ts_value = max(1, total // 5)  # fallback to 80/20-ish if oversized

    split = ds.train_test_split(test_size=ts_value, seed=seed)
    print(f"[split] test_size={ts_value} total={total} -> train={len(split['train'])}, test={len(split['test'])}")
    return split


def train_setfit_absa(
    samples: Iterable[ABSASample],
    cfg: SetFitABSAConfig,
    combine_fn: AspectCombineFn = default_setfit_combiner,
) -> dict:
    # Resolve device preference
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = samples_to_setfit_dataset(samples, combine_fn)
    split = split_dataset(ds, test_size=cfg.test_size, seed=cfg.random_state)

    model = SetFitModel.from_pretrained(
        cfg.model_id,
        multi_target_strategy=None,
    )

    trainer = SetFitTrainer(
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        batch_size=cfg.batch_size,
        num_iterations=cfg.num_iterations,
        num_epochs=cfg.num_epochs,
        column_mapping={"text": "text", "label": "label"},
    )

    trainer.train()
    metrics = trainer.evaluate()
    model.save_pretrained(cfg.output_dir)
    return metrics
