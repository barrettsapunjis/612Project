# sentalyzer/data/samples.py
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class ABSASample:
    """
    Generic ABSA unit: (text, aspect, optional label).

    How these are turned into model inputs is delegated to an AspectCombineFn.
    """
    text: str
    aspect: str
    label: Optional[str] = None


# Aspect combiner type: takes raw text + aspect -> model input string
AspectCombineFn = Callable[[str, str], str]


def concat_aspect(text: str, aspect: str, marker: str = "[ASPECT]") -> str:
    """
    Simple concatenation strategy (used for SetFit ABSA by default):
      "<text> [ASPECT] <aspect>"
    """
    return text.strip() + f" {marker} " + aspect.strip()


def default_setfit_combiner(text: str, aspect: str) -> str:
    """
    Default combiner for SetFit ABSA.
    """
    return concat_aspect(text, aspect, marker="[ASPECT]")


def sep_token_combiner(sep: str = "[SEP]") -> AspectCombineFn:
    """
    Aspect combiner using a separator token between text and aspect:
      "<text> [SEP] <aspect>"
    """
    def _fn(text: str, aspect: str) -> str:
        return concat_aspect(text, aspect, marker=sep)
    return _fn


def inline_marker_combiner(
    left_marker: str = "[ASP]",
    right_marker: str = "[/ASP]",
) -> AspectCombineFn:
    """
    Aspect combiner that tries to inline the aspect into the text with markers.

    If the (case-insensitive) aspect substring is found in the text, wrap the
    first occurrence with markers:
      "Nvidia [ASP]GPUs[/ASP] are great"

    If not found, fall back to simple concatenation with markers:
      "<text> [ASP]<aspect>[/ASP]"
    """
    def _fn(text: str, aspect: str) -> str:
        t = text
        idx = t.lower().find(aspect.lower())
        if idx == -1:
            # Fallback: append marked aspect at the end.
            return t.strip() + f" {left_marker}{aspect}{right_marker}"
        end = idx + len(aspect)
        return t[:idx] + left_marker + t[idx:end] + right_marker + t[end:]
    return _fn


def _df_to_absa_samples_from_aspects_json(
    df: pd.DataFrame,
    text_col: str = "text",
    aspects_col: str = "aspects_json",
) -> List[ABSASample]:
    """
    Shared helper: convert any DataFrame with a text column and an
    'aspects_json' column into ABSASample rows.

    aspects_json is expected to be a JSON dict mapping aspect -> sentiment label.
    """
    samples: List[ABSASample] = []
    for _, row in df.iterrows():
        raw = row.get(aspects_col)
        try:
            aspects = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            continue

        for aspect, label in aspects.items():
            samples.append(
                ABSASample(
                    text=row[text_col],
                    aspect=str(aspect),
                    label=str(label).lower(),
                )
            )
    return samples


# ---------- SentFin-specific loader ----------

def load_sentfin_csv(
    path: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    sample_count: Optional[int] = None,
    keep_all_after_slice: bool = False,
) -> List[ABSASample]:
    """
    Load SentFin-style CSV into a list[ABSASample].

    CSV schema: [id, text, aspects_json, label_id]
    aspects_json: JSON dict mapping aspect -> sentiment string.
    """
    df = pd.read_csv(path, header=None, names=["id", "text", "aspects_json", "label_id"])

    if start is not None or end is not None:
        df = df.iloc[start:end]

    samples = _df_to_absa_samples_from_aspects_json(df, text_col="text", aspects_col="aspects_json")

    if not samples:
        return []

    if not keep_all_after_slice and sample_count is not None:
        samples = samples[:sample_count]

    return samples


# ---------- Printing / evaluation helpers ----------

def print_predictions(
    samples: Sequence[ABSASample],
    preds: Sequence[str],
    label: str = "sampled",
) -> None:
    print(f"[print] Showing {len(samples)} {label} predictions")
    for s, p in zip(samples, preds):
        print("\n---")
        print("Text:", s.text)
        print("Aspect:", s.aspect)
        print("Gold label:", s.label)
        print("Predicted:", p)


def validate_predictions(
    samples: Sequence[ABSASample],
    preds: Sequence[str],
) -> None:
    gold = [s.label for s in samples if s.label is not None]
    preds = list(preds)[: len(gold)]
    acc = accuracy_score(gold, preds)
    print(f"[validate] Accuracy: {acc:.4f}")
    print("[validate] Classification report:")
    print(classification_report(gold, preds, zero_division=0))
    labels = sorted(set(gold + preds))
    cm = confusion_matrix(gold, preds, labels=labels)
    print("[validate] Confusion matrix (rows=gold, cols=pred):")
    header = ["gold\\pred"] + labels
    print("\t".join(header))
    for gold_label, row in zip(labels, cm):
        print("\t".join([gold_label] + [str(x) for x in row]))
