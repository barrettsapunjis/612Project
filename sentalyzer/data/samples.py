# sentalyzer/data/samples.py
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import json
import re
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np


@dataclass
class ABSASample:
    """
    Generic ABSA unit: (text, aspect, optional label).

    How these are turned into model inputs is delegated to an AspectCombineFn.
    """
    text: str
    aspect: str
    label: Optional[str] = None
    embedding: Optional[np.ndarray] = None


# Aspect combiner type: takes raw text + aspect -> model input string
AspectCombineFn = Callable[[str, str], str]


def mask_aspect(
    text: str,
    aspect: str,
    mask_token: str = "[ASPECT]",
) -> str:
    """
    Mask occurrences of the aspect string inside the text.

    This is useful to reduce overfitting to specific company / entity names by
    replacing them with a canonical token.

    Matching is case-insensitive and uses a simple substring replacement:
      "Apple shares rose" + aspect="Apple"  ->  "[ASPECT] shares rose"
    """
    if not text or not aspect:
        return text

    pattern = re.compile(re.escape(aspect), flags=re.IGNORECASE)
    return pattern.sub(mask_token, text)


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
      "[ASP] Nvidia[/ASP] GPUs are great"

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


# ---------- Data loading helpers ----------

def load_absa_samples_from_csv(
    path: str,
    text_col: str = "text",
    aspect_col: str = "aspect",
    label_col: str = "label",
    max_rows: Optional[int] = None,
) -> List[ABSASample]:
    """
    Load ABSASample objects from a CSV file.

    This function is flexible and handles different CSV formats:
    - If only 'text' column exists: creates samples with text only (aspect and label as None)
    - If 'text' and 'aspect' columns exist: creates samples with text and aspect (label as None)
    - If 'text', 'aspect', and 'label' columns exist: creates full samples

    Args:
        path: Path to CSV file
        text_col: Name of the text column (default: "text")
        aspect_col: Name of the aspect column (default: "aspect")
        label_col: Name of the label column (default: "label")
        max_rows: Optional limit on number of rows to load

    Returns:
        List of ABSASample objects

    Examples:
        # CSV with just text column
        samples = load_absa_samples_from_csv("texts_only.csv")
        # -> ABSASample(text="...", aspect="", label=None)

        # CSV with text and aspect
        samples = load_absa_samples_from_csv("texts_aspects.csv")
        # -> ABSASample(text="...", aspect="...", label=None)

        # CSV with text, aspect, and label
        samples = load_absa_samples_from_csv("full_data.csv")
        # -> ABSASample(text="...", aspect="...", label="...")
    """
    df = pd.read_csv(path)

    if max_rows is not None:
        df = df.head(max_rows)

    # Check which columns exist
    has_text = text_col in df.columns
    has_aspect = aspect_col in df.columns
    has_label = label_col in df.columns

    if not has_text:
        raise ValueError(f"CSV file must have a '{text_col}' column")

    samples: List[ABSASample] = []

    for _, row in df.iterrows():
        # Get text (required)
        text = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
        if not text:
            continue  # Skip rows with empty text

        # Get aspect (optional)
        if has_aspect:
            aspect_raw = row[aspect_col]
            if pd.isna(aspect_raw) or (isinstance(aspect_raw, str) and not aspect_raw.strip()):
                aspect = ""
            else:
                aspect = str(aspect_raw).strip()
        else:
            aspect = ""

        # Get label (optional)
        if has_label:
            label_raw = row[label_col]
            if pd.isna(label_raw) or (isinstance(label_raw, str) and not label_raw.strip()):
                label = None
            else:
                label = str(label_raw).strip()
        else:
            label = None

        samples.append(
            ABSASample(
                text=text,
                aspect=aspect,
                label=label,
            )
        )

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
