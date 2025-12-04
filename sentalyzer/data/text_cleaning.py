from __future__ import annotations

"""
Text cleaning and normalization utilities for Sentalyzer.

These helpers are intentionally lightweight and deterministic so they can be
used both at training time (e.g., for SVM / SetFit) and at inference time
without surprises.
"""

import re
from typing import Callable, Collection, Set

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _ENGLISH_STOP_WORDS


_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_WORD_RE = re.compile(r"[^\w\s]")


def basic_normalize(text: str) -> str:
    """
    Basic, generally safe normalization:
      - Lowercase
      - Collapse multiple whitespace into a single space
      - Strip leading/trailing whitespace
    """
    if not text:
        return ""
    t = text.lower()
    t = _MULTI_WHITESPACE_RE.sub(" ", t)
    return t.strip()


def strip_urls(text: str) -> str:
    """
    Remove URLs, leaving a single space in their place.
    """
    if not text:
        return ""
    return _URL_RE.sub(" ", text)


def strip_html(text: str) -> str:
    """
    Remove simple HTML/XML tags.
    """
    if not text:
        return ""
    return _HTML_TAG_RE.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation characters, keeping letters, digits and whitespace.

    Note: This is optional and can hurt performance if punctuation carries
    useful sentiment cues (e.g. "!!!"). Use with care.
    """
    if not text:
        return ""
    return _NON_WORD_RE.sub(" ", text)


def remove_stopwords(
    text: str,
    stop_words: Collection[str] | None = None,
) -> str:
    """
    Remove stop words from the text.

    By default uses scikit-learn's built-in English stop word list.
    """
    if not text:
        return ""
    sw: Set[str] = set(stop_words) if stop_words is not None else set(_ENGLISH_STOP_WORDS)
    tokens = text.split()
    filtered = [t for t in tokens if t not in sw]
    return " ".join(filtered)


def build_default_cleaner(remove_stops: bool = True) -> Callable[[str], str]:
    """
    Create the default text cleaning function used for classical models.

    Current pipeline (in order):
      1) strip_urls
      2) strip_html
      3) basic_normalize
      4) optional stop word removal (English)

    This intentionally does NOT remove punctuation or numbers by default.
    """

    def _clean(text: str) -> str:
        t = strip_urls(text)
        t = strip_html(t)
        t = basic_normalize(t)
        if remove_stops:
            t = remove_stopwords(t)
        return t

    return _clean



