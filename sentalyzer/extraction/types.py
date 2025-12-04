# sentalyzer/extraction/types.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Span = Tuple[int, int]  # (start_char, end_char)


@dataclass
class EntityMention:
    """
    Single entity mention extracted from text, independent of any model.
    """
    text: str
    span: Span
    label: str                   # e.g. "ORG", "PER", "TICKER", etc.
    score: float                 # model confidence in [0, 1]
    source: str                  # e.g. "jean-baptiste/roberta-large-ner-english"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AspectCandidate:
    """
    A single (text, aspect) pair, ready (or almost ready) for ABSA.
    This is the canonical format that ALL ABSA models should consume from.
    """
    text: str
    aspect: str
    span: Optional[Span] = None  # optional; some models need spans, some don't
    label: Optional[Any] = None  # filled only in training data (e.g., ordinal)
    meta: Dict[str, Any] = field(default_factory=dict)
