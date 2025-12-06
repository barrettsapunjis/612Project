# sentalyzer/extraction/types.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Span = Tuple[int, int]  # (start_char, end_char)


@dataclass
class AspectCandidate:
    """
    A single (text, aspect) pair, ready for ABSA.

    This is the canonical format that ALL ABSA models consume. It can also
    carry NER metadata (label, score, source) in the `meta` dict when
    created directly from an extractor.

    Fields:
    - text: The full input text/document
    - aspect: The extracted entity/aspect string
    - span: Optional character span (start, end) of the aspect in text
    - label: Optional sentiment label (only in training data)
    - meta: Optional metadata dict. When from NER, typically contains:
        - "ner_label": NER entity type (e.g., "ORG", "PER")
        - "ner_score": NER model confidence [0, 1]
        - "ner_source": NER model identifier
    """
    text: str
    aspect: str
    span: Optional[Span] = None
    label: Optional[Any] = None  # sentiment label (only in training data)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ner_label(self) -> Optional[str]:
        """Convenience accessor for NER label from meta."""
        return self.meta.get("ner_label")

    @property
    def ner_score(self) -> Optional[float]:
        """Convenience accessor for NER confidence score from meta."""
        return self.meta.get("ner_score")

    @property
    def ner_source(self) -> Optional[str]:
        """Convenience accessor for NER model source from meta."""
        return self.meta.get("ner_source")
