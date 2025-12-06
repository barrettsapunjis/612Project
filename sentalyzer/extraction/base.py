# sentalyzer/extraction/base.py
from abc import ABC, abstractmethod
from typing import Iterable, List
from .types import AspectCandidate


class BaseExtractor(ABC):
    """
    Contract for any entity extractor you plug into the system.

    Extractors return AspectCandidate objects directly, with NER metadata
    (label, score, source) stored in the `meta` dict.
    """

    @abstractmethod
    def extract(self, text: str) -> List[AspectCandidate]:
        """
        Extract entities from a single line of text, returning aspect candidates
        with NER metadata in the `meta` field.
        """
        raise NotImplementedError

    def extract_many(self, texts: Iterable[str]) -> List[List[AspectCandidate]]:
        """
        Optional batched extraction. Default implementation just loops over
        `extract`; concrete extractors can override with a more efficient
        vectorized call (e.g., HuggingFace pipelines).
        """
        return [self.extract(t) for t in texts]
