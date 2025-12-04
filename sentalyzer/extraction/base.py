# sentalyzer/extraction/base.py
from abc import ABC, abstractmethod
from typing import Iterable, List
from .types import EntityMention


class BaseExtractor(ABC):
    """
    Contract for any entity extractor you plug into the system.
    """

    @abstractmethod
    def extract(self, text: str) -> List[EntityMention]:
        """
        Extract entities from a single line of text.
        """
        raise NotImplementedError

    def extract_many(self, texts: Iterable[str]) -> List[List[EntityMention]]:
        """
        Optional batched extraction. Default implementation just loops over
        `extract`; concrete extractors can override with a more efficient
        vectorized call (e.g., HuggingFace pipelines).
        """
        return [self.extract(t) for t in texts]
