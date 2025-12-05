# sentalyzer/extraction/hf_ner.py
from typing import Iterable, List
from transformers import pipeline
from .types import AspectCandidate
from .base import BaseExtractor


class HFNERExtractor(BaseExtractor):
    """
    HuggingFace NER pipeline wrapper, tuned for jean-baptiste/roberta-large-ner-english.
    Can be reused for other HF NER models by changing model_name and entity mapping.
    """

    def __init__(self, model_name: str, device: int = -1, aggregation_strategy: str = "simple"):
        self.model_name = model_name
        self.pipe = pipeline("ner", model=model_name, aggregation_strategy=aggregation_strategy, device=device)

    def _to_aspect_candidates(self, text: str, raw) -> List[AspectCandidate]:
        candidates: List[AspectCandidate] = []
        for r in raw:
            # HF output example:
            # {"entity_group": "ORG", "score": 0.998, "word": "Nvidia", "start": 0, "end": 6}
            candidates.append(
                AspectCandidate(
                    text=text,
                    aspect=r["word"],
                    span=(int(r["start"]), int(r["end"])),
                    label=None,  # No sentiment label at extraction time
                    meta={
                        "ner_label": r.get("entity_group") or r.get("entity"),
                        "ner_score": float(r["score"]),
                        "ner_source": self.model_name,
                        **{k: v for k, v in r.items() if k not in {"word", "start", "end", "score", "entity", "entity_group"}},
                    },
                )
            )
        return candidates

    def extract(self, text: str) -> List[AspectCandidate]:
        if not text:
            return []

        raw = self.pipe(text)
        return self._to_aspect_candidates(text, raw)

    def extract_many(self, texts: Iterable[str]) -> List[List[AspectCandidate]]:
        texts = list(texts)
        if not texts:
            return []

        raw_batches = self.pipe(texts)
        # `pipeline("ner")` with aggregation_strategy="simple" returns a list
        # of lists, one per input text.
        return [self._to_aspect_candidates(text, raw) for text, raw in zip(texts, raw_batches)]
