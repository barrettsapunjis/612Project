# sentalyzer/extraction/api.py
from typing import List, Optional, Sequence
from .factory import build_default_extractor
from .types import AspectCandidate
from .strategies import all_aspect_candidates, targeted_aspect_candidates

# You can reuse a single extractor instance in your app to avoid reloading the model.
_EXTRACTOR = None


def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = build_default_extractor()
    return _EXTRACTOR

def extract_all_org_aspects(text: str, min_score: float = 0.0) -> List[AspectCandidate]:
    """
    Convenience helper: extract *all* ORG aspects from a line of text.
    """
    extractor = get_extractor()
    candidates = extractor.extract(text)
    return all_aspect_candidates(text, candidates, labels=["ORG"], min_score=min_score)


def extract_all_org_aspects_batch(
    texts: Sequence[str],
    min_score: float = 0.0,
) -> List[List[AspectCandidate]]:
    """
    Batched variant of `extract_all_org_aspects` for better throughput.

    Returns a list of aspect-candidate lists, aligned with the input `texts`.
    """
    if not texts:
        return []

    extractor = get_extractor()
    candidates_per_text = extractor.extract_many(texts)

    out: List[List[AspectCandidate]] = []
    for text, candidates in zip(texts, candidates_per_text):
        out.append(all_aspect_candidates(text, candidates, labels=["ORG"], min_score=min_score))
    return out


def extract_targeted_org_aspects(
    text: str,
    targets: Sequence[str],
    min_score: float = 0.0,
    match_case: bool = False,
    exact_match: bool = False,
) -> List[AspectCandidate]:
    """
    Targeted variant: apply a list of potential targets and only keep matching ORG entities.

    - targets: e.g., list of organization names or tickers you care about
    - min_score: NER confidence threshold
    - match_case / exact_match: control how strictly we match the surface form
    """
    extractor = get_extractor()
    candidates = extractor.extract(text)
    return targeted_aspect_candidates(
        text=text,
        candidates=candidates,
        targets=targets,
        labels=["ORG"],
        min_score=min_score,
        match_case=match_case,
        exact_match=exact_match,
    )
