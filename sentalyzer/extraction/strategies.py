# sentalyzer/extraction/strategies.py
from typing import Iterable, List, Optional, Sequence
from .types import AspectCandidate


def filter_candidates(
    candidates: Iterable[AspectCandidate],
    labels: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
) -> List[AspectCandidate]:
    """
    Generic filter for aspect candidates based on NER metadata.
    - labels: keep only specific NER entity labels (e.g., ["ORG"])
    - min_score: filter on NER model confidence
    """
    labels_set = set(labels) if labels else None
    out: List[AspectCandidate] = []
    for c in candidates:
        ner_label = c.ner_label
        ner_score = c.ner_score or 0.0

        if labels_set is not None and ner_label not in labels_set:
            continue
        if ner_score < min_score:
            continue
        out.append(c)
    return out


def all_aspect_candidates(
    text: str,
    candidates: Iterable[AspectCandidate],
    labels: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
) -> List[AspectCandidate]:
    """
    Modular general function to extract all aspect candidates for a line of text.
    Filters candidates by NER label and confidence score.

    Note: The `text` parameter is kept for API compatibility, but candidates
    should already have their `text` field set correctly from the extractor.
    """
    return filter_candidates(candidates, labels=labels, min_score=min_score)


def targeted_aspect_candidates(
    text: str,
    candidates: Iterable[AspectCandidate],
    targets: Sequence[str],
    labels: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
    match_case: bool = False,
    exact_match: bool = False,
) -> List[AspectCandidate]:
    """
    Targeted extraction: keep only candidates whose aspect string matches any of the
    provided targets (after optional lowercasing).

    - targets: list of potential target strings (e.g., tickers, company names)
    - labels / min_score: filter by NER label and confidence score
    - match_case: if False, compare in lowercase
    - exact_match: if True, require exact string equality; otherwise allow substring match

    Note: The `text` parameter is kept for API compatibility, but candidates
    should already have their `text` field set correctly from the extractor.
    """
    if not targets:
        # Degenerates to "all" behavior if nothing was specified.
        return all_aspect_candidates(
            text=text,
            candidates=candidates,
            labels=labels,
            min_score=min_score,
        )

    filtered = filter_candidates(candidates, labels=labels, min_score=min_score)

    if not match_case:
        norm = lambda s: s.lower()
        targets_norm = {norm(t) for t in targets}
    else:
        norm = lambda s: s
        targets_norm = set(targets)

    selected: List[AspectCandidate] = []
    for c in filtered:
        surface = norm(c.aspect)
        if exact_match:
            if surface in targets_norm:
                selected.append(c)
        else:
            if any(t in surface for t in targets_norm):
                selected.append(c)

    return selected
