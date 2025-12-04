# sentalyzer/extraction/strategies.py
from typing import Iterable, List, Optional, Sequence
from .types import EntityMention, AspectCandidate


def filter_entities(
    entities: Iterable[EntityMention],
    labels: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
) -> List[EntityMention]:
    """
    Generic filter for entities.
    - labels: keep only specific entity labels (e.g., ["ORG"])
    - min_score: filter on model confidence
    """
    labels_set = set(labels) if labels else None
    out: List[EntityMention] = []
    for e in entities:
        if labels_set is not None and e.label not in labels_set:
            continue
        if e.score < min_score:
            continue
        out.append(e)
    return out


def to_aspect_candidates(
    text: str,
    entities: Iterable[EntityMention],
    attach_span: bool = True,
) -> List[AspectCandidate]:
    """
    Convert EntityMention list into AspectCandidates (no labels yet).
    This is the canonical extraction→ABSA bridge.
    """
    candidates: List[AspectCandidate] = []
    for e in entities:
        candidates.append(
            AspectCandidate(
                text=text,
                aspect=e.text,
                span=e.span if attach_span else None,
                label=None,
                meta={"ner_label": e.label, "ner_score": e.score, "ner_source": e.source},
            )
        )
    return candidates


def all_aspect_candidates(
    text: str,
    entities: Iterable[EntityMention],
    labels: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
) -> List[AspectCandidate]:
    """
    Modular general function to extract all aspect candidates for a line of text.
    This matches your "modular general function to extract given a line of text and an input".
    """
    filtered = filter_entities(entities, labels=labels, min_score=min_score)
    return to_aspect_candidates(text, filtered)


def targeted_aspect_candidates(
    text: str,
    entities: Iterable[EntityMention],
    targets: Sequence[str],
    labels: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
    match_case: bool = False,
    exact_match: bool = False,
) -> List[AspectCandidate]:
    """
    Targeted extraction: keep only entities whose surface form matches any of the
    provided targets (after optional lowercasing).

    - targets: list of potential target strings (e.g., tickers, company names)
    - labels / min_score: same filtering knobs as `all_aspect_candidates`
    - match_case: if False, compare in lowercase
    - exact_match: if True, require exact string equality; otherwise allow substring match
    """
    if not targets:
        # Degenerates to "all" behavior if nothing was specified.
        return all_aspect_candidates(
            text=text,
            entities=entities,
            labels=labels,
            min_score=min_score,
        )

    filtered = filter_entities(entities, labels=labels, min_score=min_score)

    if not match_case:
        norm = lambda s: s.lower()
        targets_norm = {norm(t) for t in targets}
    else:
        norm = lambda s: s
        targets_norm = set(targets)

    selected: List[EntityMention] = []
    for e in filtered:
        surface = norm(e.text)
        if exact_match:
            if surface in targets_norm:
                selected.append(e)
        else:
            if any(t in surface for t in targets_norm):
                selected.append(e)

    return to_aspect_candidates(text, selected)
