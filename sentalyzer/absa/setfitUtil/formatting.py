from typing import Literal

from ...extraction.types import AspectCandidate, Span
from ...data.samples import concat_aspect


FormatScheme = Literal["marker_suffix", "marker_inline", "sep_token"]


def _insert_markers(text: str, span: Span, left_marker: str, right_marker: str) -> str:
    """
    Insert markers around the aspect span. Assumes span is char offsets in `text`.
    """
    start, end = span
    if start < 0 or end > len(text) or start >= end:
        # Fallback: don't try fancy stuff if span is invalid
        return text
    return text[:start] + left_marker + text[start:end] + right_marker + text[end:]


def format_for_setfit_absa(
    candidate: AspectCandidate,
    scheme: FormatScheme = "marker_suffix",
    sep_token: str = "[SEP]",
) -> str:
    """
    Convert a single AspectCandidate into a SetFit-ABSA friendly text input.

    Schemes:
    - marker_suffix:    "<sentence> [ASPECT] <aspect>"
    - marker_inline:    "<sentence with [ASP]...[/ASP] around the aspect>"
    - sep_token:        "<sentence> [SEP] <aspect>"  (often used in ABSA papers)
    """
    text = candidate.text
    aspect = candidate.aspect

    if scheme == "marker_suffix":
        # Reuse shared concat_aspect helper for the default SetFit pattern.
        return concat_aspect(text, aspect, marker="[ASPECT]")

    if scheme == "sep_token":
        # Same helper, but with a configurable separator token.
        return concat_aspect(text, aspect, marker=sep_token)

    if scheme == "marker_inline" and candidate.span is not None:
        # Span-aware inline markers when we have span information.
        marked = _insert_markers(text, candidate.span, "[ASP]", "[/ASP]")
        return marked

    # Fallback if span missing / invalid scheme: use simple suffix via shared helper.
    return concat_aspect(text, aspect, marker="[ASPECT]")

