from __future__ import annotations
# sentalyzer/absa/setfit_prep.py
from typing import Iterable
import pandas as pd
from ...extraction.types import AspectCandidate
from .formatting import format_for_setfit_absa


from dataclasses import dataclass
from typing import Iterable, List, Sequence

from setfit import SetFitModel

from ...data.samples import ABSASample, AspectCombineFn, default_setfit_combiner, validate_predictions



def aspects_to_setfit_df(
    aspects: Iterable[AspectCandidate],
    scheme: str = "marker_suffix",
) -> pd.DataFrame:
    """
    Convert AspectCandidates into the exact columns expected by your SetFit-ABSA training script.
    """
    rows = []
    for ac in aspects:
        rows.append(
            {
                "text": format_for_setfit_absa(ac, scheme=scheme),
                "label": ac.label,  # your ordinal / categorical label
                "aspect": ac.aspect,
            }
        )
    return pd.DataFrame(rows)

@dataclass
class SetFitABSAModel:
    """
    Wrapper around a SetFitModel for ABSA-style inputs.

    The way we turn (text, aspect) into the final model input is controlled
    by an AspectCombineFn, defaulting to the SetFit "[ASPECT]" scheme.
    """

    model_dir: str
    model: SetFitModel
    combine_fn: AspectCombineFn = default_setfit_combiner

    @classmethod
    def from_dir(
        cls,
        model_dir: str,
        combine_fn: AspectCombineFn = default_setfit_combiner,
    ) -> "SetFitABSAModel":
        model = SetFitModel.from_pretrained(model_dir)
        return cls(model_dir=model_dir, model=model, combine_fn=combine_fn)

    def _combine_texts(self, samples: Iterable[ABSASample]) -> List[str]:
        return [self.combine_fn(s.text, s.aspect) for s in samples]

    def predict_samples(self, samples: Iterable[ABSASample]) -> List[str]:
        samples = list(samples)
        texts = self._combine_texts(samples)
        print("\nCombined texts in predict_sample ex: ", texts[0])

        preds = self.model.predict(texts)
        return list(preds)

    def predict_labels(self, texts: Sequence[str], aspects: Sequence[str]) -> List[str]:
        assert len(texts) == len(aspects)
        samples = [ABSASample(text=t, aspect=a, label=None) for t, a in zip(texts, aspects)]
        return self.predict_samples(samples)

    def evaluate(self, samples: Iterable[ABSASample]) -> None:
        samples = list(samples)
        preds = self.predict_samples(samples)
        validate_predictions(samples, preds)