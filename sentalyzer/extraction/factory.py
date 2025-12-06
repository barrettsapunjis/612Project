# sentalyzer/extraction/factory.py
import os
from .hf_ner import HFNERExtractor
from .base import BaseExtractor

DEFAULT_NER_MODEL = "Jean-Baptiste/roberta-large-ner-english"
# A lighter, faster alternative NER model; you can swap this for another HF
# checkpoint if you prefer.
FAST_NER_MODEL = "dslim/bert-base-NER"


def _default_device() -> int:
    """
    Decide a good default device for NER:
      - If SENTALYZER_NER_DEVICE is set, honor it (e.g., 0 for cuda:0, -1 for CPU).
      - Otherwise, auto-detect: use GPU (0) if available, else CPU (-1).
    """
    env_val = os.getenv("SENTALYZER_NER_DEVICE")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass

    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        # If torch isn't installed or anything goes wrong, fall back to CPU.
        return -1


def build_default_extractor(device: int | None = None) -> BaseExtractor:
    """
    Default extractor: jean-baptiste large NER.
    Swap this factory to change the default globally.
    """
    if device is None:
        device = _default_device()
    return HFNERExtractor(model_name=DEFAULT_NER_MODEL, device=device)


def build_fast_extractor(device: int | None = None) -> BaseExtractor:
    """
    Faster but slightly smaller NER model, intended for higher throughput.
    """
    if device is None:
        device = _default_device()
    return HFNERExtractor(model_name=FAST_NER_MODEL, device=device)
