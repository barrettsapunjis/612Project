from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class ABSAModel:
    """Wrapper around a pretrained Aspect-Based Sentiment model (e.g., FinABSA)."""

    def __init__(self, model_name: str = "amphora/FinABSA"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    @torch.inference_mode()
    def predict(self, text: str, aspect: str):
        if not text or not aspect:
            return {"label": "neutral", "raw": ""}

        prompt = f"aspect: {aspect} sentence: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        label = self._normalize_label(decoded)
        return {"label": label, "raw": decoded}

    @staticmethod
    def _normalize_label(output_text: str) -> str:
        t = output_text.lower()
        if "positive" in t:
            return "positive"
        if "negative" in t:
            return "negative"
        if "neutral" in t:
            return "neutral"
        return "neutral"
