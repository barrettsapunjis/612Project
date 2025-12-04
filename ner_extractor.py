from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class NERExtractor:
    def __init__(self, model_name: str = "Jean-Baptiste/roberta-large-ner-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )

    def extract_companies(self, text: str):
        if not text or not text.strip():
            return []

        entities = self.nlp(text)
        candidates = [
            ent["word"]
            for ent in entities
            if ent.get("entity_group", "").upper() in {"ORG", "PER", "MISC"}
        ]

        seen = set()
        unique = []
        for name in candidates:
            clean_name = name.strip()
            if clean_name and clean_name not in seen:
                seen.add(clean_name)
                unique.append(clean_name)

        return unique
