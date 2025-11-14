# train_absa.py

from datasets import Dataset
from setfit import AbsaModel, AbsaTrainer

#https://huggingface.co/blog/setfit-absa

# ---------------------------
# 1. Build ABSA training data
# ---------------------------
# Format required:
# {
#   "text": "...",
#   "spans": ["aspect1", "aspect2", ...],
#   "polarities": ["positive", "negative", ...]
# }

raw = [
  {
    "text": "Apple beat expectations but Intel disappointed in the latest earnings call.",
    "span": "Apple",
    "label": "positive",
    "ordinal": 0
  },
  {
    "text": "Apple beat expectations but Intel disappointed in the latest earnings call.",
    "span": "Intel",
    "label": "negative",
    "ordinal": 0
  },
  {
    "text": "Nvidia posted stable results while Tesla faced production delays.",
    "span": "Nvidia",
    "label": "neutral",
    "ordinal": 0
  },
    {
    "text": "Nvidia posted stable results while Tesla faced production delays.",
    "span": "Tesla",
    "label": "negative",
    "ordinal": 0
  },
  {
    "text": "Amazon and Microsoft both reported strong cloud revenue growth.",
    "span": "Amazon",
    "label": "positive",
    "ordinal": 0
  },
  {
    "text": "Amazon and Microsoft both reported strong cloud revenue growth.",
    "span": "Microsoft",
    "label": "positive",
    "ordinal": 0
  },
  {
    "text": "Google and Meta saw mixed advertising demand during the holiday season.",
    "span": "Google",
    "label": "neutral",
    "ordinal": 0
  },
  {
    "text": "Google and Meta saw mixed advertising demand during the holiday season.",
    "span": "Meta",
    "label": "neutral",
    "ordinal": 0
  },
  {
    "text": "Tesla delivered strong vehicle numbers but Ford struggled with supply shortages.",
    "span": "Tesla",
    "label": "positive",
    "ordinal": 0
  },
  {
    "text": "Tesla delivered strong vehicle numbers but Ford struggled with supply shortages.",
    "span": "Ford",
    "label": "negative",
    "ordinal": 0
  },
  {
    "text": "Apple and Amazon remained flat as investors waited for macroeconomic updates.",
    "span": "Apple",
    "label": "neutral",
    "ordinal": 0
  },
  {
    "text": "Apple and Amazon remained flat as investors waited for macroeconomic updates.",
    "span": "Amazon",
    "label": "neutral",
    "ordinal": 0
  },
  {
    "text": "Intel and AMD competed aggressively, but AMD showed slightly better momentum.",
    "span": "Intel",
    "label": "negative",
    "ordinal": 0
  },
  {
    "text": "Intel and AMD competed aggressively, but AMD showed slightly better momentum.",
    "span": "AMD",
    "label": "positive",
    "ordinal": 0
  },
  {
    "text": "Nvidia gained traction in AI markets while Qualcomm remained steady.",
    "span": "Nvidia",
    "label": "positive",
    "ordinal": 0
  },
  {
    "text": "Nvidia gained traction in AI markets while Qualcomm remained steady.",
    "span": "Qualcomm",
    "label": "neutral",
    "ordinal": 0
  },
  {
    "text": "Microsoft announced new enterprise contracts but Google Cloud slowed slightly.",
    "span": "Microsoft",
    "label": "positive",
    "ordinal": 0
  },
  {
    "text": "Microsoft announced new enterprise contracts but Google Cloud slowed slightly.",
    "span": "Google",
    "label": "negative",
    "ordinal": 0
  },
  {
    "text": "Meta faced regulatory pressure while Apple avoided major legal issues this quarter.",
    "span": "Meta",
    "label": "negative",
    "ordinal": 0
  },
  {
    "text": "Meta faced regulatory pressure while Apple avoided major legal issues this quarter.",
    "span": "Apple",
    "label": "positive",
    "ordinal": 0
  }
]


ds = Dataset.from_list(raw)

# ---------------------------
# 2. Load ABSA model
# ---------------------------
model = AbsaModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2"
)

# ---------------------------
# 3. Train
# ---------------------------
trainer = AbsaTrainer(
    model=model,
    train_dataset=ds
)

trainer.train()

# ---------------------------
# 4. Save model
# ---------------------------
model.save_pretrained(
    "./absa_aspect_model",
    "./absa_polarity_model"
)
