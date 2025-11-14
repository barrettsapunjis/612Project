# setfit_absa.py

import json
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer

print("starting!")
# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------
BACKBONE = "sentence-transformers/all-mpnet-base-v2"
MAX_LEN = 256

label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
id2label = {v: k for k, v in label2id.items()}

# ------------------------------------------------------------
# 2. LOAD TRAIN DATA
# ------------------------------------------------------------
# format example:
# [
#   {"aspect": "Nvidia", "sentence": "Nvidia stock plunged today.", "label": "negative"},
#   {"aspect": "Apple", "sentence": "Apple reported strong sales.", "label": "positive"},
# ]
with open("data/setfit/test.json", "r") as f:
    raw = json.load(f)

texts = [f"{x['aspect']} [SEP] {x['sentence']}" for x in raw]
labels = [label2id[x["label"]] for x in raw]

train_ds = Dataset.from_dict({"text": texts, "label": labels})

# ------------------------------------------------------------
# 3. LOAD MODEL
# ------------------------------------------------------------
model = SetFitModel.from_pretrained(BACKBONE)

# ------------------------------------------------------------
# 4. TRAINING
# ------------------------------------------------------------
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=None,
    num_iterations=20,
    num_epochs=4,
    batch_size=16,
    learning_rate=2e-5,
)

trainer.train()

# ------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------
model.save_pretrained("./setfit_absa_model")

# ------------------------------------------------------------
# 6. PREDICT FUNCTION
# ------------------------------------------------------------
def predict(aspect, sentence):
    x = f"{aspect} [SEP] {sentence}"
    out = model.predict([x])[0]
    return id2label[out]

# ------------------------------------------------------------
# 7. QUICK TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    print(predict("Nvidia", "Nvidia crushed earnings and raised guidance."))
    print(predict("Tesla", "Analysts say Tesla demand is weakening."))


