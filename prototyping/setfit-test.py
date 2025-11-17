
import json
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from setfit import AbsaModel

# Load trained SetFit ABSA classifier
# model = SetFitModel.from_pretrained("./setfit_absa_model")

# def classify(aspect, sentences):
#     # Build SetFit inputs
#     pairs = [f"{aspect} [SEP] {s}" for s in sentences]
#     preds = model.predict(pairs)
#     return preds

# # Example sentences
# examples = [
#     "apple developer beats estimates and intel in the first quarter graphics card usage lowers",
#     "Apple saw china iPhone units fall in november credit suisse says",
#     "Apple closing all china stores and offices",
#     "ive owned the Apple card for 3 months this is why it sucks",
#     "munster doubles down says Apple has 40 upside this year",
#     "rosenblatt projects downside In Apple Shares, Warns Of Drop In iPhone Production",
#     "You can buy things at stored like Apple or"

# ]

# # Target aspects
# aspects = ["Apple", "intel"]

# for a in aspects:
#     print(f"\n=== Aspect: {a} ===")
#     preds = classify(a, examples)
#     print(preds)



print(f"\n== ABSA ====")


aspect_path = "./local_absa/aspect/absa_aspect_model"
polarity_path = "./local_absa/polarity/absa_polarity_model"

model = AbsaModel.from_pretrained(
    aspect_path,
    polarity_path,
)

text = "Apple beats expectations while Intel struggles."
spans = ["Apple", "Intel"]

preds = model.predict(text)
print(preds)