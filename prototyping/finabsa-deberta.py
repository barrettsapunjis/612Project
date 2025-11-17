import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "amphora/FinABSA-DeBERTa"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify(text, aspect):
    text_tgt = text.replace(aspect, "[TGT]")
    inp = f"{text_tgt}"
    print(f"\n #### INPUT #### \n {inp} \n")
    enc = tokenizer(inp, return_tensors="pt", truncation=True)
    logits = model(**enc).logits
    pred_id = torch.argmax(logits).item()
    labels = ["negative", "neutral", "positive"]
    return labels[pred_id]

correct = 0
total = 0

with open("converted_clean_edited.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        text = row["text"]
        aspect = row["span"]
        true = row["label"].lower()

        pred = classify(text, aspect)

        total += 1
        if pred == true:
            correct += 1

        print(
            f"id={row['id']} | true={true} | pred={pred} | acc={correct}/{total} = {correct/total:.4f}"
        )

print("\nFINAL ACCURACY:", correct/total)
