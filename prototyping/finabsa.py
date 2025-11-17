import json
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

model = AutoModelForSeq2SeqLM.from_pretrained("amphora/FinABSA").to(device)
tokenizer = AutoTokenizer.from_pretrained("amphora/FinABSA")

path = "converted_clean_edited.jsonl"

def extract_sentiment(out_text):
    t = out_text.upper()
    if "POSITIVE" in t:
        return "positive"
    if "NEGATIVE" in t:
        return "negative"
    if "NEUTRAL" in t:
        return "neutral"
    return "unknown"

correct = 0
total = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        
        row = json.loads(line)

        text = row.get("text", "")
        span = row.get("span", "")
        true_label = row.get("label", "").lower()

        if not span:
            continue

        text_tgt = text.replace(span, "[TGT]")

        inp = f"{text_tgt}"
        print(f"\ninput sent {total}: {inp}")

        # TOKENIZER OUTPUT → GPU
        enc = tokenizer(inp, return_tensors="pt").to(device)

        # GENERATE ON GPU
        out = model.generate(**enc, max_length=30)

        # DECODE stays on CPU
        dec = tokenizer.decode(out[0], skip_special_tokens=True)

        pred = extract_sentiment(dec)
        print(f"\noutput sent: {dec}\n")

        total += 1
        if pred == true_label:
            correct += 1

        print(
            f"id={row.get('id')} | true={true_label} | pred={pred} "
            f"| acc={correct}/{total} = {correct/total:.4f}"
        )

print("\nFINAL ACCURACY:", correct/total)
