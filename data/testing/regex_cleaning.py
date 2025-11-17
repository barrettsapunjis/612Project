import json
import re

INPUT = "converted.jsonl"
OUTPUT = "converted_clean.jsonl"

def clean_text(t):
    t = re.sub(r"…\s*https?://\S+", "", t)
    t = re.sub(r"https?://\S+", "", t)
    t = t.replace("Ã‚Â", "")
    t = t.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def fix_span(span, text):
    if span and span.strip():
        return span.strip()

    # leading tickers $XYZ
    m = re.search(r"\$[A-Za-z]{1,5}", text)
    if m:
        return m.group().lstrip("$")

    institutions = [
        "Fed",
        "NY Fed",
        "New York Fed",
        "St. Louis Fed",
        "IMF",
        "ECB",
        "Treasury",
        "Bank of England",
        "Bank of Japan"
    ]
    lower = text.lower()
    for inst in institutions:
        if inst.lower() in lower:
            return inst

    return ""

label_map = {0: "negative", 1: "positive", 2: "neutral"}

def fix_label(label):
    return label_map.get(label, "neutral")

def process():
    with open(INPUT, "r", encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            obj["text"] = clean_text(obj["text"])
            obj["span"] = fix_span(obj.get("span", ""), obj["text"])
            obj["label"] = fix_label(obj.get("label", 2))

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

process()
