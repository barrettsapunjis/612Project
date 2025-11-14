from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import time
from my_utils.time_tracking import baseTracker as bt

model_path = "Jean-Baptiste/roberta-large-ner-english"
timers = []
print("initializing tokenizer")
tokenTimer = bt()
timers.append(tokenTimer)
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenTimer.stop()
modelTimer = bt()
print(modelTimer)
model = AutoModelForTokenClassification.from_pretrained(model_path)
print(modelTimer.stop())

print("initializing pipeline")
pipeTimer = bt()
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
pipeTimer.stop()

examples = ["ios developer beats estimates and intel in the first quarter; graphics card usage lowers",
            "mcdonalds blows up",
            "some dude died, apparently it was right after he took a sip of gatorade",
            "liquid gold is filling the rivers of asia"]


outputs = [(text, nlp(text)) for text in examples ]

i = 0


for (in_txt, out_rslt) in outputs:
    print(f"\n\n### EXAMPLE {i} ###")
    print(f"\n{in_txt}")
    print(f"\n{out_rslt}")
    i += 1
