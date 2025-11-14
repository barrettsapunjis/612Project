from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import time
from my_utils.time_tracking import baseTracker as bt

toiken = AutoTokenizer.from_pretrained("yangheng/deberta-v3-large-absa-v1.1")
classifier = pipeline("text-classification", model="yangheng/deberta-v3-large-absa-v1.1", tokenizer=toiken)

examples = ["apple developer beats estimates and intel in the first quarter graphics card usage lowers",
            "Apple saw china iPhone units fall in november credit suisse says",
            "Apple closing all china stores and offices",
            "ive owned the Apple card for 3 months this is why it sucks",
            "munster doubles down says Apple has 40 upside this year",
            "rosenblatt rojects  downside In Apple Shares, Warns Of Drop In iPhone Production",]

targets = ["ios, intel, mcdonalds, gatorade, gold"]

results = classifier(examples, text_pair="Apple")
print(results)

results = classifier(examples, text_pair="intel")
print(f"results 2: {results}")


