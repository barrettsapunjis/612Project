# train_absa_modal.py
import os
import modal
from modal import App, Image, gpu
from datasets import Dataset
from setfit import AbsaModel, AbsaTrainer
import json

app = App("absa-training")

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "setfit",
        "datasets",
        "spacy",
        "sentence-transformers"
    )
    .run_commands("python -m spacy download en_core_web_md")
)



@app.function(image=image, gpu=gpu.A10G(), volumes={"/data": modal.Volume.from_name("absa-data")})
def train():
    with open("/data/test.json", "r") as f:
        raw = json.load(f)

    train_ds = Dataset.from_list(raw)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2",
        spacy_model="en_core_web_md"
    )

    trainer = AbsaTrainer(
        model=model,
        train_dataset=train_ds
    )

    trainer.train()

    model.save_pretrained(
        "/data/setfit/absa_aspect_model",
        "/data/setfit/absa_polarity_model"
    )

    return "Training complete."
