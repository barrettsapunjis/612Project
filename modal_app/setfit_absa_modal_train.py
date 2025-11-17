import modal
import json
from datasets import Dataset
from setfit import AbsaModel, AbsaTrainer

app = modal.App("setfit-absa")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "datasets",
        "setfit",
        "sentence-transformers",
        "torch",
        "scikit-learn",
        "spacy",
        "spacy-transformers"
    )
    .run_commands("python -m spacy download en_core_web_lg")
)

# Attach named volume "absa" at /data inside the container
vol = modal.Volume.from_name("absa", create_if_missing=True)

@app.function(
    image=image,
    timeout=7200,
    gpu="A10G",                 # <= THIS MAKES GPU HAPPEN
    volumes={"/data": vol},
)
def train():
    # -----------------------------------------
    # 1. Load RAW TRAINING DATA EXACTLY AS IS
    # -----------------------------------------
    raw = []
    with open("/data/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))
    
    # LIMIT SIZE HERE
    MAX_ROWS = 150     # ← choose your cap
    raw = raw[:MAX_ROWS]

    ds = Dataset.from_list(raw)


    
    # -----------------------------------------
    # 2. Load ABSA MODEL
    # -----------------------------------------
    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2"
    )

    # -----------------------------------------
    # 3. TRAIN
    # -----------------------------------------
    trainer = AbsaTrainer(
        model=model,
        train_dataset=ds,
    )
    trainer.train(max_steps=2000)

    # -----------------------------------------
    # 4. SAVE MODELS INTO THE SAME VOLUME
    # -----------------------------------------
    model.save_pretrained(
        "/data/absa_aspect_model",
        "/data/absa_polarity_model",
    )

    # Ensure model dirs are persisted
    vol.commit()

    return "done"
