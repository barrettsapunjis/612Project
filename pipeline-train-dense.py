from sentalyzer import (
    SVMABSAConfig,
    SVMABSAModel,
    train_svm_absa,
    default_setfit_combiner,
    load_sentfin_absa_samples
)

from sentalyzer.embeddings import (
    EmbeddingConfig,
    load_embedding_model,
    encode_texts,
)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data"

DATA_CSV = DATA_PATH / "data_42_8-2" / "train.csv"
SVM_MODEL_DIR = MODEL_PATH / "svm_sentfin-dense"
MAX_ROWS = None  # set to an int to limit rows
REPORT_DIR = BASE_DIR / "reports"

def main():
    samples = load_sentfin_absa_samples(
        path=str(DATA_CSV),
        max_rows=MAX_ROWS,
    )
    if not samples:
        raise SystemExit(f"[error] No training samples loaded from {DATA_CSV}")
    
    for sample in samples: 
        embeddings = encode_texts(sample.text)
        sample.embedding = embeddings
    
    cfg = SVMABSAConfig(
        model_dir=str(SVM_MODEL_DIR),
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=1,
        C=1.0,
        class_weight="balanced",
        random_state=42,
        test_size=0.1,
        use_dense=True,
    )
    train_svm_absa(samples=samples, cfg=cfg, combine_fn=default_setfit_combiner)