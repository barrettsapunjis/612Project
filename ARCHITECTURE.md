# Sentalyzer ABSA Architecture (High-Level)

## Data flow (with examples)
1. **Data loading** (`sentalyzer/data/sentfin.py`):
   - Input row example (CSV):  
     ```
     S No.,Title,Decisions,Words
     669,"Ponzi schemes: Sebi seeks quarterly meetings of state panels","{""Sebi"": ""neutral""}",9
     ```
   - `load_sentfin_df("sentfin.csv")` → DataFrame with columns `id, text, aspects_json, word_count`
   - `load_sentfin_absa_samples("sentfin.csv")` → list of per-aspect `ABSASample(text, aspect, label)`  
     Example output items:  
     - `ABSASample(text="Ponzi schemes: Sebi seeks quarterly meetings of state panels", aspect="Sebi", label="neutral")`
2. **Aspect combination** (`sentalyzer/absa/samples.py`):
   - Default: `default_setfit_combiner("Company beats estimates", "Apple")` → `"Company beats estimates [ASPECT] Apple"`
3. **Model training**:
   - **SVM** (`sentalyzer/absa/SVMUtil/training.py`): TF-IDF + `LinearSVC`, saves to `models/svm_sentfin` (vectorizer, classifier, config).  
     - Run: `python -m sentalyzer.testing.train_svm_absa`
   - **SetFit** (`sentalyzer/absa/setfitUtil/training.py`): sentence-transformer fine-tuning, saves to `setfit-absa-sentfin`.  
     - Run: `python setFitTrain.py`
4. **Inference pipeline** (`pipeline-inference.py`):
   - Load texts (e.g., `data/data_42_8-2/test.csv`)
   - Run NER: `extract_all_org_aspects(text)` to propose ORG aspects
   - Build unlabeled `ABSASample`s
   - Score with ABSA (SVM by default; SetFit optional)
   - If labels exist, run `validate_predictions` (accuracy/report/confusion matrix).

## Components (with usage hints)
- **Loaders**: `sentalyzer/data/sentfin.py`
  - `load_sentfin_df(path, max_rows=None)` – raw DataFrame
  - `load_sentfin_absa_samples(path, ...)` – exploded labeled samples

- **Samples / combiners**: `sentalyzer/absa/samples.py`
  - `ABSASample(text, aspect, label=None)`
  - `default_setfit_combiner(text, aspect)` → `"<text> [ASPECT] <aspect>"`
  - `validate_predictions(samples, preds)` – accuracy/report/confusion matrix

- **SVM utilities**:
  - Train: `sentalyzer/absa/SVMUtil/training.py` (`train_svm_absa`)
  - Infer: `sentalyzer/absa/SVMUtil/svm_absa_model.py` (`SVMABSAModel.from_dir(...).predict_samples(...)`)

- **SetFit utilities**:
  - Train: `sentalyzer/absa/setfitUtil/training.py` (`train_setfit_absa`)
  - Infer: `sentalyzer/absa/setfitUtil/prep.py` (`SetFitABSAModel.from_dir(...).predict_samples(...)`)

- **Testing scripts**:
  - Train SVM: `python -m sentalyzer.testing.train_svm_absa`
  - Eval SVM: `python -m sentalyzer.testing.test_svm_absa_sentfin` (uses deterministic split)

- **End-to-end inference**: `pipeline-inference.py`
  - NER + ABSA scoring + optional labeled validation; toggle model loader to switch SVM/SetFit.

## Extensibility
- Swap combiners to change how text/aspect are joined (e.g., sep-token or inline markers).
- Swap ABSA backend by changing the model loader in the pipeline.
- Replace NER (`extract_all_org_aspects`) with another aspect source if needed.
