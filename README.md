# Financial News Sentiment Analysis

- [BEST DATASET](https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news)

## Overview
This project performs sentiment analysis on financial news to infer market sentiment for individual tickers.  
It progresses from basic text preprocessing and TF-IDF classification to advanced transformer-based, time-aware, and multimodal approaches.

## TODO 
- Create Pipeline
  i. start with ingesting the data then push into step 1
  1. [NER](https://encord.com/blog/named-entity-recognition/) Portion: use prototyping/bert-base-ner.py file as an example with the [RoBerta model](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english) for org extraction
  2. entity - AKA  - specific term - mapping - (Jacob 1/2) 
  3. utilities to modify text in the required format for model, its different per model.
     2.1 must be able to find and replace extracted subject
     2.2 must be able to append to or infront of provided text, optionally with above function.
  4. [ABSA](https://www.gautamnaik.com/blog/aspect-based-sentiment-analyisis) Portion: plug-in formatted text and get formatted output currently using [FinABSA](https://github.com/guijinSON/FinABSA) (barrett responsible for this, will float around) 
  5. Finalization / cleanup (managing pipeline, making sure we have validation and testing 
  7. optionally add live data grabber and text processor to run through system. 


# Sentalyzer ABSA Quickstart

Simple steps to train, evaluate, and run inference with the SentFin ABSA pipelines.

## Setup
- Python 3.10+ recommended
- Install deps: `pip install -r requirements.txt`
- Ensure data is present (e.g., `sentfin.csv` or split CSVs under `./data/...`).

## Train models
- **SVM ABSA** (aspects concatenated with `[ASPECT]`):
  - Run: `python -m sentalyzer.testing.train_svm_absa`
  - Artifacts: `models/svm_sentfin/*` (vectorizer, classifier, config).

- **SetFit ABSA** (plain text, optional GPU):
  - Run: `python setFitTrain.py`
  - Artifacts: `setfit-absa-sentfin/`
  - GPU auto-selected if available.

## Inference
- Use `pipeline-inference.py` for end-to-end NER + ABSA:
  - Configure paths near the top (e.g., `DATA_CSV`, `SVM_MODEL_DIR`, `MAX_TEXT_ROWS`).
  - Run: `python pipeline-inference.py`
  - Outputs: predictions for NER-extracted aspects; if labels exist in the CSV, prints accuracy/report/confusion matrix.

## Evaluation-only (SVM)
- Use the test script to score a trained SVM on SentFin:
  - `python -m sentalyzer.testing.test_svm_absa_sentfin`
  - Uses the same split seed as training (test_size=0.1, random_state=42).

## Notes
- Data loader `load_sentfin_absa_samples` explodes each SentFin row into one `ABSASample` per aspect/label, so every sample has exactly one aspect and one label.
- Default aspect combiner matches SetFit (`[ASPECT]` marker); SVM and SetFit both accept custom combiners if needed.


> [!NOTE]
> Advanced “complexity layers” can improve performance and realism. Add only as needed.

---

### Optional Complexity Layers

#### Level 1 — Contextual Embeddings
Replace TF-IDF with pretrained sentence embeddings.

**Practical**
- [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html) (`all-MiniLM-L6-v2`)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

**Educational**
- [vector space](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/?utm)
- *Reimers & Gurevych (2019).* “Sentence-BERT: Sentence Embeddings using Siamese BERT Networks.” *EMNLP.*
- *Devlin et al. (2019).* “BERT: Pre-training of Deep Bidirectional Transformers.” *NAACL.*

---

#### Level 2 — Sentence-Level Aggregation
Analyze sentiment at the sentence level, then aggregate.

**Practical**
- [Hierarchical Attention Networks paper](https://www.aclweb.org/anthology/N16-1174/)
- [Simpler resource](https://medium.com/analytics-vidhya/hierarchical-attention-networks-d220318cf87e?utm)

**Educational**
- *Yang et al. (2016).* “Hierarchical Attention Networks for Document Classification.” *NAACL.*
- *Bahdanau et al. (2015).* “Neural Machine Translation by Jointly Learning to Align and Translate.” *ICLR.*

---

#### Level 3 — Fine-Tuned Transformer
Fine-tune BERT or similar transformer models for domain-specific sentiment.

**Practical**
- [HuggingFace Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Text classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)

**Educational**
- [simple resource](https://jalammar.github.io/illustrated-transformer/?utm)
- *Howard & Ruder (2018).* “Universal Language Model Fine-tuning for Text Classification (ULMFiT).” *ACL.*
- *Sun et al. (2019).* “How to Fine-Tune BERT for Text Classification.” *arXiv:1905.05583.*

---

#### Level 4 — Temporal Dynamics
Account for how sentiment effects decay over time.

**Practical**
- [Exponential decay weighting in Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
- [LSTM overview](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [temporal graph learning in 2024](https://medium.com/data-science/temporal-graph-learning-in-2024-feaa9371b8e2)

**Educational**
- *Hochreiter & Schmidhuber (1997).* “Long Short-Term Memory.” *Neural Computation.*
- *Zhang et al. (2020).* “Time-Series Momentum with Deep Learning.” *SSRN.*

---

#### Level 5 — Multimodal Fusion
Combine textual sentiment with numerical market data.

**Practical**
- [Scikit-learn ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Research: Multimodal fusion in finance](https://arxiv.org/abs/2012.01630)

**Educational**
- *Xu et al. (2022).* “Multimodal Learning for Financial Forecasting.” *Expert Systems with Applications.*
- *Baltrušaitis et al. (2019).* “Multimodal Machine Learning: A Survey and Taxonomy.” *IEEE TPAMI.*

---
