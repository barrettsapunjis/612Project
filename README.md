# Financial News Sentiment Analysis

## Minimum Requirement

### 1. Data Acquisition
- Pull from [Yahoo Finance API](https://pypi.org/project/yahoo-finance/) or [yfinance library](https://pypi.org/project/yfinance/).
- [Finnhub API docs](https://finnhub.io/docs/api)
- [Kaggle financial news datasets](https://www.kaggle.com/datasets?search=financial+news)
- Ensure labeled sentiment (positive/neutral/negative).

### 2. Preprocessing
- [NLTK](https://www.nltk.org/) for tokenization, stopword removal, and lemmatization.
- [spaCy](https://spacy.io/) for efficient text processing.
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) for HTML cleanup.

### 3. Feature Extraction
- [Scikit-learn TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Concept reference: [TF-IDF explained](https://monkeylearn.com/blog/what-is-tf-idf/)

### 4. Model Training
- [Scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- Evaluation metrics:
  - [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
  - [Confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

### 5. Output
- Aggregate sentiment per ticker: weighted average of article predictions.
- Reference: [Weighted averages in Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.aggregate.html)

> [!NOTE]
> Extra Complexity layers can be added from below.

---

## Optional Complexity Layers

### Level 1 — Contextual Embeddings
Replace TF-IDF with transformer-based embeddings.
- [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html) (`all-MiniLM-L6-v2`)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

### Level 2 — Sentence-Level Aggregation
Add intra-article granularity.
- [Attention mechanisms overview](https://towardsdatascience.com/attention-in-neural-networks-4317c3bdb473)
- Example: [Hierarchical Attention Networks](https://www.aclweb.org/anthology/N16-1174/)

### Level 3 — Fine-Tuned Transformer
Fine-tune a pre-trained model (e.g., BERT) on your labeled dataset.
- [HuggingFace Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Text classification with BERT](https://huggingface.co/docs/transformers/tasks/sequence_classification)

### Level 4 — Temporal Dynamics
Incorporate time weighting.
- [Exponential decay weighting](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
- [LSTM model guide](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### Level 5 — Multimodal Fusion
Fuse textual sentiment with market metrics.
- [Scikit-learn ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Research paper: Multimodal fusion in finance](https://arxiv.org/abs/2012.01630)
























