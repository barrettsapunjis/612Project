# Financial News Sentiment Analysis

IMPORTANT: to run prototype files, you must be in the parent directory and use an example command such as `python -m prototyping.bert-base-ner` (windows)

- [BEST DATASET](https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news)

## Overview
This project performs sentiment analysis on financial news to infer market sentiment for individual tickers.  
It progresses from basic text preprocessing and TF-IDF classification to advanced transformer-based, time-aware, and multimodal approaches.

---
---
SEE BELOW FOR INITIAL README / OUTLINE
---
---

## Minimum Requirement

### 1. Data Acquisition
Collect labeled financial text data for model training and validation.

**Practical References**
- [Yahoo Finance API](https://pypi.org/project/yahoo-finance/)
- [Finnhub API](https://finnhub.io/docs/api)
- [Kaggle financial news datasets](https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news) <-- this is the best one! has aspect and sentiment lables! 
- [hugging face dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/viewer/default/train?p=4&views%5B%5D=train)

**Educational References**
- *Loughran, T., & McDonald, B. (2011).* "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10‐Ks." *Journal of Finance.*
- *Tetlock, P. (2007).* "Giving content to investor sentiment: The role of media in the stock market." *Journal of Finance.*

---

### 2. Preprocessing
Prepare text for feature extraction by cleaning, normalizing, and tokenizing.

**Practical References**
- [NLTK](https://www.nltk.org/) — tokenization, stopword removal, lemmatization  
- [spaCy](https://spacy.io/) — efficient NLP pipeline  
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) — HTML parsing

**Educational References**
- *Jurafsky & Martin (2023).* *Speech and Language Processing (3rd ed.)* — Chapters on text normalization and tokenization.  
- *Cambria et al. (2017).* “Affective Computing and Sentiment Analysis.” *IEEE Intelligent Systems.*

---

### 3. Feature Extraction
Convert text into numerical form using TF-IDF or embeddings.

**Practical References**
- [Scikit-learn TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [TF-IDF/vector space](https://mbrenndoerfer.com/writing/vector-space-model-tfidf-information-retrieval-semantic-search-history?utm)


**Educational References**
- *Salton & McGill (1983).* *Introduction to Modern Information Retrieval.* — Foundation of TF-IDF weighting.
- *Manning, Raghavan, & Schütze (2008).* *Introduction to Information Retrieval.* — Vector space models.

---

### 4. Model Training
Train a classifier (typically linear SVM) on TF-IDF or embeddings.

**Practical References**
- [Scikit-learn SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- Evaluation metrics:
  - [F1 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
  - [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

**Educational References**
- *Cortes & Vapnik (1995).* "Support-vector networks." *Machine Learning.*  
- *Goodfellow, Bengio, & Courville (2016).* *Deep Learning.* — Chapter on linear classifiers and loss functions.

---

### 5. Output
Aggregate sentiment scores per ticker using weighted averages.

**Practical References**
- [Weighted aggregation in Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.aggregate.html)

**Educational References**
- *Nassirtoussi et al. (2014).* “Text mining for market prediction: A systematic review.” *Expert Systems with Applications.*
- *Bollen et al. (2011).* “Twitter mood predicts the stock market.” *Journal of Computational Science.*

---

> [!NOTE]
> Advanced “complexity layers” can improve performance and realism. Add only as needed.

---

## Optional Complexity Layers

### Level 1 — Contextual Embeddings
Replace TF-IDF with pretrained sentence embeddings.

**Practical**
- [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html) (`all-MiniLM-L6-v2`)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

**Educational**
- [vector space](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/?utm)
- *Reimers & Gurevych (2019).* “Sentence-BERT: Sentence Embeddings using Siamese BERT Networks.” *EMNLP.*
- *Devlin et al. (2019).* “BERT: Pre-training of Deep Bidirectional Transformers.” *NAACL.*

---

### Level 2 — Sentence-Level Aggregation
Analyze sentiment at the sentence level, then aggregate.

**Practical**
- [Hierarchical Attention Networks paper](https://www.aclweb.org/anthology/N16-1174/)
- [Simpler resource](https://medium.com/analytics-vidhya/hierarchical-attention-networks-d220318cf87e?utm)

**Educational**
- *Yang et al. (2016).* “Hierarchical Attention Networks for Document Classification.” *NAACL.*
- *Bahdanau et al. (2015).* “Neural Machine Translation by Jointly Learning to Align and Translate.” *ICLR.*

---

### Level 3 — Fine-Tuned Transformer
Fine-tune BERT or similar transformer models for domain-specific sentiment.

**Practical**
- [HuggingFace Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Text classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)

**Educational**
- [simple resource](https://jalammar.github.io/illustrated-transformer/?utm)
- *Howard & Ruder (2018).* “Universal Language Model Fine-tuning for Text Classification (ULMFiT).” *ACL.*
- *Sun et al. (2019).* “How to Fine-Tune BERT for Text Classification.” *arXiv:1905.05583.*

---

### Level 4 — Temporal Dynamics
Account for how sentiment effects decay over time.

**Practical**
- [Exponential decay weighting in Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
- [LSTM overview](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [temporal graph learning in 2024](https://medium.com/data-science/temporal-graph-learning-in-2024-feaa9371b8e2)

**Educational**
- *Hochreiter & Schmidhuber (1997).* “Long Short-Term Memory.” *Neural Computation.*
- *Zhang et al. (2020).* “Time-Series Momentum with Deep Learning.” *SSRN.*

---

### Level 5 — Multimodal Fusion
Combine textual sentiment with numerical market data.

**Practical**
- [Scikit-learn ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Research: Multimodal fusion in finance](https://arxiv.org/abs/2012.01630)

**Educational**
- *Xu et al. (2022).* “Multimodal Learning for Financial Forecasting.” *Expert Systems with Applications.*
- *Baltrušaitis et al. (2019).* “Multimodal Machine Learning: A Survey and Taxonomy.” *IEEE TPAMI.*

---
