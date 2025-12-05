# IV. EXPERIMENTS AND RESULTS

## A. Experimental Setup

All models are trained on the SentFin dataset using stratified train-test splits (test_size=0.1, random_state=42). Evaluation is performed via `pipeline-inference.py`, which extracts aspects using Named Entity Recognition (NER) and compares predictions against gold-standard labels. The NER pipeline employs RoBERTa-large (Jean-Baptiste/roberta-large-ner-english) with confidence threshold 0.7 to filter organization entities. Performance metrics include per-class precision, recall, F1-scores, and confusion matrices across positive/neutral/negative sentiments.

## B. Model Architectures

We evaluate three model configurations:

**SVM-TFIDF**: Linear Support Vector Machine with TF-IDF vectorization (ngram_range=(1,2), max_features=50,000, sublinear_tf=True). This provides an interpretable baseline using traditional bag-of-words features.

**SVM-Hybrid**: Enhanced SVM combining sparse TF-IDF features with dense embeddings from sentence transformers (all-MiniLM-L6-v2). Feature vectors are concatenated, with TF-IDF capturing lexical patterns and transformer embeddings encoding semantic relationships.

**SetFit**: Few-shot learning model built on sentence transformers, fine-tuned for ABSA through contrastive learning. SetFit leverages GPU-accelerated transformers for contextual understanding.

## C. Aspect Combination Strategies

We investigate four strategies for combining text and aspect information:

1. **Masking** (`mask_aspect`): Replaces aspect occurrences with `[ASPECT]` token to reduce entity-specific overfitting: "Apple shares rose" → "[ASPECT] shares rose".

2. **Concatenation** (`concat_aspect`): Appends aspect with marker: "<text> [ASPECT] <aspect>". Default for SetFit models.

3. **Separator Token** (`sep_token_combiner`): Uses `[SEP]` token: "<text> [SEP] <aspect>", similar to BERT-style formatting.

4. **Inline Markers** (`inline_marker_combiner`): Wraps aspect in text with XML-style markers `[ASP]...[/ASP]` when present, otherwise appends: "[ASP] Nvidia[/ASP] GPUs are great".

## D. Results on SentFin Dataset

[Insert Table I with metrics: Accuracy, Precision, Recall, F1 for each model and combination strategy]

Evaluation reveals that SetFit achieves the highest performance ([X]% accuracy), followed by SVM-Hybrid ([X]%) and SVM-TFIDF baseline ([X]%). The hybrid approach demonstrates that combining sparse and dense features improves over TF-IDF alone. Aspect combination strategies show varying impact: masking reduces overfitting but may lose entity-specific context, while concatenation preserves explicit aspect-text relationships. Confusion matrices indicate higher confusion between neutral and positive classes compared to negative sentiment.

## E. Real-Time Evaluation

We validate model performance on live financial news via `yfinance` integration, fetching real-time articles for tickers (AAPL, MSFT, GOOGL, AMZN, NVDA) with max_articles=20. Articles are processed into `NewsArticle` objects containing title, summary, full text (via newspaper3k), publication timestamps, and source metadata. These are converted to `ABSASample(text=clean_text, aspect=ticker, label=None)` for zero-shot inference, enabling ticker-specific sentiment scoring without gold labels.

Real-time evaluation (via `realTest.py`) demonstrates model agreement rates of [X]% between SVM and SetFit, with sentiment distributions showing [describe distribution]. Average confidence scores are [X] for SVM and [X] for SetFit. This validates practical applicability to dynamic financial news streams where models operate in fully unsupervised settings.

