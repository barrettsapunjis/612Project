## Minimum requirement 

**1. Data acquisition**
- Pull from Yahoo Finance, Finnhub, or Kaggle financial news datasets.
- Ensure labeled sentiment (positive/neutral/negative).

**2. Preprocessing**
- Clean text (HTML removal, punctuation stripping, lowercase).
- Tokenize and lemmatize.
- Remove stopwords.

**3. Feature extraction**
- Use **TF-IDF vectorization** on entire article or averaged sentence tokens.

**4. Model training**
- Train **SVM (linear kernel)** using TF-IDF features.
- Evaluate using F1, accuracy, confusion matrix.

**5. Output**
- Aggregate sentiment per ticker: weighted average of article predictions.

> [!Note]
> Extra Complexity layers can be added from below. 

<details>
<summary>Optional Complexity Layers</summary>

**Level 1 — Contextual embeddings**  
Replace TF-IDF with transformer-based embeddings.  
- Use `sentence-transformers` (e.g., `all-MiniLM-L6-v2`).  
- Generate embeddings for sentences or entire articles.  
- Train SVM or logistic regression on those embeddings.  
- Complexity: moderate, significant performance gain.  

**Level 2 — Sentence-level aggregation**  
Add intra-article granularity.  
- Split each article into sentences.  
- Get sentence embeddings and sentiment probabilities.  
- Aggregate using attention or weighted averages.  
- Formula: `article_sentiment = f(sentence_sentiments)`.  

**Level 3 — Fine-tuned transformer**  
Fine-tune a pre-trained model (e.g., BERT) on your labeled dataset.  
- Use HuggingFace `Trainer`.  
- Add a 3-class classification head.  
- Resource intensive but highest accuracy potential.  

**Level 4 — Temporal dynamics**  
Incorporate time weighting.  
- Group articles by publication date.  
- Apply exponential decay weighting for older samples.  
- Optionally use LSTM or transformer encoders for sequence tracking.  

**Level 5 — Multimodal fusion**  
Fuse textual sentiment with market metrics.  
- Combine sentiment outputs with stock data (price change, volume, volatility).  
- Use a meta-regressor for short-term price direction prediction.  
- High complexity, research-grade scope.

</details>

