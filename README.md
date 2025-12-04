# Financial News Sentiment Analysis Using Multi-Entity ABSA  
*A complete real-time and historical sentiment pipeline for financial news.*

---

##  Overview

This project implements a full **Multi-Entity Aspect-Based Sentiment Analysis (ABSA)** pipeline designed to extract **company-specific sentiment** from financial news articles.

Unlike simple article-level sentiment models, this system identifies **multiple companies mentioned in the same article** and computes **individual sentiment scores per ticker** — enabling more detailed, actionable intelligence for financial analysis.

The system supports both:

- **Historical analysis** (CSV datasets)
- **Real-time analysis** (Yahoo Finance news)

---

## Core Design

### **Named Entity Recognition (NER) + Ticker Mapping**
- Detect company names from unstructured text  
- Normalize and map them to stock tickers (AAPL, MSFT, TSLA)

### **Aspect-Based Sentiment Analysis (ABSA)**
- Use a financial-domain ABSA model to compute **sentiment for each company** in context  
- Results per company:
  - `positive`
  - `negative`
  - `neutral`

###  **Combined Multi-Entity Sentiment Analysis Pipeline**
Detect companies → Map tickers → Run ABSA → Produce ticker-level sentiment.



# Features

### 1. Company Extraction via NER  
Uses `Jean-Baptiste/roberta-large-ner-english` to extract financial entities.

### 2. Company to Ticker Mapping  
Maps entity names to stock tickers using a curated dictionary.

### 3. Aspect-Based Sentiment Analysis (ABSA)  
Uses `amphora/FinABSA` to compute sentiment **specific to each company** mentioned.

### 4. Real-Time News Fetching  
Uses `yfinance.Search` to pull fresh financial news and `newspaper3k` to extract full text.

### 5. Multi-Entity Sentiment Output  
Each article may map to multiple companies and multiple sentiment results.

### 6. Sentiment Aggregation  
Computes summarized sentiment scores per ticker


### 7. A Fully Working Pipelines  
- Real-time pipeline (`absa_realtime_demo.ipynb`)


# Architecture Diagram

       ┌─────────────┐
       │  News Input  │  ← Real-time Yahoo Finance
       └──────┬──────┘
              │
      ┌───────▼────────┐
      │ NERExtractor    │ 
      │ Extract Companies│
      └───────┬────────┘
              │
    ┌─────────▼──────────┐
    │ Ticker Mapping      │
    │ Company → Stock ID  │
    └─────────┬──────────┘
              │
     ┌────────▼─────────┐
     │ ABSA Model        │ 
     │ Sentiment per Co. │
     └────────┬─────────┘
              │
 ┌────────────▼─────────────┐
 │ Multi-Entity ABSA Output │
 └────────────┬─────────────┘
              │
       ┌──────▼─────┐
       │ Aggregation │
       └─────────────┘



## Installation

### 1 Clone the project
git clone
cd into the directory

### Install dependencies 
!pip install pandas numpy requests newspaper3k transformers torch yfinance tqdm scikit-learn (Google Colab)
pip install -r requirements.txt (local)


# Running the Real-Time Pipeline
## For Jupyter Notebook
Open absa_realtime_demo.ipynb


## Performs:

1. Fetch recent articles

2. Extract entities

3. Map entities → tickers

4. Run ABSA

5. Display results

6. Compute final aggregated sentiment


# Models Used 

## NER Model: Jean-Baptiste/roberta-large-ner-english

## ABSA Model: amphora/FinABSA

Both are loaded through HuggingFace Transformers.



# Example Output

## Per-Article Sentiment (raw)
Article: Apple rises as Tesla drops on demand worries…

Apple (AAPL): Positive
Microsoft (MSFT): Positive
Tesla (TSLA): Negative

## Aggregated Output
Ticker: AAPL   Score: +0.667
Ticker: MSFT   Score: +0.500
Ticker: TSLA   Score: -0.333



# Core Modules Summary

## absa_model.py
Loads and runs the financial ABSA model.

## ner_extractor.py
Extracts companies using NER.

## ticker_mapping.py
Maps extracted names to stock tickers.

## multi_entity_absa.py
Combines NER + mapping + ABSA into a single pipeline.

## aggregate_sentiment.py
Converts sentiment labels into numerical scores.

## yahoo_scraper.py
Retrieves real-time financial news.

## full_pipeline.py
Complete end-to-end command-line execution pipeline.


