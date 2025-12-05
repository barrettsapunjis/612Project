"""
Script to build a test set by fetching news article titles from Yahoo Finance
and saving them in ABSASample format (text, aspect, label).

Edit the TICKERS list below to specify which tickers to fetch data for.
This script saves raw data without extraction - you can extract aspects later if needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from sentalyzer.realtime.yahoo_scraper import fetch_news_for_ticker

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# List of tickers to fetch news for
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "SLI"
    # Add more tickers here
]

# Maximum number of articles to fetch per ticker
MAX_ARTICLES = 50

# Output directory for CSV files
OUTPUT_DIR = "data"




def main():
    """Build test set for all configured tickers and save to a single CSV file in ABSASample format."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building test set for {len(TICKERS)} ticker(s): {', '.join(TICKERS)}")
    print(f"Configuration:")
    print(f"  Max articles per ticker: {MAX_ARTICLES}")
    print(f"  Fetching titles/headers only (no full text)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()

    all_rows: List[dict] = []

    for ticker in TICKERS:
        print(f"{'=' * 60}")
        print(f"Processing {ticker}...")
        print(f"{'=' * 60}")

        # Fetch articles (titles only, no full text)
        print(f"Fetching news titles for {ticker}...")
        articles = fetch_news_for_ticker(
            ticker=ticker,
            max_articles=MAX_ARTICLES,
            fetch_full_text=False,  # Only get titles/headers
        )

        if not articles:
            print(f"[warn] No articles fetched for {ticker}")
            continue

        print(f"Fetched {len(articles)} articles")

        # Build rows in ABSASample format (text, aspect, label)
        # Using title as text, ticker as aspect, and empty label
        for article in articles:
            # Use title as the text (or summary if title is empty)
            text = article.get("title", "").strip()
            if not text:
                text = article.get("summary", "").strip()
            
            if not text:
                continue  # Skip articles with no title or summary

            all_rows.append(
                {
                    "text": text,
                    "aspect": ticker,  # Use ticker as aspect (can extract later)
                    "label": "",  # Empty label for test set
                }
            )

        print(f"Added {len(articles)} articles from {ticker}")
        print()

    # Save all articles to a single CSV file
    if not all_rows:
        print("[error] No articles collected. Nothing to save.")
        return

    df = pd.DataFrame(all_rows)
    output_path = output_dir / "test_set.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    print(f"  Total articles: {len(df)}")
    print(f"  Tickers processed: {len(TICKERS)}")
    print(f"  Output file: {output_path}")
    print(f"  Format: ABSASample (text, aspect, label)")


if __name__ == "__main__":
    main()

