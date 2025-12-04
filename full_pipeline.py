import sys
from typing import List, Dict

from pipeline.multi_entity_absa import MultiEntityABSA
from analysis.aggregate_sentiment import aggregate_by_ticker

try:
    from fetch_news.yahoo_scraper import fetch_news_for_ticker
    REAL_SCRAPER_AVAILABLE = True
except ImportError:
    REAL_SCRAPER_AVAILABLE = False


def fetch_articles_for_ticker_demo(ticker: str) -> List[Dict]:

    return [
        {
            "id": "demo-1",
            "text": f"""
            {ticker} rallied today after strong quarterly earnings.
            Apple Inc. and Microsoft Corporation also posted gains,
            while Tesla Inc. slipped on concerns about demand.
            """
        },
        {
            "id": "demo-2",
            "text": """
            Tesla Inc. shares fell sharply after the company cut prices again,
            raising worries about profit margins. Meanwhile, Amazon.com Inc.
            and Microsoft advanced on optimism around cloud and AI.
            """
        },
    ]


def fetch_articles_for_ticker(ticker: str) -> List[Dict]:
    if REAL_SCRAPER_AVAILABLE:
        print("[INFO] Using real Yahoo scraper.")
        articles = fetch_news_for_ticker(ticker, max_articles=10)
        wrapped = []
        for i, art in enumerate(articles):
            text = art.get("clean_text") or art.get("body") or art.get("summary") or art.get("title") or ""
            if not text:
                continue
            wrapped.append({"id": art.get("id", f"art-{i}"), "text": text})
        return wrapped
    else:
        print("[INFO] Using demo articles (no real scraper wired in yet).")
        return fetch_articles_for_ticker_demo(ticker)


def run_full_pipeline_for_ticker(ticker: str):
    print(f"\n=== FULL PIPELINE: TICKER = {ticker} ===\n")
    articles = fetch_articles_for_ticker(ticker)
    if not articles:
        print("No articles found.")
        return

    print(f"Fetched {len(articles)} articles.")
    multi_absa = MultiEntityABSA()
    all_records: List[Dict] = []

    for art in articles:
        art_id = art.get("id", "unknown")
        text = art.get("text", "").strip()
        if not text:
            continue

        print(f"\n--- Analyzing article {art_id} ---")
        print(f"Snippet: {text[:200].strip()}...\n")

        results = multi_absa.analyze_article(text)
        if not results:
            print("No mapped companies/tickers found in this article.")
            continue

        for r in results:
            record = {
                "article_id": art_id,
                "company": r["company"],
                "ticker": r["ticker"],
                "sentiment": r["sentiment"],
                "raw_output": r["raw_output"],
            }
            all_records.append(record)

            print(
                f"  -> Company: {record['company']:<25} "
                f"Ticker: {record['ticker']:<8} "
                f"Sentiment: {record['sentiment']}"
            )

    if not all_records:
        print("\nNo sentiment records produced. Check NER/ticker mapping.")
        return

    print("\n=== AGGREGATED SENTIMENT BY TICKER ===")
    agg = aggregate_by_ticker(all_records)
    for tkr, score in agg.items():
        print(f"Ticker: {tkr:<8}  Score: {score:+.3f}")

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker_arg = sys.argv[1].upper()
    else:
        ticker_arg = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    run_full_pipeline_for_ticker(ticker_arg)
