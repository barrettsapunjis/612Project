from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional

import yfinance as yf
from newspaper import Article


@dataclass
class NewsArticle:
    id: str
    ticker: str
    title: str
    summary: str
    link: str
    source: str
    published: Optional[datetime]
    text: str


def _parse_published_time(ts) -> Optional[datetime]:
    try:
        if ts is None:
            return None
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except Exception:
        return None


def _extract_full_text(url: str) -> str:
    if not url:
        return ""
    try:
        a = Article(url)
        a.download()
        a.parse()
        text = a.text or ""
        return text.strip()
    except Exception:
        return ""


def fetch_news_for_ticker(
    ticker: str,
    max_articles: int = 20,
    fetch_full_text: bool = True,
) -> List[Dict]:
    ticker = ticker.upper()
    search_query = ticker

    search = yf.Search(search_query, news_count=max_articles)
    news_items = search.news or []

    results: List[Dict] = []

    for i, item in enumerate(news_items):
        title = item.get("title", "").strip()
        link = item.get("link", "").strip()
        publisher = item.get("publisher", "").strip()
        provider_time = item.get("providerPublishTime")

        published_dt = _parse_published_time(provider_time)
        summary = item.get("summary", "").strip() if "summary" in item else ""

        body = ""
        if fetch_full_text and link:
            body = _extract_full_text(link)

        if body:
            clean_text = body
        else:
            parts = [p for p in [title, summary] if p]
            clean_text = "\n".join(parts)

        article_id = item.get("uuid") or f"{ticker}-news-{i}"

        results.append(
            {
                "id": article_id,
                "ticker": ticker,
                "title": title,
                "summary": summary,
                "link": link,
                "source": publisher,
                "published": published_dt.isoformat() if published_dt else None,
                "body": body,
                "clean_text": clean_text,
            }
        )

    return results


if __name__ == "__main__":
    test_ticker = "AAPL"
    arts = fetch_news_for_ticker(test_ticker, max_articles=3, fetch_full_text=False)
    print(f"Fetched {len(arts)} articles for {test_ticker}")
    for art in arts:
        print("-" * 40)
        print("Title:", art["title"])
        print("Source:", art["source"])
        print("Published:", art["published"])
        print("Snippet:", (art["clean_text"] or "")[:150].replace("\n", " "))
