from typing import List, Dict
from collections import defaultdict

SENTIMENT_TO_SCORE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def aggregate_by_ticker(records: List[Dict]) -> Dict[str, float]:
    sums = defaultdict(float)
    counts = defaultdict(int)

    for r in records:
        ticker = r.get("ticker")
        sentiment = r.get("sentiment", "neutral")
        if not ticker:
            continue
        score = SENTIMENT_TO_SCORE.get(sentiment, 0.0)
        sums[ticker] += score
        counts[ticker] += 1

    agg = {}
    for ticker, total in sums.items():
        agg[ticker] = total / counts[ticker]

    return agg
