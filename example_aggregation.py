"""
Example script demonstrating entity-level sentiment aggregation.

This shows how to aggregate sentiment predictions across multiple texts
for the same entity to get a single sentiment score per entity.
"""

from sentalyzer import (
    ABSASample,
    aggregate_by_entity,
    aggregate_by_entity_simple,
    EntitySentiment,
)

# Example: Multiple texts mentioning "Nvidia"
samples = [
    ABSASample(text="Nvidia stock rises on strong earnings", aspect="Nvidia", label=None),
    ABSASample(text="Nvidia GPUs are in high demand", aspect="Nvidia", label=None),
    ABSASample(text="Nvidia faces supply chain challenges", aspect="Nvidia", label=None),
    ABSASample(text="Apple announces new product", aspect="Apple", label=None),
    ABSASample(text="Apple stock performs well", aspect="Apple", label=None),
]

# Example predictions (from your model)
predictions = [
    "positive",  # Nvidia - strong earnings
    "positive",  # Nvidia - high demand
    "negative",  # Nvidia - supply chain issues
    "neutral",   # Apple - announcement
    "positive",  # Apple - performs well
]

# Example scores (probabilities from your model)
scores = [
    {"positive": 0.9, "neutral": 0.1, "negative": 0.0},  # Nvidia
    {"positive": 0.8, "neutral": 0.15, "negative": 0.05},  # Nvidia
    {"positive": 0.2, "neutral": 0.3, "negative": 0.5},  # Nvidia
    {"positive": 0.3, "neutral": 0.6, "negative": 0.1},  # Apple
    {"positive": 0.7, "neutral": 0.25, "negative": 0.05},  # Apple
]

# Aggregate by entity
results = aggregate_by_entity(
    samples=samples,
    predictions=predictions,
    scores=scores,
    aggregation_method="majority_vote",  # or "weighted_average", "average_scores"
)

# Print results
print("Entity-Level Sentiment Aggregation")
print("=" * 60)
for entity, sentiment in results.items():
    print(f"\nEntity: {entity}")
    print(f"  Dominant Sentiment: {sentiment.dominant_sentiment}")
    print(f"  Total Mentions: {sentiment.total_mentions}")
    print(f"  Distribution: {sentiment.sentiment_distribution}")
    print(f"  Average Scores: {sentiment.average_scores}")
    print(f"  Confidence: {sentiment.confidence:.3f}")

# Or use the simplified version
print("\n" + "=" * 60)
print("Simplified Format:")
print("=" * 60)
simple_results = aggregate_by_entity_simple(samples, predictions)
for entity, data in simple_results.items():
    print(f"{entity}: {data['sentiment']} (from {data['count']} mentions, confidence: {data['confidence']:.3f})")

