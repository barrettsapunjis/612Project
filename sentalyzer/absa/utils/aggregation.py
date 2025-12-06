"""
Entity-level sentiment aggregation utilities.

This module provides functions to aggregate sentiment predictions across multiple
texts for the same entity, producing a single sentiment score per entity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sentalyzer.data.samples import ABSASample


@dataclass
class EntitySentiment:
    """
    Aggregated sentiment score for an entity across multiple predictions.
    """
    entity: str
    dominant_sentiment: str  # Most common prediction
    sentiment_distribution: Dict[str, int]  # Count of each sentiment
    average_scores: Dict[str, float]  # Average probability/score per sentiment
    total_mentions: int  # Number of texts mentioning this entity
    confidence: float  # Overall confidence (average of max scores)


def aggregate_by_entity(
    samples: Sequence[ABSASample],
    predictions: Sequence[str],
    scores: Optional[Sequence[Dict[str, float]]] = None,
    aggregation_method: str = "majority_vote",
) -> Dict[str, EntitySentiment]:
    """
    Aggregate sentiment predictions by entity/aspect.

    Groups all predictions for the same entity and produces a single aggregated
    sentiment score using the specified method.

    Args:
        samples: ABSASample objects with text and aspect information
        predictions: List of predicted sentiment labels (aligned with samples)
        scores: Optional list of score dictionaries (label -> probability/score)
        aggregation_method: Method for aggregation:
            - "majority_vote": Most common prediction (default)
            - "weighted_average": Weight by confidence scores
            - "average_scores": Average probability scores per label

    Returns:
        Dictionary mapping entity name -> EntitySentiment object

    Example:
        >>> samples = [
        ...     ABSASample(text="Nvidia stock rises", aspect="Nvidia", label=None),
        ...     ABSASample(text="Nvidia GPUs are great", aspect="Nvidia", label=None),
        ... ]
        >>> predictions = ["positive", "positive"]
        >>> scores = [{"positive": 0.9, "neutral": 0.1}, {"positive": 0.8, "neutral": 0.2}]
        >>> results = aggregate_by_entity(samples, predictions, scores)
        >>> results["Nvidia"].dominant_sentiment
        'positive'
        >>> results["Nvidia"].total_mentions
        2
    """
    samples = list(samples)
    predictions = list(predictions)
    scores = list(scores) if scores else [{}] * len(samples)

    if len(samples) != len(predictions):
        raise ValueError(f"Samples ({len(samples)}) and predictions ({len(predictions)}) must have same length")
    if scores and len(scores) != len(samples):
        raise ValueError(f"Scores ({len(scores)}) must have same length as samples ({len(samples)})")

    # Group by entity
    entity_data: Dict[str, List[tuple]] = defaultdict(list)
    for sample, pred, score_dict in zip(samples, predictions, scores):
        entity = sample.aspect.strip()
        if entity:  # Only include samples with non-empty aspects
            entity_data[entity].append((pred, score_dict))

    # Aggregate per entity
    results: Dict[str, EntitySentiment] = {}
    for entity, data_list in entity_data.items():
        preds = [d[0] for d in data_list]
        score_dicts = [d[1] for d in data_list]

        # Sentiment distribution (counts)
        sentiment_dist = Counter(preds)

        # Dominant sentiment based on aggregation method
        if aggregation_method == "majority_vote":
            dominant = sentiment_dist.most_common(1)[0][0]
        elif aggregation_method == "weighted_average":
            # Weight by confidence (max score)
            weighted_votes: Dict[str, float] = defaultdict(float)
            for pred, score_dict in zip(preds, score_dicts):
                if score_dict:
                    confidence = max(score_dict.values())
                    weighted_votes[pred] += confidence
            dominant = max(weighted_votes.items(), key=lambda x: x[1])[0] if weighted_votes else sentiment_dist.most_common(1)[0][0]
        elif aggregation_method == "average_scores":
            # Use average probability scores
            label_scores: Dict[str, List[float]] = defaultdict(list)
            for score_dict in score_dicts:
                for label, score in score_dict.items():
                    label_scores[label].append(score)
            avg_scores = {label: sum(scores) / len(scores) for label, scores in label_scores.items()}
            dominant = max(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else sentiment_dist.most_common(1)[0][0]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Average scores per label
        all_labels = set()
        for score_dict in score_dicts:
            all_labels.update(score_dict.keys())

        average_scores: Dict[str, float] = {}
        for label in all_labels:
            label_scores_list = [sd.get(label, 0.0) for sd in score_dicts if sd]
            if label_scores_list:
                average_scores[label] = sum(label_scores_list) / len(label_scores_list)

        # Overall confidence (average of max scores)
        confidences = [max(sd.values()) if sd else 0.0 for sd in score_dicts]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        results[entity] = EntitySentiment(
            entity=entity,
            dominant_sentiment=dominant,
            sentiment_distribution=dict(sentiment_dist),
            average_scores=average_scores,
            total_mentions=len(data_list),
            confidence=avg_confidence,
        )

    return results


def aggregate_by_entity_simple(
    samples: Sequence[ABSASample],
    predictions: Sequence[str],
) -> Dict[str, Dict[str, any]]:
    """
    Simplified aggregation that returns a dictionary format.

    Args:
        samples: ABSASample objects
        predictions: Predicted sentiment labels

    Returns:
        Dictionary mapping entity -> {
            "sentiment": dominant sentiment,
            "count": number of mentions,
            "distribution": {sentiment: count}
        }
    """
    results = aggregate_by_entity(samples, predictions, aggregation_method="majority_vote")
    
    return {
        entity: {
            "sentiment": es.dominant_sentiment,
            "count": es.total_mentions,
            "distribution": es.sentiment_distribution,
            "confidence": es.confidence,
        }
        for entity, es in results.items()
    }

