# sentalyzer/absa/utils/__init__.py
from .aggregation import (
    EntitySentiment,
    aggregate_by_entity,
    aggregate_by_entity_simple,
)
from .reporting import write_eval_report

__all__ = [
    "EntitySentiment",
    "aggregate_by_entity",
    "aggregate_by_entity_simple",
    "write_eval_report",
]

