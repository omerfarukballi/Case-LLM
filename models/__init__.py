"""Data models for the Podcast Knowledge Graph System."""

from .entities import (
    Entity,
    TranscriptSegment,
    Episode,
    EntityType,
    Sentiment,
    QueryResult,
    VerificationResult
)

from .graph_schema import (
    NodeSchema,
    RelationshipSchema,
    GraphQuery,
    CypherResult
)

__all__ = [
    "Entity",
    "TranscriptSegment", 
    "Episode",
    "EntityType",
    "Sentiment",
    "QueryResult",
    "VerificationResult",
    "NodeSchema",
    "RelationshipSchema",
    "GraphQuery",
    "CypherResult"
]
