"""
Entity data models for the Podcast Knowledge Graph System.
Defines Pydantic models for entities, transcripts, episodes, and query results.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime


class EntityType(str, Enum):
    """Types of entities that can be extracted from podcasts."""
    PERSON = "PERSON"
    BOOK = "BOOK"
    MOVIE = "MOVIE"
    MUSIC = "MUSIC"
    COMPANY = "COMPANY"
    PRODUCT = "PRODUCT"
    LOCATION = "LOCATION"
    TOPIC = "TOPIC"
    QUOTE = "QUOTE"


class Sentiment(str, Enum):
    """Sentiment classification for entity mentions."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Entity(BaseModel):
    """
    Represents an extracted entity from a podcast transcript.
    
    Attributes:
        type: The type of entity (PERSON, BOOK, etc.)
        value: The name or value of the entity
        context: Surrounding context where entity was mentioned
        timestamp: When in the video the entity was mentioned (seconds)
        sentiment: The sentiment of the mention
        speaker: Who mentioned the entity
        confidence: Extraction confidence score (0.0-1.0)
        ad_read: Whether this was part of an advertisement
        metadata: Additional entity-specific metadata
    """
    type: EntityType
    value: str
    context: str
    timestamp: float
    sentiment: Sentiment = Sentiment.NEUTRAL
    speaker: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    ad_read: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('value')
    @classmethod
    def value_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Entity value cannot be empty")
        return v.strip()
    
    def to_node_properties(self) -> Dict[str, Any]:
        """Convert entity to Neo4j node properties."""
        base_props = {
            "id": self.generate_id(),
            "name": self.value if self.type != EntityType.BOOK else None,
            "title": self.value if self.type in [EntityType.BOOK, EntityType.MOVIE, EntityType.MUSIC] else None,
        }
        base_props.update(self.metadata)
        return {k: v for k, v in base_props.items() if v is not None}
    
    def generate_id(self) -> str:
        """Generate a unique ID for this entity."""
        clean_value = self.value.lower().replace(" ", "_").replace("'", "")
        return f"{self.type.value.lower()}_{clean_value}"


class TranscriptSegment(BaseModel):
    """
    Represents a segment of a podcast transcript with timing and speaker info.
    
    Attributes:
        text: The transcribed text
        start: Start time in seconds
        end: End time in seconds
        speaker: Speaker label from diarization
        confidence: Transcription confidence score
    """
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    
    @property
    def duration(self) -> float:
        """Get the duration of this segment in seconds."""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "confidence": self.confidence
        }


class Episode(BaseModel):
    """
    Represents a podcast episode with metadata.
    
    Attributes:
        video_id: YouTube video ID
        title: Episode title
        podcast_name: Name of the podcast
        publish_date: Publication date (YYYY-MM-DD)
        duration: Episode duration in seconds
        hosts: List of host names
        guests: List of guest names
        video_url: Full YouTube URL
        description: Episode description
    """
    video_id: str
    title: str
    podcast_name: str
    publish_date: str
    duration: float = 0.0
    hosts: List[str] = Field(default_factory=list)
    guests: List[str] = Field(default_factory=list)
    video_url: Optional[str] = None
    description: Optional[str] = None
    
    def model_post_init(self, __context: Any) -> None:
        """Set video_url after initialization if not provided."""
        if self.video_url is None:
            self.video_url = f"https://www.youtube.com/watch?v={self.video_id}"
    
    @field_validator('publish_date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("publish_date must be in YYYY-MM-DD format")
    
    def to_node_properties(self) -> Dict[str, Any]:
        """Convert episode to Neo4j node properties."""
        return {
            "id": f"episode_{self.video_id}",
            "video_id": self.video_id,
            "title": self.title,
            "publish_date": self.publish_date,
            "duration": self.duration,
            "video_url": self.video_url
        }
    
    def get_timestamp_url(self, timestamp: float) -> str:
        """Get YouTube URL with timestamp."""
        return f"{self.video_url}&t={int(timestamp)}"


class QueryResult(BaseModel):
    """
    Result from a knowledge graph query.
    
    Attributes:
        query: The original query
        type: Query type (graph, semantic, hybrid, verify)
        answer: The synthesized answer
        results: Raw results from the query
        sources: Source citations with timestamps
        confidence: Overall confidence in the answer
        execution_time: Query execution time in seconds
        cypher_query: The Cypher query used (if applicable)
        verified: Whether the result was verified against the graph
    """
    query: str
    type: Literal["graph", "semantic", "hybrid", "verify"]
    answer: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    execution_time: float = 0.0
    cypher_query: Optional[str] = None
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "type": self.type,
            "answer": self.answer,
            "results": self.results,
            "sources": self.sources,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "cypher_query": self.cypher_query,
            "verified": self.verified
        }


class VerificationResult(BaseModel):
    """
    Result from a claim verification query.
    
    Attributes:
        claim: The original claim being verified
        verified: Whether the claim is verified as true
        evidence: Supporting or refuting evidence
        reason: Explanation for the verification result
        sources: Sources checked during verification
        confidence: Confidence in the verification
    """
    claim: str
    verified: bool
    evidence: List[str] = Field(default_factory=list)
    reason: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "claim": self.claim,
            "verified": self.verified,
            "evidence": self.evidence,
            "reason": self.reason,
            "sources": self.sources,
            "confidence": self.confidence
        }


class TranscriptChunk(BaseModel):
    """
    A chunk of transcript for processing and embedding.
    
    Attributes:
        text: The chunk text
        start_time: Start timestamp
        end_time: End timestamp
        speaker: Primary speaker in the chunk
        chunk_index: Index of this chunk in the episode
        video_id: Associated video ID
        podcast_name: Associated podcast name
        publish_date: Episode publish date
        topics: Extracted topics for this chunk
        has_ad: Whether this chunk contains advertisement
    """
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    chunk_index: int = 0
    video_id: str = ""
    podcast_name: str = ""
    publish_date: str = ""
    topics: List[str] = Field(default_factory=list)
    has_ad: bool = False
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dictionary for vector store."""
        return {
            "video_id": self.video_id,
            "podcast_name": self.podcast_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker or "unknown",
            "chunk_index": self.chunk_index,
            "publish_date": self.publish_date,
            "topics": ",".join(self.topics) if self.topics else "",
            "has_ad": self.has_ad
        }
