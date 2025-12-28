"""
Integration Tests for Podcast Knowledge Graph System

Tests the complete pipeline and system integration.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModels:
    """Test data models."""
    
    def test_entity_model(self):
        from models.entities import Entity, EntityType, Sentiment
        
        entity = Entity(
            type=EntityType.PERSON,
            value="John Doe",
            context="John Doe discussed AI",
            timestamp=125.5,
            sentiment=Sentiment.POSITIVE,
            speaker="Host",
            confidence=0.95
        )
        
        assert entity.type == EntityType.PERSON
        assert entity.value == "John Doe"
        assert entity.confidence == 0.95
        
        # Test ID generation
        assert "person_john_doe" in entity.generate_id()
    
    def test_episode_model(self):
        from models.entities import Episode
        
        episode = Episode(
            video_id="abc123",
            title="Test Episode",
            podcast_name="Test Podcast",
            publish_date="2025-01-15",
            duration=3600.0,
            hosts=["Host 1"],
            guests=["Guest 1"]
        )
        
        assert episode.video_id == "abc123"
        assert episode.video_url == "https://www.youtube.com/watch?v=abc123"
        assert "2025-01-15" in episode.get_timestamp_url(100)
    
    def test_episode_date_validation(self):
        from models.entities import Episode
        from pydantic import ValidationError
        
        # Valid date
        episode = Episode(
            video_id="abc123",
            title="Test",
            podcast_name="Test",
            publish_date="2025-01-15"
        )
        assert episode.publish_date == "2025-01-15"
        
        # Invalid date format
        with pytest.raises(ValidationError):
            Episode(
                video_id="abc123",
                title="Test",
                podcast_name="Test",
                publish_date="15-01-2025"  # Wrong format
            )
    
    def test_transcript_segment_model(self):
        from models.entities import TranscriptSegment
        
        segment = TranscriptSegment(
            text="Hello world",
            start=10.0,
            end=15.0,
            speaker="Speaker A"
        )
        
        assert segment.duration == 5.0
        assert segment.speaker == "Speaker A"


class TestConfig:
    """Test configuration."""
    
    def test_settings_loading(self):
        # This may fail if .env is not configured
        try:
            from config import get_settings
            settings = get_settings()
            
            assert settings is not None
            assert hasattr(settings, 'openai_api_key')
            assert hasattr(settings, 'neo4j_uri')
        except Exception:
            pytest.skip("Settings require .env configuration")
    
    def test_graph_schema(self):
        from config import GRAPH_SCHEMA, CYPHER_SCHEMA_STRING
        
        assert "nodes" in GRAPH_SCHEMA
        assert "relationships" in GRAPH_SCHEMA
        assert "Person" in GRAPH_SCHEMA["nodes"]
        assert "Episode" in GRAPH_SCHEMA["nodes"]
        
        assert "Person" in CYPHER_SCHEMA_STRING
        assert "APPEARED_ON" in CYPHER_SCHEMA_STRING


class TestGraphSchema:
    """Test graph schema definitions."""
    
    def test_node_types(self):
        from models.graph_schema import NodeType
        
        assert NodeType.PERSON.value == "Person"
        assert NodeType.BOOK.value == "Book"
        assert NodeType.EPISODE.value == "Episode"
    
    def test_relationship_types(self):
        from models.graph_schema import RelationshipType
        
        assert RelationshipType.APPEARED_ON.value == "APPEARED_ON"
        assert RelationshipType.DISCUSSED_IN.value == "DISCUSSED_IN"
    
    def test_predefined_queries(self):
        from models.graph_schema import PredefinedQueries
        
        assert "MERGE" in PredefinedQueries.MERGE_PERSON
        assert "Person" in PredefinedQueries.MERGE_PERSON
        assert "Episode" in PredefinedQueries.MERGE_EPISODE


class TestEntityExtraction:
    """Test entity extraction logic."""
    
    def test_chunk_transcript(self):
        from services.entity_extraction import EntityExtractor
        from models.entities import TranscriptSegment
        
        extractor = EntityExtractor()
        
        # Create mock segments
        segments = [
            TranscriptSegment(
                text="This is segment one with some content." * 10,
                start=0.0,
                end=30.0,
                speaker="Speaker A"
            ),
            TranscriptSegment(
                text="This is segment two with more content." * 10,
                start=30.0,
                end=60.0,
                speaker="Speaker B"
            )
        ]
        
        chunks = extractor.chunk_transcript(segments, max_tokens=100)
        
        assert len(chunks) > 0
        assert all(chunk.start_time >= 0 for chunk in chunks)
    
    def test_deduplicate_entities(self):
        from services.entity_extraction import EntityExtractor
        from models.entities import Entity, EntityType, Sentiment
        
        extractor = EntityExtractor()
        
        entities = [
            Entity(
                type=EntityType.PERSON,
                value="John Doe",
                context="Context 1",
                timestamp=10.0,
                sentiment=Sentiment.POSITIVE,
                confidence=0.9
            ),
            Entity(
                type=EntityType.PERSON,
                value="John Doe",  # Duplicate
                context="Context 2",
                timestamp=20.0,
                sentiment=Sentiment.NEUTRAL,
                confidence=0.8
            ),
            Entity(
                type=EntityType.BOOK,
                value="The Book",
                context="About the book",
                timestamp=30.0,
                sentiment=Sentiment.POSITIVE,
                confidence=0.95
            )
        ]
        
        deduplicated = extractor.deduplicate_entities(entities)
        
        # Should have 2 unique entities (1 person + 1 book)
        assert len(deduplicated) == 2
        
        # Check that person was merged with higher confidence
        person = next(e for e in deduplicated if e.type == EntityType.PERSON)
        assert person.confidence == 0.9  # Max of 0.9 and 0.8


class TestQueryEngine:
    """Test query engine logic."""
    
    @pytest.mark.asyncio
    async def test_query_type_classification(self):
        """Test that query types are handled."""
        from services.query_engine import QueryType
        
        # Verify enum values
        assert QueryType.GRAPH.value == "graph"
        assert QueryType.SEMANTIC.value == "semantic"
        assert QueryType.HYBRID.value == "hybrid"
        assert QueryType.VERIFY.value == "verify"


class TestVectorStore:
    """Test vector store operations."""
    
    def test_metadata_format(self):
        from models.entities import TranscriptChunk
        
        chunk = TranscriptChunk(
            text="Test content",
            start_time=10.0,
            end_time=20.0,
            speaker="Host",
            chunk_index=0,
            video_id="abc123",
            podcast_name="Test Podcast",
            publish_date="2025-01-15",
            topics=["AI", "Technology"],
            has_ad=False
        )
        
        metadata = chunk.to_metadata()
        
        assert metadata["video_id"] == "abc123"
        assert metadata["podcast_name"] == "Test Podcast"
        assert "AI" in metadata["topics"]
        assert metadata["has_ad"] == False


class TestHelperFunctions:
    """Test utility functions."""
    
    def test_parse_video_url(self):
        from main import parse_video_url
        
        # Standard URL
        assert parse_video_url("https://www.youtube.com/watch?v=d6EMk6dyrOU") == "d6EMk6dyrOU"
        
        # Short URL
        assert parse_video_url("https://youtu.be/d6EMk6dyrOU") == "d6EMk6dyrOU"
        
        # Just video ID
        assert parse_video_url("d6EMk6dyrOU") == "d6EMk6dyrOU"
        
        # With timestamp
        assert parse_video_url("https://www.youtube.com/watch?v=d6EMk6dyrOU&t=100") == "d6EMk6dyrOU"


class TestCacheLogic:
    """Test caching functionality."""
    
    def test_cache_key_generation(self):
        from services.transcription import TranscriptionService
        
        service = TranscriptionService()
        
        key1 = service._get_cache_key("/path/to/audio1.mp3")
        key2 = service._get_cache_key("/path/to/audio2.mp3")
        
        # Different inputs should produce different keys
        assert key1 != key2
        
        # Same input should produce same key
        assert service._get_cache_key("/path/to/audio1.mp3") == key1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
