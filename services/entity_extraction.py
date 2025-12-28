"""
Entity Extraction Service for the Podcast Knowledge Graph System.
Uses GPT-4 to extract structured entities from podcast transcripts.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

from openai import AsyncOpenAI
import tiktoken

from models.entities import (
    Entity, 
    EntityType, 
    Sentiment, 
    TranscriptSegment, 
    Episode,
    TranscriptChunk
)
from config import get_settings, get_logger

logger = get_logger(__name__)


# Entity extraction prompt template
ENTITY_EXTRACTION_PROMPT = """You are an expert entity extractor for podcast transcripts. Your task is to extract ALL meaningful entities from this transcript chunk accurately.

For each entity, provide:
1. type: One of PERSON, BOOK, MOVIE, MUSIC, COMPANY, PRODUCT, LOCATION, TOPIC, QUOTE
2. value: The exact name/value of the entity
3. context: 1-2 sentences of surrounding context explaining the mention
4. timestamp: Approximate timestamp from the chunk (use the provided start time)
5. sentiment: positive, negative, or neutral - how is this entity discussed?
6. speaker: Who mentioned this entity (if identifiable)
7. confidence: 0.0-1.0 score for extraction confidence
8. ad_read: true if this is part of a sponsor/advertisement segment

CRITICAL RULES:
- For BOOKS: Extract title AND author if mentioned. Format as "Title by Author" or just "Title" if author unknown.
- For MOVIES: Include director if mentioned. Format as "Title (Director: Name)" or just "Title".
- For MUSIC: Include artist. Format as "Song by Artist" or "Album by Artist".
- For QUOTES: Must be actual quotes someone said. Include the speaker in the context.
- For TOPICS: Use broad semantic categories like: AI, Politics, Business, Health, Philosophy, Technology, Finance, Science, Sports, Entertainment, History.
- Disambiguate carefully: "Dune" could be a book OR movie - use context clues (e.g., "reading" = book, "theater/IMAX" = movie).
- Mark ad_read: true ONLY if the segment contains clear advertising language like "brought to you by", "use code", "sponsored by".
- Do NOT extract generic words or common phrases as entities.
- Do NOT hallucinate - only extract what is explicitly mentioned.
- For COMPANY mentions discussing stocks, add "stock_discussed: true" to metadata.

Context about this podcast:
- Podcast: {podcast_name}
- Hosts: {hosts}
- Guests: {guests}
- Episode Date: {date}

Transcript chunk (timestamp {start}s - {end}s):
---
{transcript}
---

Return ONLY a valid JSON array of entities. If no entities found, return [].

Example output format:
[
  {{
    "type": "PERSON",
    "value": "Elon Musk",
    "context": "They discussed Elon Musk's approach to running Twitter.",
    "timestamp": 125.5,
    "sentiment": "neutral",
    "speaker": "Host Name",
    "confidence": 0.95,
    "ad_read": false,
    "metadata": {{}}
  }},
  {{
    "type": "BOOK",
    "value": "The Almanack of Naval Ravikant by Eric Jorgenson",
    "context": "David recommended The Almanack of Naval Ravikant as essential reading for entrepreneurs.",
    "timestamp": 130.0,
    "sentiment": "positive",
    "speaker": "David Senra",
    "confidence": 0.92,
    "ad_read": false,
    "metadata": {{"recommended": true, "author": "Eric Jorgenson"}}
  }}
]"""


class EntityExtractor:
    """
    Service for extracting entities from podcast transcripts using GPT-4.
    
    Features:
    - Chunked processing for long transcripts
    - Structured JSON output parsing
    - Entity deduplication and merging
    - Ad read detection and filtering
    - Cross-reference detection
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_transcript(
        self,
        segments: List[TranscriptSegment],
        max_tokens: int = None
    ) -> List[TranscriptChunk]:
        """
        Split transcript into chunks for processing.
        
        Args:
            segments: List of transcript segments
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of TranscriptChunk objects
        """
        max_tokens = max_tokens or self.settings.max_tokens_per_chunk
        chunks = []
        current_text = []
        current_start = 0
        current_tokens = 0
        current_speaker = None
        chunk_index = 0
        
        for i, segment in enumerate(segments):
            segment_tokens = self.count_tokens(segment.text)
            
            # Start new chunk if would exceed limit
            if current_tokens + segment_tokens > max_tokens and current_text:
                chunk = TranscriptChunk(
                    text=" ".join(current_text),
                    start_time=current_start,
                    end_time=segments[i-1].end if i > 0 else segment.start,
                    speaker=current_speaker,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                current_text = []
                current_start = segment.start
                current_tokens = 0
            
            # Add segment to current chunk
            if not current_text:
                current_start = segment.start
                current_speaker = segment.speaker
            
            current_text.append(segment.text)
            current_tokens += segment_tokens
        
        # Add final chunk
        if current_text:
            chunk = TranscriptChunk(
                text=" ".join(current_text),
                start_time=current_start,
                end_time=segments[-1].end,
                speaker=current_speaker,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        logger.info(f"Split transcript into {len(chunks)} chunks")
        return chunks
    
    async def extract_entities_from_chunk(
        self,
        chunk: TranscriptChunk,
        podcast_context: Dict[str, Any]
    ) -> List[Entity]:
        """
        Extract entities from a single transcript chunk.
        
        Args:
            chunk: The transcript chunk to process
            podcast_context: Context about the podcast (name, hosts, date)
            
        Returns:
            List of extracted Entity objects
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            podcast_name=podcast_context.get('podcast_name', 'Unknown'),
            hosts=", ".join(podcast_context.get('hosts', [])) or "Unknown",
            guests=", ".join(podcast_context.get('guests', [])) or "None",
            date=podcast_context.get('date', 'Unknown'),
            start=chunk.start_time,
            end=chunk.end_time,
            transcript=chunk.text
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a precise entity extraction assistant. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            entities = self._parse_entity_response(content, chunk.start_time)
            
            logger.debug(f"Extracted {len(entities)} entities from chunk {chunk.chunk_index}")
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed for chunk {chunk.chunk_index}: {e}")
            return []
    
    def _parse_entity_response(self, content: str, default_timestamp: float) -> List[Entity]:
        """Parse GPT response into Entity objects."""
        try:
            # Handle both array and object responses
            data = json.loads(content)
            if isinstance(data, dict):
                # GPT sometimes wraps in an object
                data = data.get('entities', data.get('results', []))
            
            if not isinstance(data, list):
                data = [data] if data else []
            
            entities = []
            for item in data:
                try:
                    # Map string type to EntityType enum
                    entity_type = EntityType(item.get('type', 'TOPIC').upper())
                    
                    # Map sentiment
                    sentiment_str = item.get('sentiment', 'neutral').lower()
                    sentiment = Sentiment(sentiment_str) if sentiment_str in ['positive', 'negative', 'neutral'] else Sentiment.NEUTRAL
                    
                    entity = Entity(
                        type=entity_type,
                        value=item.get('value', '').strip(),
                        context=item.get('context', ''),
                        timestamp=float(item.get('timestamp', default_timestamp)),
                        sentiment=sentiment,
                        speaker=item.get('speaker'),
                        confidence=float(item.get('confidence', 0.8)),
                        ad_read=bool(item.get('ad_read', False)),
                        metadata=item.get('metadata', {})
                    )
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to parse entity: {e} - {item}")
                    continue
            
            return entities
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            # Try to extract JSON from the response
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                try:
                    return self._parse_entity_response(match.group(), default_timestamp)
                except:
                    pass
            return []
    
    async def extract_all_entities(
        self,
        segments: List[TranscriptSegment],
        episode: Episode,
        progress_callback: callable = None
    ) -> List[Entity]:
        """
        Extract all entities from a complete transcript.
        
        Args:
            segments: All transcript segments
            episode: Episode metadata
            progress_callback: Optional callback for progress updates
            
        Returns:
            Deduplicated list of all extracted entities
        """
        # Create chunks
        chunks = self.chunk_transcript(segments)
        
        # Build podcast context
        podcast_context = {
            'podcast_name': episode.podcast_name,
            'hosts': episode.hosts,
            'guests': episode.guests,
            'date': episode.publish_date,
            'video_id': episode.video_id
        }
        
        all_entities = []
        
        # Process chunks with rate limiting
        batch_size = self.settings.batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.extract_entities_from_chunk(chunk, podcast_context)
                for chunk in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_entities.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
            
            if progress_callback:
                progress = min(i + batch_size, len(chunks)) / len(chunks)
                progress_callback(progress)
            
            # Rate limiting delay
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.5)
        
        # Deduplicate entities
        deduplicated = self.deduplicate_entities(all_entities)
        
        # Filter out ad reads if configured
        filtered = [e for e in deduplicated if not e.ad_read]
        
        logger.info(f"Extracted {len(filtered)} unique entities (filtered from {len(all_entities)})")
        return filtered
    
    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities by merging similar ones.
        
        Args:
            entities: List of potentially duplicate entities
            
        Returns:
            Deduplicated list with merged metadata
        """
        entity_map = defaultdict(list)
        
        for entity in entities:
            # Create a normalized key
            key = (entity.type, self._normalize_entity_value(entity.value))
            entity_map[key].append(entity)
        
        deduplicated = []
        for key, group in entity_map.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge entities
                merged = self._merge_entities(group)
                deduplicated.append(merged)
        
        return deduplicated
    
    def _normalize_entity_value(self, value: str) -> str:
        """Normalize entity value for comparison."""
        # Convert to lowercase and remove extra whitespace
        normalized = value.lower().strip()
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        # Remove punctuation at the end
        normalized = re.sub(r'[.,!?]+$', '', normalized)
        return normalized
    
    def _merge_entities(self, entities: List[Entity]) -> Entity:
        """Merge multiple entities into one."""
        # Take the entity with highest confidence
        best = max(entities, key=lambda e: e.confidence)
        
        # Collect all contexts
        contexts = list(set(e.context for e in entities if e.context))
        
        # Collect all timestamps
        timestamps = [e.timestamp for e in entities]
        
        # Aggregate sentiment (take most common)
        sentiments = [e.sentiment for e in entities]
        sentiment = max(set(sentiments), key=sentiments.count)
        
        # Merge metadata
        merged_metadata = {}
        for e in entities:
            merged_metadata.update(e.metadata)
        merged_metadata['mention_count'] = len(entities)
        merged_metadata['all_timestamps'] = timestamps
        
        return Entity(
            type=best.type,
            value=best.value,
            context=contexts[0] if contexts else "",
            timestamp=min(timestamps),  # First mention
            sentiment=sentiment,
            speaker=best.speaker,
            confidence=max(e.confidence for e in entities),
            ad_read=any(e.ad_read for e in entities),
            metadata=merged_metadata
        )
    
    def detect_cross_references(self, entities: List[Entity]) -> List[Dict[str, Any]]:
        """
        Detect cross-references between people mentioned.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of cross-reference relationships
        """
        cross_refs = []
        
        # Get all person entities
        people = [e for e in entities if e.type == EntityType.PERSON]
        
        # Look for mentions of one person by another
        for entity in entities:
            if entity.speaker and entity.type == EntityType.PERSON:
                # Someone mentioned another person
                speaker = entity.speaker
                mentioned = entity.value
                
                if speaker != mentioned:
                    cross_refs.append({
                        'from': speaker,
                        'to': mentioned,
                        'context': entity.context,
                        'timestamp': entity.timestamp,
                        'sentiment': entity.sentiment.value
                    })
        
        return cross_refs


# Convenience functions
def chunk_transcript(
    segments: List[TranscriptSegment],
    max_tokens: int = 2000
) -> List[TranscriptChunk]:
    """Chunk transcript (convenience function)."""
    extractor = EntityExtractor()
    return extractor.chunk_transcript(segments, max_tokens)


async def extract_entities_from_chunk(
    chunk: Dict[str, Any],
    podcast_context: Dict[str, Any]
) -> List[Entity]:
    """Extract entities from chunk (convenience function)."""
    extractor = EntityExtractor()
    chunk_obj = TranscriptChunk(**chunk) if isinstance(chunk, dict) else chunk
    return await extractor.extract_entities_from_chunk(chunk_obj, podcast_context)


async def extract_all_entities(
    segments: List[TranscriptSegment],
    episode: Episode
) -> List[Entity]:
    """Extract all entities (convenience function)."""
    extractor = EntityExtractor()
    return await extractor.extract_all_entities(segments, episode)


def deduplicate_entities(entities: List[Entity]) -> List[Entity]:
    """Deduplicate entities (convenience function)."""
    extractor = EntityExtractor()
    return extractor.deduplicate_entities(entities)


def detect_cross_references(entities: List[Entity]) -> List[Dict[str, Any]]:
    """Detect cross-references (convenience function)."""
    extractor = EntityExtractor()
    return extractor.detect_cross_references(entities)
