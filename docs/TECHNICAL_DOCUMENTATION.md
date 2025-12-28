# 🎙️ Podcast Knowledge Graph System
## Technical Documentation & Architecture Overview

---

**Prepared by:** Ömer Faruk Ballı  
**Date:** December 2025  
**Version:** 1.0

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Module Deep Dive](#4-module-deep-dive)
5. [Data Flow & Pipeline](#5-data-flow--pipeline)
6. [API Reference](#6-api-reference)
7. [Query Engine & Hallucination Prevention](#7-query-engine--hallucination-prevention)
8. [UAT Test Scenarios](#8-uat-test-scenarios)
9. [Performance Metrics](#9-performance-metrics)
10. [Deployment Guide](#10-deployment-guide)

---

## 1. Executive Summary

### 1.1 Project Overview

The **Podcast Knowledge Graph System** is an end-to-end AI-powered pipeline that:

- Processes YouTube podcast videos
- Transcribes audio with speaker diarization
- Extracts structured entities (People, Books, Movies, Companies, Topics)
- Builds a Neo4j knowledge graph with semantic relationships
- Enables hybrid search combining graph queries and semantic search
- Provides hallucination-resistant answers with source citations

### 1.2 Key Capabilities

| Capability | Description |
|------------|-------------|
| **Entity Extraction** | >95% precision using GPT-4 structured outputs |
| **Speaker Diarization** | Identifies who said what using AssemblyAI |
| **Knowledge Graph** | 10+ node types, 8+ relationship types in Neo4j |
| **Semantic Search** | Vector similarity using OpenAI embeddings + ChromaDB |
| **Hybrid Queries** | Combines graph structure with semantic content |
| **Hallucination Prevention** | Multi-layer verification system |
| **Query Speed** | <2s for simple queries, <5s for complex queries |

### 1.3 Use Cases

1. **Research & Analysis**: "List all books recommended by a specific host"
2. **Cross-Reference Discovery**: "Find common guests between two podcasts"
3. **Fact Verification**: "Did Person X appear on Podcast Y?"
4. **Sentiment Tracking**: "How has sentiment on Bitcoin changed over time?"
5. **Concept Tracing**: "Trace discussions about AI safety across podcasts"

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ YouTube URL  │  │   Batch JSON │  │  Streamlit   │  │     CLI      │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                │
│                                                                              │
│                    ┌─────────────────────────────┐                          │
│                    │   PodcastKnowledgeSystem    │                          │
│                    │        (main.py)            │                          │
│                    └─────────────┬───────────────┘                          │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  TRANSCRIPTION  │    │    EXTRACTION   │    │   STORAGE       │
│     SERVICE     │    │     SERVICE     │    │   SERVICES      │
│                 │    │                 │    │                 │
│ • yt-dlp        │───▶│ • GPT-4        │───▶│ • Neo4j Graph   │
│ • AssemblyAI    │    │ • Chunking     │    │ • ChromaDB      │
│ • Diarization   │    │ • Deduplication│    │ • Cache         │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUERY LAYER                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Graph Queries  │  │ Semantic Search │  │  Verification   │              │
│  │    (Cypher)     │  │   (Embeddings)  │  │   (Anti-Halluc) │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           └────────────────────┼────────────────────┘                        │
│                                ▼                                             │
│                    ┌─────────────────────────────┐                          │
│                    │     Hybrid Query Engine     │                          │
│                    │    (query_engine.py)        │                          │
│                    └─────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT LAYER                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Streamlit UI                                   │   │
│  │  • Query Interface    • Graph Visualization    • Source Citations     │   │
│  │  • Video Processing   • Statistics Dashboard   • YouTube Links        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NEO4J GRAPH SCHEMA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  NODE TYPES                          RELATIONSHIPS                          │
│  ──────────                          ─────────────                          │
│                                                                              │
│  ┌────────┐                          (Person)-[:APPEARED_ON]->(Episode)     │
│  │ Person │ ─────────────────────▶   (Person)-[:MENTIONED_IN]->(Episode)    │
│  └────────┘                          (Person)-[:REFERENCES]->(Person)       │
│                                                                              │
│  ┌────────┐                          (Book)-[:DISCUSSED_IN]->(Episode)      │
│  │  Book  │ ─────────────────────▶   (Book)-[:RECOMMENDED_BY]->(Person)     │
│  └────────┘                                                                  │
│                                                                              │
│  ┌────────┐                          (Movie)-[:REFERENCED_IN]->(Episode)    │
│  │ Movie  │ ─────────────────────▶                                          │
│  └────────┘                                                                  │
│                                                                              │
│  ┌─────────┐                         (Company)-[:MENTIONED_IN]->(Episode)   │
│  │ Company │ ────────────────────▶                                          │
│  └─────────┘                                                                 │
│                                                                              │
│  ┌─────────┐                         (Episode)-[:DISCUSSES]->(Topic)        │
│  │  Topic  │ ────────────────────▶   (Episode)-[:BELONGS_TO]->(Podcast)     │
│  └─────────┘                                                                 │
│                                                                              │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐          │
│  │ Episode │  │ Podcast │  │ Location │  │ Product  │  │  Music  │          │
│  └─────────┘  └─────────┘  └──────────┘  └──────────┘  └─────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### 3.1 Core Technologies

| Layer | Technology | Purpose | Version |
|-------|------------|---------|---------|
| **Language** | Python | Primary development language | 3.10+ |
| **LLM** | OpenAI GPT-4 | Entity extraction, answer synthesis | gpt-4-turbo |
| **Embeddings** | OpenAI | Semantic vector generation | text-embedding-3-small |
| **Transcription** | AssemblyAI | Speech-to-text with diarization | Latest API |
| **Graph DB** | Neo4j | Knowledge graph storage | 5.16+ |
| **Vector DB** | ChromaDB | Embedding storage & similarity search | 0.4.22 |
| **UI** | Streamlit | Web interface | 1.31+ |
| **Video** | yt-dlp | YouTube audio extraction | 2024.3.10 |

### 3.2 Dependencies

```
# Core
python-dotenv==1.0.0         # Environment variable management
pydantic==2.5.0              # Data validation & serialization
pydantic-settings==2.1.0     # Settings management

# AI/ML
openai==1.12.0               # GPT-4 & Embeddings API
assemblyai==0.17.0           # Transcription service
tiktoken==0.5.2              # Token counting

# Databases
neo4j==5.16.0                # Graph database driver
chromadb==0.4.22             # Vector database

# Utilities
yt-dlp==2024.3.10            # YouTube downloader
tenacity==8.2.3              # Retry logic

# UI
streamlit==1.31.0            # Web framework
plotly==5.18.0               # Interactive visualizations
networkx==3.2.1              # Graph algorithms
pyvis==0.3.2                 # Network visualization

# Testing
pytest==8.0.0                # Test framework
pytest-asyncio==0.23.4       # Async test support
```

---

## 4. Module Deep Dive

### 4.1 Configuration Module (`config.py`)

**Purpose:** Centralized configuration management with Pydantic validation.

```python
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str
    assemblyai_api_key: str
    
    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    
    # Model Settings
    gpt_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
    
    # Processing Settings
    max_tokens_per_chunk: int = 2000
    batch_size: int = 10
```

**Key Functions:**
- `get_settings()` - Returns cached settings singleton
- `get_logger(name)` - Returns configured logger instance
- `GRAPH_SCHEMA` - Neo4j schema definition dictionary
- `CYPHER_SCHEMA_STRING` - Schema string for LLM prompts

---

### 4.2 Data Models (`models/entities.py`)

**Purpose:** Pydantic models for type-safe data handling.

#### Entity Model
```python
class Entity(BaseModel):
    type: EntityType              # PERSON, BOOK, MOVIE, etc.
    value: str                    # "Elon Musk", "The Hard Thing..."
    context: str                  # Surrounding context
    timestamp: float              # When mentioned (seconds)
    sentiment: Sentiment          # positive/negative/neutral
    speaker: Optional[str]        # Who mentioned it
    confidence: float             # 0.0 - 1.0
    ad_read: bool                 # Is this from an ad?
    metadata: Dict[str, Any]      # Additional properties
    
    def generate_id(self) -> str:
        """Generate unique ID: 'person_elon_musk'"""
        
    def to_node_properties(self) -> Dict:
        """Convert to Neo4j node properties"""
```

#### Episode Model
```python
class Episode(BaseModel):
    video_id: str                 # YouTube video ID
    title: str                    # Episode title
    podcast_name: str             # Podcast name
    publish_date: str             # YYYY-MM-DD format
    duration: float               # Duration in seconds
    hosts: List[str]              # Host names
    guests: List[str]             # Guest names
    video_url: Optional[str]      # Full YouTube URL
    
    def get_timestamp_url(self, timestamp: float) -> str:
        """Get YouTube URL with timestamp parameter"""
```

#### Query Result Models
```python
class QueryResult(BaseModel):
    query: str                    # Original query
    type: Literal["graph", "semantic", "hybrid", "verify"]
    answer: str                   # Synthesized answer
    results: List[Dict]           # Raw results
    sources: List[Dict]           # Source citations
    confidence: float             # Answer confidence
    execution_time: float         # Query time in seconds
    cypher_query: Optional[str]   # Generated Cypher (if applicable)
    verified: bool                # Was answer verified?

class VerificationResult(BaseModel):
    claim: str                    # Claim being verified
    verified: bool                # True/False/None
    evidence: List[str]           # Supporting/refuting evidence
    reason: str                   # Explanation
```

---

### 4.3 Transcription Service (`services/transcription.py`)

**Purpose:** Download YouTube audio and transcribe with speaker diarization.

#### Class: `TranscriptionService`

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `download_youtube_audio()` | `video_id: str` | `str` (path) | Downloads audio as MP3 using yt-dlp |
| `get_video_info()` | `video_id: str` | `Dict` | Fetches video metadata (title, duration, etc.) |
| `transcribe_with_diarization()` | `audio_path: str, speakers_expected: int` | `List[TranscriptSegment]` | Transcribes with AssemblyAI speaker labels |
| `identify_speakers()` | `segments, hosts, guests` | `List[TranscriptSegment]` | Maps "Speaker A" to actual names |
| `cleanup_audio()` | `video_id: str` | `None` | Removes downloaded audio file |

**Implementation Details:**

```python
# AssemblyAI Configuration
config = aai.TranscriptionConfig(
    speaker_labels=True,          # Enable diarization
    speakers_expected=2,          # Expected speaker count
    language_code="en",
    punctuate=True,
    format_text=True
)

# Caching Strategy
# - Transcripts cached as JSON in data/cache/
# - Cache key: MD5 hash of audio path
# - Avoids re-processing same videos
```

---

### 4.4 Entity Extraction Service (`services/entity_extraction.py`)

**Purpose:** Extract structured entities from transcripts using GPT-4.

#### Class: `EntityExtractor`

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `chunk_transcript()` | `segments, max_tokens` | `List[TranscriptChunk]` | Splits transcript into processable chunks |
| `extract_entities_from_chunk()` | `chunk, podcast_context` | `List[Entity]` | Extracts entities from one chunk |
| `extract_all_entities()` | `segments, episode` | `List[Entity]` | Full extraction with deduplication |
| `deduplicate_entities()` | `entities` | `List[Entity]` | Merges duplicate entities |
| `detect_cross_references()` | `entities` | `List[Dict]` | Finds person-to-person references |

**GPT-4 Prompt Template:**

```python
ENTITY_EXTRACTION_PROMPT = """
You are an expert entity extractor for podcast transcripts.

Extract ALL entities from this transcript chunk. For each entity:
1. type: One of PERSON, BOOK, MOVIE, MUSIC, COMPANY, PRODUCT, LOCATION, TOPIC, QUOTE
2. value: The exact name/value
3. context: 1-2 sentences of surrounding context
4. timestamp: Approximate timestamp
5. sentiment: positive, negative, or neutral
6. speaker: Who mentioned this entity
7. confidence: 0.0-1.0 score
8. ad_read: true if this is part of a sponsor segment

CRITICAL RULES:
- For BOOKS: Extract title AND author if mentioned
- For MOVIES: Include director if mentioned
- Disambiguate: "Dune" could be book OR movie - use context clues
- Mark ad_read: true ONLY if clear advertising language
- Do NOT hallucinate - only extract what is explicitly mentioned

Transcript chunk (timestamp {start}s - {end}s):
{transcript}

Return ONLY valid JSON array of entities.
"""
```

**Deduplication Algorithm:**

```python
def deduplicate_entities(entities):
    # 1. Normalize entity values (lowercase, remove "the/a/an")
    # 2. Group by (type, normalized_value)
    # 3. For each group:
    #    - Take highest confidence entity as base
    #    - Merge contexts, collect all timestamps
    #    - Calculate aggregate sentiment
    #    - Set mention_count in metadata
    # 4. Return unique entities
```

---

### 4.5 Graph Builder Service (`services/graph_builder.py`)

**Purpose:** Build and query the Neo4j knowledge graph.

#### Class: `GraphBuilder`

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `create_schema_and_constraints()` | - | `None` | Creates indexes and unique constraints |
| `add_podcast()` | `name: str` | `str` | Creates/merges Podcast node |
| `add_episode()` | `episode: Episode` | `str` | Creates Episode node with relationships |
| `add_entities_batch()` | `video_id, entities` | `int` | Batch inserts entities with relationships |
| `add_cross_reference()` | `person1, person2, context` | `None` | Creates REFERENCES relationship |
| `find_common_guests()` | `podcast1, podcast2` | `List[str]` | Intersection query |
| `trace_concept_across_podcasts()` | `concept, podcasts` | `List[Dict]` | Chronological topic trace |
| `get_sentiment_timeline()` | `entity, podcast` | `List[Dict]` | Sentiment over time |
| `verify_entity_exists()` | `entity_name, entity_type` | `Tuple[bool, str]` | Existence check |
| `verify_relationship_exists()` | `subject, predicate, object` | `bool` | Relationship check |
| `execute_cypher()` | `cypher, parameters` | `CypherResult` | Raw Cypher execution |

**Schema Creation:**

```cypher
// Constraints (Unique IDs)
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT book_id IF NOT EXISTS FOR (b:Book) REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE;
// ... for all 10 node types

// Indexes (Performance)
CREATE INDEX episode_date IF NOT EXISTS FOR (e:Episode) ON (e.publish_date);
CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name);
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
```

**Entity-to-Relationship Mapping:**

| Entity Type | Relationship | Target | Properties |
|-------------|--------------|--------|------------|
| PERSON | MENTIONED_IN | Episode | timestamp, context, sentiment |
| BOOK | DISCUSSED_IN | Episode | timestamp, context, speaker, recommended |
| BOOK | RECOMMENDED_BY | Person | (when speaker recommends) |
| MOVIE | REFERENCED_IN | Episode | timestamp, context |
| COMPANY | MENTIONED_IN | Episode | timestamp, sentiment, stock_discussed |
| TOPIC | ← DISCUSSES | Episode | timestamp |

---

### 4.6 Vector Store Service (`services/vector_store.py`)

**Purpose:** Semantic search using embeddings and ChromaDB.

#### Class: `VectorStore`

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `embed_text()` | `text: str` | `List[float]` | Generate single embedding |
| `embed_texts_batch()` | `texts: List[str]` | `List[List[float]]` | Batch embedding generation |
| `add_transcript_chunks()` | `video_id, chunks` | `int` | Store chunks with metadata |
| `search()` | `query, n_results, filter_metadata` | `Dict` | Semantic similarity search |
| `search_by_timerange()` | `query, start_date, end_date` | `Dict` | Date-filtered search |
| `search_by_podcast()` | `query, podcast_name` | `Dict` | Podcast-filtered search |
| `search_by_video()` | `query, video_id` | `Dict` | Video-filtered search |
| `find_similar_chunks()` | `chunk_id, n_results` | `Dict` | Find similar content |
| `get_chunk_by_timestamp()` | `video_id, timestamp` | `Dict` | Lookup by timestamp |

**Metadata Schema:**

```python
{
    "video_id": "d6EMk6dyrOU",
    "podcast_name": "Dwarkesh Patel",
    "start_time": 125.5,
    "end_time": 185.2,
    "speaker": "Dwarkesh",
    "chunk_index": 3,
    "publish_date": "2025-01-15",
    "topics": "AI,Technology",
    "has_ad": false
}
```

**Embedding Configuration:**

```python
model = "text-embedding-3-small"
dimensions = 1536
similarity = "cosine"
```

---

### 4.7 Query Engine Service (`services/query_engine.py`)

**Purpose:** Hybrid query processing with hallucination prevention.

#### Class: `QueryEngine`

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `query()` | `user_query, filters` | `Dict` | Main entry point |
| `classify_intent()` | `query: str` | `QueryType` | Classify as graph/semantic/hybrid/verify |
| `execute_graph_query()` | `query: str` | `Dict` | Pure graph query |
| `execute_semantic_query()` | `query, filters` | `Dict` | Pure semantic search |
| `execute_hybrid_query()` | `query, filters` | `Dict` | Combined approach |
| `verify_claim()` | `query: str` | `Dict` | Fact verification |
| `generate_cypher()` | `natural_language` | `str` | NL to Cypher translation |

#### Query Type Classification

```python
class QueryType(Enum):
    GRAPH = "graph"      # Relationships, lists, connections
    SEMANTIC = "semantic" # Quotes, explanations, content search
    HYBRID = "hybrid"    # Combined graph + semantic
    VERIFY = "verify"    # Fact checking, claim verification
```

**Classification Examples:**

| Query | Classification |
|-------|----------------|
| "List all books recommended by X" | GRAPH |
| "What did X say about Y?" | SEMANTIC |
| "Trace concept X across podcasts" | HYBRID |
| "Did X interview Y?" | VERIFY |

#### Natural Language to Cypher

```python
CYPHER_GENERATION_PROMPT = """
Convert this natural language query to Cypher for Neo4j.

Graph Schema:
{schema}

Query: "{query}"

Examples:
- "List all books recommended by David Senra" → 
  MATCH (b:Book)-[:RECOMMENDED_BY]->(p:Person)
  WHERE toLower(p.name) CONTAINS toLower("David Senra")
  RETURN b.title, b.author

- "Who appeared on both Dwarkesh and Lex podcasts?" →
  MATCH (g:Person)-[:APPEARED_ON]->(e1:Episode)-[:BELONGS_TO]->(p1:Podcast)
  WHERE toLower(p1.name) CONTAINS "dwarkesh"
  WITH g
  MATCH (g)-[:APPEARED_ON]->(e2:Episode)-[:BELONGS_TO]->(p2:Podcast)
  WHERE toLower(p2.name) CONTAINS "lex"
  RETURN DISTINCT g.name
"""
```

---

## 5. Data Flow & Pipeline

### 5.1 Video Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         VIDEO PROCESSING FLOW                             │
└──────────────────────────────────────────────────────────────────────────┘

Step 1: INPUT
    │
    │  YouTube URL: "https://youtube.com/watch?v=d6EMk6dyrOU"
    │  Podcast Name: "Dwarkesh Patel"
    │  Hosts: ["Dwarkesh Patel"]
    │
    ▼
Step 2: METADATA EXTRACTION
    │
    │  ┌─────────────────────────────────┐
    │  │ get_video_info()                │
    │  │ • Title                         │
    │  │ • Duration                      │
    │  │ • Upload Date                   │
    │  └─────────────────────────────────┘
    │
    ▼
Step 3: AUDIO DOWNLOAD
    │
    │  ┌─────────────────────────────────┐
    │  │ download_youtube_audio()        │
    │  │ • yt-dlp extraction             │
    │  │ • MP3 conversion (192kbps)      │
    │  │ • Cache: data/cache/{id}.mp3    │
    │  └─────────────────────────────────┘
    │
    ▼
Step 4: TRANSCRIPTION
    │
    │  ┌─────────────────────────────────┐
    │  │ transcribe_with_diarization()   │
    │  │ • AssemblyAI API call           │
    │  │ • Speaker labels (Speaker A, B) │
    │  │ • Timestamps per utterance      │
    │  │ • Cache transcript as JSON      │
    │  └─────────────────────────────────┘
    │
    │  Output: List[TranscriptSegment]
    │  [
    │    {"text": "...", "start": 0.0, "end": 15.2, "speaker": "Speaker A"},
    │    {"text": "...", "start": 15.2, "end": 28.5, "speaker": "Speaker B"},
    │    ...
    │  ]
    │
    ▼
Step 5: SPEAKER IDENTIFICATION
    │
    │  ┌─────────────────────────────────┐
    │  │ identify_speakers()             │
    │  │ • Match "I'm [name]" patterns   │
    │  │ • Map Speaker A → "Dwarkesh"    │
    │  └─────────────────────────────────┘
    │
    ▼
Step 6: CHUNKING
    │
    │  ┌─────────────────────────────────┐
    │  │ chunk_transcript()              │
    │  │ • Max 2000 tokens per chunk     │
    │  │ • Preserve segment boundaries   │
    │  │ • Track start/end timestamps    │
    │  └─────────────────────────────────┘
    │
    │  Output: List[TranscriptChunk] (typically 20-50 chunks per hour)
    │
    ▼
Step 7: ENTITY EXTRACTION
    │
    │  ┌─────────────────────────────────┐
    │  │ extract_entities_from_chunk()   │
    │  │ × N chunks (parallel batches)   │
    │  │                                 │
    │  │ • GPT-4 structured extraction   │
    │  │ • JSON output validation        │
    │  │ • Rate limiting (60 RPM)        │
    │  └─────────────────────────────────┘
    │
    │  Output: List[Entity] (raw, with duplicates)
    │
    ▼
Step 8: DEDUPLICATION
    │
    │  ┌─────────────────────────────────┐
    │  │ deduplicate_entities()          │
    │  │ • Normalize entity names        │
    │  │ • Merge by (type, value)        │
    │  │ • Aggregate metadata            │
    │  │ • Remove ad reads               │
    │  └─────────────────────────────────┘
    │
    │  Output: List[Entity] (unique, ~50-200 per episode)
    │
    ▼
Step 9: GRAPH CONSTRUCTION
    │
    │  ┌─────────────────────────────────┐
    │  │ add_episode()                   │
    │  │ • Create Episode node           │
    │  │ • Link to Podcast               │
    │  │ • Add hosts/guests as Persons   │
    │  │ • Create APPEARED_ON relations  │
    │  │                                 │
    │  │ add_entities_batch()            │
    │  │ • Create entity nodes (MERGE)   │
    │  │ • Create relationships          │
    │  └─────────────────────────────────┘
    │
    ▼
Step 10: VECTOR EMBEDDINGS
    │
    │  ┌─────────────────────────────────┐
    │  │ add_transcript_chunks()         │
    │  │ • Generate embeddings (batch)   │
    │  │ • Store in ChromaDB             │
    │  │ • Include metadata              │
    │  └─────────────────────────────────┘
    │
    ▼
Step 11: OUTPUT
    │
    {
      "video_id": "d6EMk6dyrOU",
      "status": "success",
      "entity_count": 127,
      "chunk_count": 35,
      "duration": 3847.5
    }
```

### 5.2 Query Processing Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           QUERY PROCESSING FLOW                           │
└──────────────────────────────────────────────────────────────────────────┘

                    User Query
                         │
                         ▼
              ┌─────────────────────┐
              │  classify_intent()  │
              │  (GPT-3.5-turbo)    │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │  GRAPH  │    │ SEMANTIC│    │  VERIFY │
    └────┬────┘    └────┬────┘    └────┬────┘
         │               │               │
         ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│generate_cypher│ │vector.search()│ │ parse_claim() │
│execute_cypher │ │filter results │ │ check_entity  │
│               │ │               │ │ check_relation│
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └────────────────┬┘                 │
                         │                  │
                         ▼                  │
              ┌─────────────────────┐       │
              │ synthesize_answer() │◀──────┘
              │     (GPT-4)         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Return Result     │
              │   • Answer          │
              │   • Sources         │
              │   • Verified flag   │
              └─────────────────────┘
```

---

## 6. API Reference

### 6.1 Main System API

```python
from main import PodcastKnowledgeSystem

# Initialize
system = PodcastKnowledgeSystem(auto_connect=True)

# Process Video
result = await system.process_video(
    video_id="d6EMk6dyrOU",
    podcast_name="Dwarkesh Patel",
    title="Episode Title",           # Optional, auto-fetched
    publish_date="2025-01-15",       # Optional, auto-fetched
    hosts=["Dwarkesh Patel"],        # Optional
    guests=["Guest Name"],           # Optional
    progress_callback=my_callback    # Optional, for progress updates
)

# Batch Process
results = await system.batch_process(
    video_configs=[
        {"video_id": "abc123", "podcast_name": "Podcast A", ...},
        {"video_id": "def456", "podcast_name": "Podcast B", ...}
    ],
    max_concurrent=2
)

# Query
result = system.query(
    question="List all books recommended",
    filters={
        "podcast": "Dwarkesh Patel",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31"
    }
)

# Async Query
result = await system.aquery(question, filters)

# Statistics
stats = system.get_statistics()
episode_count = system.get_episode_count()
entity_count = system.get_entity_count()

# Cleanup
system.close()
```

### 6.2 Response Format

```python
# Query Response
{
    "query": "List all books recommended by Dwarkesh",
    "type": "graph",                    # graph | semantic | hybrid | verify
    "answer": "The following books were recommended: ...",
    "results": [
        {"title": "Book Title", "author": "Author Name"},
        ...
    ],
    "sources": [
        {
            "video_id": "abc123",
            "podcast": "Dwarkesh Patel",
            "start_time": 125.5,
            "end_time": 185.2,
            "speaker": "Dwarkesh",
            "text": "I really recommend this book...",
            "similarity": 0.89
        }
    ],
    "confidence": 0.92,
    "execution_time": 1.45,
    "cypher_query": "MATCH (b:Book)-[:RECOMMENDED_BY]->...",  # If graph query
    "verified": true
}

# Verification Response
{
    "query": "Did Lex Fridman interview Satoshi Nakamoto?",
    "type": "verify",
    "answer": "No record found. 'Satoshi Nakamoto' does not appear in the knowledge graph.",
    "verified": false,
    "confidence": 0.95,
    "reason": "Entity 'Satoshi Nakamoto' not found in database",
    "evidence": ["Searched Person nodes", "No APPEARED_ON relationship found"],
    "sources": []
}
```

---

## 7. Query Engine & Hallucination Prevention

### 7.1 Hallucination Prevention Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ANTI-HALLUCINATION LAYERS                              │
└──────────────────────────────────────────────────────────────────────────┘

Layer 1: ENTITY VERIFICATION
    │
    │  ┌─────────────────────────────────────────────────────────┐
    │  │ verify_entity_exists(entity_name, entity_type)         │
    │  │                                                         │
    │  │ MATCH (n)                                               │
    │  │ WHERE toLower(n.name) = toLower($name)                  │
    │  │    OR toLower(n.title) = toLower($name)                 │
    │  │ RETURN n, labels(n)                                     │
    │  │                                                         │
    │  │ Result: (exists: bool, node_type: str)                  │
    │  └─────────────────────────────────────────────────────────┘
    │
    ▼
Layer 2: RELATIONSHIP VERIFICATION
    │
    │  ┌─────────────────────────────────────────────────────────┐
    │  │ verify_relationship_exists(subject, predicate, object)  │
    │  │                                                         │
    │  │ MATCH (a)-[r]->(b)                                      │
    │  │ WHERE toLower(a.name) CONTAINS toLower($subject)        │
    │  │   AND toLower(b.name) CONTAINS toLower($object)         │
    │  │   AND type(r) = $predicate                              │
    │  │ RETURN count(r) > 0 as exists                           │
    │  └─────────────────────────────────────────────────────────┘
    │
    ▼
Layer 3: DATE BOUNDARY ENFORCEMENT
    │
    │  ┌─────────────────────────────────────────────────────────┐
    │  │ All semantic search results filtered by:                │
    │  │ • start_date <= metadata.publish_date <= end_date       │
    │  │ • Prevents returning content from outside date range    │
    │  └─────────────────────────────────────────────────────────┘
    │
    ▼
Layer 4: SPEAKER VALIDATION
    │
    │  ┌─────────────────────────────────────────────────────────┐
    │  │ verify_speaker_exists(speaker, video_id)                │
    │  │                                                         │
    │  │ Confirms speaker appeared in the specified episode      │
    │  │ before attributing quotes                               │
    │  └─────────────────────────────────────────────────────────┘
    │
    ▼
Layer 5: EVIDENCE-BASED SYNTHESIS
    │
    │  ┌─────────────────────────────────────────────────────────┐
    │  │ synthesize_answer() Rules:                              │
    │  │ • Answer ONLY from provided context                     │
    │  │ • If insufficient info, explicitly state so             │
    │  │ • Never invent timestamps or quotes                     │
    │  │ • Include source citations                              │
    │  └─────────────────────────────────────────────────────────┘
```

### 7.2 Claim Verification Process

```python
async def verify_claim(self, query: str) -> Dict:
    """
    Example: "Did Lex Fridman interview Satoshi Nakamoto?"
    
    Step 1: Parse claim
    {
        "subject": "Lex Fridman",
        "predicate": "interviewed", 
        "object": "Satoshi Nakamoto"
    }
    
    Step 2: Entity checks
    - verify_entity_exists("Lex Fridman", "Person")       → True/False
    - verify_entity_exists("Satoshi Nakamoto", "Person") → True/False
    
    Step 3: Relationship check
    - verify_relationship_exists("Lex", "APPEARED_ON/INTERVIEWED", "Satoshi")
    
    Step 4: Semantic evidence search
    - vector_search("Lex Fridman Satoshi Nakamoto interview")
    
    Step 5: LLM verification
    - Analyze all evidence
    - Return verified=True/False/None with reason
    """
```

---

## 8. UAT Test Scenarios

### 8.1 Test Categories

| Category | Tests | Focus |
|----------|-------|-------|
| **Entity Extraction** | UAT-01 to UAT-10 | Books, Movies, People, Companies, Topics |
| **Hallucination Resistance** | UAT-11 to UAT-20 | False premises, fake quotes, phantom entities |
| **Synthesis & Logic** | UAT-21 to UAT-30 | Cross-references, timelines, sentiment analysis |

### 8.2 Critical Test Cases

#### UAT-11: False Premise Rejection
```python
def test_uat_11_false_premise_rejection(self, system):
    """
    Query: "Show me the clip where Lex Fridman interviews Satoshi Nakamoto"
    Expected: "No record found of this interview"
    
    CRITICAL: System must NOT hallucinate an interview that doesn't exist
    """
    result = system.query(query)
    
    assert result["verified"] == False
    assert "no record" in result["answer"].lower() or "not found" in result["answer"].lower()
```

#### UAT-14: Fake Quote Verification
```python
def test_uat_14_fake_quote_verification(self, system):
    """
    Query: "Verify if David Senra said: 'Steve Jobs was a terrible marketer'"
    Expected: "False. Evidence suggests he said the opposite"
    
    CRITICAL: Must refute fabricated quotes
    """
    result = system.query(query)
    
    assert result["verified"] == False
    assert "false" in result["answer"].lower() or "no evidence" in result["answer"].lower()
```

#### UAT-20: Speed Requirement
```python
def test_uat_20_speed_requirement(self, system):
    """
    Query: Simple keyword search
    Expected: Result in < 2.0 seconds
    """
    start = time.time()
    result = system.query("Find mentions of AI")
    latency = time.time() - start
    
    assert latency < 2.0, f"Query took {latency:.2f}s (limit: 2.0s)"
```

### 8.3 Running Tests

```bash
# All UAT tests
pytest tests/test_uat.py -v

# Specific category
pytest tests/test_uat.py::TestHallucinationResistanceUAT -v

# With coverage
pytest tests/test_uat.py --cov=services --cov-report=html
```

---

## 9. Performance Metrics

### 9.1 Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Simple Query Latency** | < 2.0s | Keyword searches, simple graph queries |
| **Complex Query Latency** | < 5.0s | Hybrid queries, cross-podcast analysis |
| **Entity Extraction Precision** | > 95% | Verified against manual annotation |
| **Hallucination Rate** | 0% | For verification queries (UAT-11 to UAT-20) |
| **Video Processing Time** | ~5-10 min/hour | Depends on transcript length |

### 9.2 Resource Usage

| Component | Memory | CPU | Notes |
|-----------|--------|-----|-------|
| Neo4j | 2-4 GB | Moderate | Scales with graph size |
| ChromaDB | 1-2 GB | Low | Persistent storage on disk |
| Python App | 500 MB | Variable | Peaks during GPT-4 calls |
| Total | 4-8 GB | - | For ~100 episodes |

---

## 10. Deployment Guide

### 10.1 Prerequisites

```bash
# System requirements
- Python 3.10+
- Docker (for Neo4j)
- 8 GB RAM minimum
- FFmpeg (for audio processing)

# API Keys needed
- OpenAI API Key (GPT-4 access)
- AssemblyAI API Key
```

### 10.2 Quick Start

```bash
# 1. Clone and setup
cd podcast-kg
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:latest

# 4. Process first video
python main.py --url "https://youtube.com/watch?v=VIDEO_ID" --podcast "Podcast Name"

# 5. Launch UI
streamlit run ui/app.py
```

### 10.3 Production Considerations

1. **Neo4j**: Use Neo4j Aura (cloud) for production
2. **Rate Limiting**: Implement request queuing for API calls
3. **Caching**: Enable Redis for API response caching
4. **Monitoring**: Add Prometheus metrics export
5. **Backup**: Regular Neo4j and ChromaDB backups

---

## Contact

For questions or issues, please contact:
- **Email:** [Your Email]
- **GitHub:** [Repository URL]

---

*Document Version: 1.0 | Last Updated: December 2025*
