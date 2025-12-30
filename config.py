"""
Configuration module for Podcast Knowledge Graph System.
Uses Pydantic for validation and environment variable loading.
"""

import os
from typing import List, Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
import logging

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(default="sk-placeholder", env="OPENAI_API_KEY")
    assemblyai_api_key: str = Field(default="placeholder", env="ASSEMBLYAI_API_KEY")
    
    # Local LLM Settings (Ollama)
    use_local_llm: bool = Field(default=True, env="USE_LOCAL_LLM")
    local_llm_model: str = Field(default="mistral", env="LOCAL_LLM_MODEL")
    local_LLM_api_base: str = Field(default="http://localhost:11434/v1", env="LOCAL_LLM_API_BASE")
    local_whisper_model: str = Field(default="base", env="LOCAL_WHISPER_MODEL")

    # Neo4j Settings
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # OpenAI Model Settings
    gpt_model: str = Field(default="gpt-4-turbo-preview")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    
    # Processing Settings
    max_tokens_per_chunk: int = Field(default=2000)
    chunk_overlap: int = Field(default=200)
    batch_size: int = Field(default=10)
    
    # Rate Limiting
    openai_requests_per_minute: int = Field(default=60)
    openai_tokens_per_minute: int = Field(default=90000)
    assemblyai_concurrent_limit: int = Field(default=5)
    
    # Retry Settings
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    retry_exponential_base: float = Field(default=2.0)
    
    # Paths
    data_dir: str = Field(default="./data")
    cache_dir: str = Field(default="./data/cache")
    chroma_persist_dir: str = Field(default="./chroma_db")
    
    # Entity Types
    entity_types: List[str] = Field(default=[
        "PERSON", "BOOK", "MOVIE", "MUSIC", "COMPANY", 
        "PRODUCT", "LOCATION", "TOPIC", "QUOTE"
    ])
    
    # Confidence Thresholds
    entity_confidence_threshold: float = Field(default=0.7)
    semantic_similarity_threshold: float = Field(default=0.75)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class LogConfig:
    """Logging configuration."""
    
    @staticmethod
    def setup_logging(level: str = "INFO") -> logging.Logger:
        """Set up logging configuration."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        
        # Suppress noisy loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        
        return root_logger


# Graph Schema Definition
GRAPH_SCHEMA = {
    "nodes": {
        "Person": ["id", "name"],
        "Book": ["id", "title", "author"],
        "Movie": ["id", "title", "director"],
        "Music": ["id", "title", "artist"],
        "Company": ["id", "name"],
        "Product": ["id", "name"],
        "Location": ["id", "name"],
        "Topic": ["id", "name"],
        "Episode": ["id", "title", "video_id", "publish_date", "duration", "video_url"],
        "Podcast": ["id", "name"]
    },
    "relationships": {
        "APPEARED_ON": {"from": "Person", "to": "Episode", "properties": ["role"]},
        "MENTIONED_IN": {"from": ["Person", "Company"], "to": "Episode", "properties": ["timestamp", "context", "sentiment"]},
        "DISCUSSED_IN": {"from": "Book", "to": "Episode", "properties": ["timestamp", "context", "speaker", "recommended"]},
        "RECOMMENDED_BY": {"from": "Book", "to": "Person", "properties": []},
        "REFERENCED_IN": {"from": "Movie", "to": "Episode", "properties": ["timestamp", "context"]},
        "DISCUSSES": {"from": "Episode", "to": "Topic", "properties": ["timestamp"]},
        "BELONGS_TO": {"from": "Episode", "to": "Podcast", "properties": []},
        "REFERENCES": {"from": "Person", "to": "Person", "properties": ["count", "contexts"]}
    }
}

# Cypher Schema String for LLM
CYPHER_SCHEMA_STRING = """
Node Types:
- Person(id, name)
- Book(id, title, author)
- Movie(id, title, director)
- Music(id, title, artist)
- Company(id, name)
- Product(id, name)
- Location(id, name)
- Topic(id, name)
- Episode(id, title, video_id, publish_date, duration, video_url)
- Podcast(id, name)

Relationships:
- (Person)-[:APPEARED_ON {role: "host"|"guest"}]->(Episode)
- (Person)-[:MENTIONED_IN {timestamp, context, sentiment}]->(Episode)
- (Book)-[:DISCUSSED_IN {timestamp, context, speaker, recommended: bool}]->(Episode)
- (Book)-[:RECOMMENDED_BY]->(Person)
- (Movie)-[:REFERENCED_IN {timestamp, context}]->(Episode)
- (Company)-[:MENTIONED_IN {timestamp, sentiment, stock_discussed: bool}]->(Episode)
- (Episode)-[:DISCUSSES {timestamp}]->(Topic)
- (Episode)-[:BELONGS_TO]->(Podcast)
- (Person)-[:REFERENCES {count, contexts: []}]->(Person)
"""


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
