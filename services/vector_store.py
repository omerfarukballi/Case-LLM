"""
Vector Store Service for the Podcast Knowledge Graph System.
Uses ChromaDB for semantic search with OpenAI embeddings.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from models.entities import TranscriptChunk
from config import get_settings, get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Service for managing vector embeddings and semantic search.
    
    Features:
    - OpenAI text-embedding-3-small embeddings
    - ChromaDB persistent storage
    - Metadata filtering
    - Hybrid search with filters
    - Similarity threshold filtering
    """
    
    COLLECTION_NAME = "podcast_transcripts"
    
    def __init__(self, persist_directory: str = None):
        self.settings = get_settings()
        self.persist_directory = persist_directory or self.settings.chroma_persist_dir
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client for embeddings
        self.openai = OpenAI(api_key=self.settings.openai_api_key)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store initialized with {self.collection.count()} documents")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.openai.embeddings.create(
                model=self.settings.embedding_model,
                input=text,
                dimensions=self.settings.embedding_dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai.embeddings.create(
                model=self.settings.embedding_model,
                input=texts,
                dimensions=self.settings.embedding_dimensions
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise
    
    def add_transcript_chunks(
        self,
        video_id: str,
        chunks: List[TranscriptChunk],
        podcast_name: str = None,
        publish_date: str = None
    ) -> int:
        """
        Add transcript chunks to the vector store.
        
        Args:
            video_id: The video ID
            chunks: List of TranscriptChunk objects
            podcast_name: Optional podcast name (can be in chunk metadata)
            publish_date: Optional publish date (can be in chunk metadata)
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Prepare documents, IDs, and metadata
        documents = []
        ids = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{video_id}_{chunk.chunk_index if hasattr(chunk, 'chunk_index') else i}"
            
            # Use chunk metadata or provided values
            metadata = {
                "video_id": video_id,
                "podcast_name": chunk.podcast_name or podcast_name or "Unknown",
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "speaker": chunk.speaker or "unknown",
                "chunk_index": chunk.chunk_index if hasattr(chunk, 'chunk_index') else i,
                "publish_date": chunk.publish_date or publish_date or "",
                "has_ad": chunk.has_ad if hasattr(chunk, 'has_ad') else False,
            }
            
            # Add topics as comma-separated string (ChromaDB doesn't support lists in metadata)
            if hasattr(chunk, 'topics') and chunk.topics:
                metadata["topics"] = ",".join(chunk.topics)
            
            documents.append(chunk.text)
            ids.append(doc_id)
            metadatas.append(metadata)
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.embed_texts_batch(batch)
            all_embeddings.extend(embeddings)
        
        # Add to collection
        try:
            self.collection.add(
                documents=documents,
                ids=ids,
                embeddings=all_embeddings,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks for video {video_id}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None,
        include_distances: bool = True
    ) -> Dict[str, Any]:
        """
        Search for similar chunks using semantic search.
        
        Args:
            query: Search query text
            n_results: Maximum number of results
            filter_metadata: Optional metadata filters
            include_distances: Whether to include distance scores
            
        Returns:
            Dictionary with search results
        """
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Build where clause for filtering
        where = None
        if filter_metadata:
            where = self._build_where_clause(filter_metadata)
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"] if include_distances else ["documents", "metadatas"]
            )
            
            # Format results
            formatted = self._format_results(results)
            return formatted
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"results": [], "error": str(e)}
    
    def search_by_timerange(
        self,
        query: str,
        start_date: str,
        end_date: str,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search with date range filtering.
        
        Args:
            query: Search query text
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            n_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        # ChromaDB doesn't support range queries on strings directly
        # We'll fetch more results and filter
        
        query_embedding = self.embed_text(query)
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 3,  # Fetch extra to filter
                include=["documents", "metadatas", "distances"]
            )
            
            # Filter by date
            filtered_results = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
            
            for i, metadata in enumerate(results["metadatas"][0]):
                date = metadata.get("publish_date", "")
                if date and start_date <= date <= end_date:
                    filtered_results["ids"][0].append(results["ids"][0][i])
                    filtered_results["documents"][0].append(results["documents"][0][i])
                    filtered_results["metadatas"][0].append(metadata)
                    if results.get("distances"):
                        filtered_results["distances"][0].append(results["distances"][0][i])
            
            # Limit to n_results
            for key in filtered_results:
                filtered_results[key][0] = filtered_results[key][0][:n_results]
            
            return self._format_results(filtered_results)
            
        except Exception as e:
            logger.error(f"Date range search failed: {e}")
            return {"results": [], "error": str(e)}
    
    def search_by_podcast(
        self,
        query: str,
        podcast_name: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search within a specific podcast.
        
        Args:
            query: Search query text
            podcast_name: Name of the podcast to search in
            n_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        return self.search(
            query=query,
            n_results=n_results,
            filter_metadata={"podcast_name": podcast_name}
        )
    
    def search_by_video(
        self,
        query: str,
        video_id: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search within a specific video.
        
        Args:
            query: Search query text
            video_id: Video ID to search in
            n_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        return self.search(
            query=query,
            n_results=n_results,
            filter_metadata={"video_id": video_id}
        )
    
    def find_similar_chunks(
        self,
        chunk_id: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the chunk to find similar to
            n_results: Maximum number of results
            
        Returns:
            Dictionary with similar chunks
        """
        try:
            # Get the chunk's embedding
            chunk_data = self.collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not chunk_data["ids"]:
                return {"results": [], "error": "Chunk not found"}
            
            # Search using the chunk's embedding
            results = self.collection.query(
                query_embeddings=chunk_data["embeddings"],
                n_results=n_results + 1,  # +1 because it will include itself
                include=["documents", "metadatas", "distances"]
            )
            
            # Remove the query chunk itself from results
            formatted = self._format_results(results)
            formatted["results"] = [r for r in formatted["results"] if r["id"] != chunk_id][:n_results]
            
            return formatted
            
        except Exception as e:
            logger.error(f"Similar chunk search failed: {e}")
            return {"results": [], "error": str(e)}
    
    def get_chunk_by_timestamp(
        self,
        video_id: str,
        timestamp: float,
        tolerance: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get a chunk at a specific timestamp.
        
        Args:
            video_id: Video ID
            timestamp: Target timestamp in seconds
            tolerance: Tolerance window in seconds
            
        Returns:
            Chunk data if found
        """
        try:
            # Get all chunks for this video
            results = self.collection.get(
                where={"video_id": video_id},
                include=["documents", "metadatas"]
            )
            
            if not results["ids"]:
                return None
            
            # Find chunk containing the timestamp
            for i, metadata in enumerate(results["metadatas"]):
                start = metadata.get("start_time", 0)
                end = metadata.get("end_time", 0)
                
                if start - tolerance <= timestamp <= end + tolerance:
                    return {
                        "id": results["ids"][i],
                        "text": results["documents"][i],
                        "metadata": metadata
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Timestamp lookup failed: {e}")
            return None
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        conditions = []
        
        for key, value in filters.items():
            if value is not None:
                if isinstance(value, bool):
                    conditions.append({key: {"$eq": value}})
                elif isinstance(value, str):
                    # Case-insensitive partial match not directly supported
                    # Using exact match for now
                    conditions.append({key: {"$eq": value}})
                elif isinstance(value, (int, float)):
                    conditions.append({key: {"$eq": value}})
                elif isinstance(value, list):
                    conditions.append({key: {"$in": value}})
        
        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    def _format_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format ChromaDB results into a clean structure."""
        formatted_results = []
        
        if results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                }
                
                if results.get("distances"):
                    # Convert distance to similarity score (1 - distance for cosine)
                    distance = results["distances"][0][i]
                    result["distance"] = distance
                    result["similarity"] = 1 - distance
                
                formatted_results.append(result)
        
        return {
            "results": formatted_results,
            "count": len(formatted_results)
        }
    
    def delete_video(self, video_id: str) -> int:
        """
        Delete all chunks for a video.
        
        Args:
            video_id: Video ID to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            # Get IDs to delete
            results = self.collection.get(
                where={"video_id": video_id}
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for video {video_id}")
                return len(results["ids"])
            
            return 0
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            count = self.collection.count()
            
            # Get unique video IDs
            results = self.collection.get(include=["metadatas"])
            video_ids = set()
            podcast_names = set()
            
            for metadata in results.get("metadatas", []):
                if metadata:
                    video_ids.add(metadata.get("video_id", ""))
                    podcast_names.add(metadata.get("podcast_name", ""))
            
            return {
                "total_chunks": count,
                "unique_videos": len(video_ids) - (1 if "" in video_ids else 0),
                "unique_podcasts": len(podcast_names) - (1 if "" in podcast_names else 0),
                "collection_name": self.COLLECTION_NAME
            }
            
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            return {"error": str(e)}
    
    def reset(self) -> None:
        """Reset the vector store (delete all data)."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector store reset")
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise
