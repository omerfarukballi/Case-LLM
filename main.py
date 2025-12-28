"""
Podcast Knowledge Graph System - Main Orchestrator

This is the main entry point for the podcast knowledge graph pipeline.
It coordinates transcription, entity extraction, graph building, and querying.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings, LogConfig, get_logger
from models.entities import Episode, TranscriptSegment
from services.transcription import TranscriptionService
from services.entity_extraction import EntityExtractor
from services.graph_builder import GraphBuilder
from services.vector_store import VectorStore
from services.query_engine import QueryEngine

# Initialize logging
settings = get_settings()
LogConfig.setup_logging(settings.log_level)
logger = get_logger(__name__)


class PodcastKnowledgeSystem:
    """
    Main orchestrator for the Podcast Knowledge Graph System.
    
    Coordinates the entire pipeline:
    1. YouTube audio download
    2. Transcription with speaker diarization
    3. Entity extraction with GPT-4
    4. Knowledge graph construction in Neo4j
    5. Vector embeddings in ChromaDB
    6. Hybrid query execution
    """
    
    def __init__(self, auto_connect: bool = True):
        """
        Initialize the system.
        
        Args:
            auto_connect: Whether to connect to databases on init
        """
        self.settings = get_settings()
        
        # Initialize services
        self.transcription = TranscriptionService()
        self.entity_extractor = EntityExtractor()
        
        if auto_connect:
            try:
                self.graph = GraphBuilder()
                self.vector_store = VectorStore()
                self.query_engine = QueryEngine(self.graph, self.vector_store)
                
                # Initialize schema
                self.graph.create_schema_and_constraints()
                logger.info("Podcast Knowledge System initialized successfully")
            except Exception as e:
                logger.error(f"Failed to connect to databases: {e}")
                raise
        else:
            self.graph = None
            self.vector_store = None
            self.query_engine = None
    
    async def process_video(
        self,
        video_id: str,
        podcast_name: str,
        title: str = None,
        publish_date: str = None,
        hosts: List[str] = None,
        guests: List[str] = None,
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline for processing one video.
        
        Args:
            video_id: YouTube video ID (e.g., 'd6EMk6dyrOU')
            podcast_name: Name of the podcast
            title: Episode title (fetched if not provided)
            publish_date: Publish date YYYY-MM-DD (fetched if not provided)
            hosts: List of host names
            guests: List of guest names
            progress_callback: Optional callback(step, progress, message)
            
        Returns:
            Processing stats and results
        """
        def update_progress(step: str, progress: float, message: str = ""):
            if progress_callback:
                progress_callback(step, progress, message)
            logger.info(f"[{step}] {progress*100:.0f}% - {message}")
        
        try:
            update_progress("init", 0, f"Starting processing for {video_id}")
            
            # Step 1: Get video info if needed
            if not title or not publish_date:
                update_progress("metadata", 0.05, "Fetching video metadata...")
                video_info = await self.transcription.get_video_info(video_id)
                title = title or video_info.get("title", f"Episode {video_id}")
                publish_date = publish_date or video_info.get("upload_date", "2025-01-01")
            
            # Step 2: Download audio
            update_progress("download", 0.1, "Downloading audio from YouTube...")
            audio_path = await self.transcription.download_youtube_audio(video_id)
            
            # Step 3: Transcribe with diarization
            update_progress("transcription", 0.2, "Transcribing with speaker diarization...")
            num_speakers = len(hosts or []) + len(guests or [])
            segments = await self.transcription.transcribe_with_diarization(
                audio_path,
                speakers_expected=num_speakers if num_speakers > 0 else None
            )
            
            # Step 4: Identify speakers
            if hosts:
                update_progress("speakers", 0.35, "Identifying speakers...")
                segments = self.transcription.identify_speakers(segments, hosts, guests)
            
            # Step 5: Create episode object
            episode = Episode(
                video_id=video_id,
                title=title,
                podcast_name=podcast_name,
                publish_date=publish_date,
                duration=segments[-1].end if segments else 0,
                hosts=hosts or [],
                guests=guests or []
            )
            
            # Step 6: Extract entities
            update_progress("extraction", 0.4, "Extracting entities with GPT-4...")
            
            def entity_progress(p):
                update_progress("extraction", 0.4 + p * 0.3, f"Extracting entities... {p*100:.0f}%")
            
            entities = await self.entity_extractor.extract_all_entities(
                segments, 
                episode,
                progress_callback=entity_progress
            )
            
            # Step 7: Build knowledge graph
            update_progress("graph", 0.75, "Building knowledge graph...")
            self.graph.add_episode(episode)
            entity_count = self.graph.add_entities_batch(video_id, entities)
            
            # Step 8: Detect and add cross-references
            update_progress("crossrefs", 0.85, "Detecting cross-references...")
            cross_refs = self.entity_extractor.detect_cross_references(entities)
            for ref in cross_refs:
                self.graph.add_cross_reference(
                    ref["from"], ref["to"], ref["context"]
                )
            
            # Step 9: Create vector embeddings
            update_progress("embeddings", 0.9, "Creating vector embeddings...")
            chunks = self.entity_extractor.chunk_transcript(segments)
            
            # Enrich chunks with metadata
            for chunk in chunks:
                chunk.video_id = video_id
                chunk.podcast_name = podcast_name
                chunk.publish_date = publish_date
            
            self.vector_store.add_transcript_chunks(
                video_id=video_id,
                chunks=chunks,
                podcast_name=podcast_name,
                publish_date=publish_date
            )
            
            update_progress("complete", 1.0, f"Processing complete!")
            
            result = {
                "video_id": video_id,
                "title": title,
                "status": "success",
                "entity_count": entity_count,
                "chunk_count": len(chunks),
                "segment_count": len(segments),
                "cross_reference_count": len(cross_refs),
                "duration": episode.duration
            }
            
            logger.info(f"✓ Successfully processed: {title} ({entity_count} entities)")
            return result
            
        except Exception as e:
            logger.error(f"✗ Failed to process {video_id}: {str(e)}")
            return {
                "video_id": video_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def batch_process(
        self,
        video_configs: List[Dict[str, Any]],
        max_concurrent: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos with concurrency control.
        
        Args:
            video_configs: List of video configuration dicts
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(config):
            async with semaphore:
                return await self.process_video(**config)
        
        tasks = [process_with_semaphore(config) for config in video_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "video_id": video_configs[i].get("video_id", "unknown"),
                    "status": "failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def query(
        self,
        question: str,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph.
        
        Args:
            question: Natural language question
            filters: Optional filters (video_id, podcast, date_range)
            
        Returns:
            Query result with answer and sources
        """
        if not self.query_engine:
            raise RuntimeError("System not initialized with database connections")
        
        return asyncio.run(self.query_engine.query(question, filters))
    
    async def aquery(
        self,
        question: str,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Async version of query."""
        return await self.query_engine.query(question, filters)
    
    def get_episode_count(self) -> int:
        """Get the number of episodes in the graph."""
        if not self.graph:
            return 0
        stats = self.graph.get_statistics()
        return stats.get("episode_count", 0)
    
    def get_entity_count(self) -> int:
        """Get the total number of entities in the graph."""
        if not self.graph:
            return 0
        stats = self.graph.get_statistics()
        total = 0
        for key, value in stats.items():
            if key.endswith("_count") and key not in ["episode_count", "podcast_count", "relationship_count"]:
                total += value
        return total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        graph_stats = self.graph.get_statistics() if self.graph else {}
        vector_stats = self.vector_store.get_statistics() if self.vector_store else {}
        
        return {
            "graph": graph_stats,
            "vectors": vector_stats
        }
    
    def close(self):
        """Close all connections."""
        if self.graph:
            self.graph.close()
        logger.info("System connections closed")


def parse_video_url(url: str) -> str:
    """Extract video ID from YouTube URL."""
    import re
    
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be/)([0-9A-Za-z_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Assume it's already a video ID
    if len(url) == 11:
        return url
    
    raise ValueError(f"Could not extract video ID from: {url}")


# CLI Interface
def main():
    parser = argparse.ArgumentParser(
        description="Podcast Knowledge Graph System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  python main.py --process d6EMk6dyrOU --podcast "All-In" --title "Episode 1"
  
  # Process from URL
  python main.py --url "https://www.youtube.com/watch?v=d6EMk6dyrOU" --podcast "All-In"
  
  # Batch process from JSON
  python main.py --batch videos.json
  
  # Query the knowledge graph
  python main.py --query "List all books recommended by David Senra"
  
  # Get statistics
  python main.py --stats
        """
    )
    
    parser.add_argument("--process", help="Process a video by ID")
    parser.add_argument("--url", help="Process a video by URL")
    parser.add_argument("--batch", help="Process videos from JSON file")
    parser.add_argument("--query", help="Query the knowledge graph")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    
    parser.add_argument("--podcast", help="Podcast name")
    parser.add_argument("--title", help="Episode title")
    parser.add_argument("--date", help="Publish date (YYYY-MM-DD)")
    parser.add_argument("--hosts", help="Comma-separated host names")
    parser.add_argument("--guests", help="Comma-separated guest names")
    
    args = parser.parse_args()
    
    # Initialize system
    system = PodcastKnowledgeSystem()
    
    try:
        if args.process or args.url:
            video_id = args.process or parse_video_url(args.url)
            
            if not args.podcast:
                args.podcast = input("Podcast name: ")
            
            result = asyncio.run(system.process_video(
                video_id=video_id,
                podcast_name=args.podcast,
                title=args.title,
                publish_date=args.date,
                hosts=args.hosts.split(",") if args.hosts else [],
                guests=args.guests.split(",") if args.guests else []
            ))
            print(json.dumps(result, indent=2))
        
        elif args.batch:
            with open(args.batch) as f:
                videos = json.load(f)
            results = asyncio.run(system.batch_process(videos))
            print(json.dumps(results, indent=2))
        
        elif args.query:
            result = system.query(args.query)
            print("\n" + "="*60)
            print(f"Query: {args.query}")
            print("="*60)
            print(f"\nType: {result.get('type', 'unknown')}")
            print(f"\nAnswer:\n{result.get('answer', 'No answer')}")
            
            if result.get("sources"):
                print(f"\nSources ({len(result['sources'])}):")
                for source in result["sources"][:5]:
                    video_id = source.get("video_id", "")
                    timestamp = source.get("start_time", 0)
                    if video_id:
                        print(f"  - https://youtube.com/watch?v={video_id}&t={int(timestamp)}")
            
            if result.get("cypher_query"):
                print(f"\nCypher Query:\n{result['cypher_query']}")
            
            print(f"\nExecution Time: {result.get('execution_time', 0):.2f}s")
            print("="*60)
        
        elif args.stats:
            stats = system.get_statistics()
            print("\n" + "="*40)
            print("PODCAST KNOWLEDGE GRAPH STATISTICS")
            print("="*40)
            print("\nGraph Database:")
            for key, value in stats.get("graph", {}).items():
                print(f"  {key}: {value}")
            print("\nVector Store:")
            for key, value in stats.get("vectors", {}).items():
                print(f"  {key}: {value}")
            print("="*40)
        
        else:
            parser.print_help()
    
    finally:
        system.close()


if __name__ == "__main__":
    main()
