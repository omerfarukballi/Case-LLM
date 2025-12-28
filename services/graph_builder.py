"""
Graph Builder Service for the Podcast Knowledge Graph System.
Handles Neo4j database operations for building and querying the knowledge graph.
"""

from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import logging

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from models.entities import Entity, EntityType, Episode, Sentiment
from models.graph_schema import (
    NodeType, 
    RelationshipType, 
    PredefinedQueries,
    CypherResult
)
from config import get_settings, get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """
    Service for building and querying the Neo4j knowledge graph.
    
    Features:
    - Schema creation with constraints and indexes
    - Batch node and relationship creation
    - Entity deduplication at database level
    - Cross-reference tracking
    - Sentiment analysis aggregation
    """
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        self.settings = get_settings()
        self.uri = uri or self.settings.neo4j_uri
        self.user = user or self.settings.neo4j_user
        self.password = password or self.settings.neo4j_password
        
        self._driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connection
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
    
    @property
    def driver(self) -> Driver:
        """Get the Neo4j driver, reconnecting if necessary."""
        if self._driver is None:
            self._connect()
        return self._driver
    
    @contextmanager
    def session(self) -> Session:
        """Context manager for Neo4j sessions."""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def close(self) -> None:
        """Close the database connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    def create_schema_and_constraints(self) -> None:
        """Create database schema with constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT book_id IF NOT EXISTS FOR (b:Book) REQUIRE b.id IS UNIQUE",
            "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT music_id IF NOT EXISTS FOR (m:Music) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
            "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT podcast_id IF NOT EXISTS FOR (p:Podcast) REQUIRE p.id IS UNIQUE",
        ]
        
        indexes = [
            "CREATE INDEX episode_date IF NOT EXISTS FOR (e:Episode) ON (e.publish_date)",
            "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX episode_video IF NOT EXISTS FOR (e:Episode) ON (e.video_id)",
            "CREATE INDEX book_title IF NOT EXISTS FOR (b:Book) ON (b.title)",
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
        ]
        
        with self.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")
        
        logger.info("Database schema created/verified")
    
    def add_podcast(self, name: str) -> str:
        """Add or merge a podcast node."""
        podcast_id = f"podcast_{name.lower().replace(' ', '_')}"
        
        with self.session() as session:
            session.run(
                """
                MERGE (p:Podcast {id: $id})
                ON CREATE SET p.name = $name
                """,
                id=podcast_id,
                name=name
            )
        
        return podcast_id
    
    def add_episode(self, episode: Episode) -> str:
        """
        Add an episode to the graph.
        
        Args:
            episode: Episode object with metadata
            
        Returns:
            Episode node ID
        """
        episode_id = f"episode_{episode.video_id}"
        podcast_id = self.add_podcast(episode.podcast_name)
        
        with self.session() as session:
            # Create episode node
            session.run(
                """
                MERGE (e:Episode {id: $id})
                ON CREATE SET 
                    e.video_id = $video_id,
                    e.title = $title,
                    e.publish_date = $publish_date,
                    e.duration = $duration,
                    e.video_url = $video_url
                ON MATCH SET
                    e.title = $title,
                    e.publish_date = $publish_date,
                    e.duration = $duration,
                    e.video_url = $video_url
                """,
                id=episode_id,
                video_id=episode.video_id,
                title=episode.title,
                publish_date=episode.publish_date,
                duration=episode.duration,
                video_url=episode.video_url
            )
            
            # Link to podcast
            session.run(
                """
                MATCH (e:Episode {id: $episode_id})
                MATCH (p:Podcast {id: $podcast_id})
                MERGE (e)-[:BELONGS_TO]->(p)
                """,
                episode_id=episode_id,
                podcast_id=podcast_id
            )
            
            # Add hosts
            for host in episode.hosts:
                self._add_person_to_episode(session, host, episode_id, "host")
            
            # Add guests
            for guest in episode.guests:
                self._add_person_to_episode(session, guest, episode_id, "guest")
        
        logger.info(f"Added episode: {episode.title} ({episode_id})")
        return episode_id
    
    def _add_person_to_episode(
        self,
        session: Session,
        name: str,
        episode_id: str,
        role: str
    ) -> None:
        """Add a person and link to episode."""
        person_id = f"person_{name.lower().replace(' ', '_')}"
        
        session.run(
            """
            MERGE (p:Person {id: $id})
            ON CREATE SET p.name = $name
            WITH p
            MATCH (e:Episode {id: $episode_id})
            MERGE (p)-[r:APPEARED_ON]->(e)
            SET r.role = $role
            """,
            id=person_id,
            name=name,
            episode_id=episode_id,
            role=role
        )
    
    def add_entities_batch(
        self,
        video_id: str,
        entities: List[Entity]
    ) -> int:
        """
        Add multiple entities to the graph in batch.
        
        Args:
            video_id: The video ID to link entities to
            entities: List of Entity objects
            
        Returns:
            Number of entities added
        """
        episode_id = f"episode_{video_id}"
        count = 0
        
        with self.session() as session:
            for entity in entities:
                try:
                    self._add_entity(session, entity, episode_id)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to add entity {entity.value}: {e}")
        
        logger.info(f"Added {count} entities for video {video_id}")
        return count
    
    def _add_entity(
        self,
        session: Session,
        entity: Entity,
        episode_id: str
    ) -> None:
        """Add a single entity and its relationships."""
        entity_id = entity.generate_id()
        
        # Map entity type to Neo4j label
        label = entity.type.value.title()
        if entity.type == EntityType.MUSIC:
            label = "Music"
        
        # Create entity node based on type
        if entity.type == EntityType.PERSON:
            session.run(
                """
                MERGE (n:Person {id: $id})
                ON CREATE SET n.name = $value
                """,
                id=entity_id,
                value=entity.value
            )
            self._create_mentioned_in(session, entity_id, episode_id, entity)
            
        elif entity.type == EntityType.BOOK:
            # Parse book title and author
            title, author = self._parse_book_info(entity.value)
            session.run(
                """
                MERGE (n:Book {id: $id})
                ON CREATE SET n.title = $title, n.author = $author
                """,
                id=entity_id,
                title=title,
                author=author
            )
            # Create DISCUSSED_IN relationship
            recommended = entity.metadata.get('recommended', False)
            session.run(
                """
                MATCH (b:Book {id: $book_id})
                MATCH (e:Episode {id: $episode_id})
                MERGE (b)-[r:DISCUSSED_IN]->(e)
                SET r.timestamp = $timestamp, 
                    r.context = $context,
                    r.speaker = $speaker,
                    r.recommended = $recommended
                """,
                book_id=entity_id,
                episode_id=episode_id,
                timestamp=entity.timestamp,
                context=entity.context,
                speaker=entity.speaker,
                recommended=recommended
            )
            
            # If recommended and speaker known, create RECOMMENDED_BY
            if recommended and entity.speaker:
                speaker_id = f"person_{entity.speaker.lower().replace(' ', '_')}"
                session.run(
                    """
                    MATCH (b:Book {id: $book_id})
                    MATCH (p:Person {id: $person_id})
                    MERGE (b)-[:RECOMMENDED_BY]->(p)
                    """,
                    book_id=entity_id,
                    person_id=speaker_id
                )
        
        elif entity.type == EntityType.MOVIE:
            title, director = self._parse_movie_info(entity.value)
            session.run(
                """
                MERGE (n:Movie {id: $id})
                ON CREATE SET n.title = $title, n.director = $director
                """,
                id=entity_id,
                title=title,
                director=director
            )
            # Create REFERENCED_IN relationship
            session.run(
                """
                MATCH (m:Movie {id: $movie_id})
                MATCH (e:Episode {id: $episode_id})
                MERGE (m)-[r:REFERENCED_IN]->(e)
                SET r.timestamp = $timestamp, r.context = $context
                """,
                movie_id=entity_id,
                episode_id=episode_id,
                timestamp=entity.timestamp,
                context=entity.context
            )
        
        elif entity.type == EntityType.MUSIC:
            title, artist = self._parse_music_info(entity.value)
            session.run(
                """
                MERGE (n:Music {id: $id})
                ON CREATE SET n.title = $title, n.artist = $artist
                """,
                id=entity_id,
                title=title,
                artist=artist
            )
            self._create_mentioned_in(session, entity_id, episode_id, entity)
        
        elif entity.type == EntityType.COMPANY:
            session.run(
                """
                MERGE (n:Company {id: $id})
                ON CREATE SET n.name = $value
                """,
                id=entity_id,
                value=entity.value
            )
            # Include stock_discussed in relationship
            stock_discussed = entity.metadata.get('stock_discussed', False)
            session.run(
                """
                MATCH (c:Company {id: $company_id})
                MATCH (e:Episode {id: $episode_id})
                MERGE (c)-[r:MENTIONED_IN]->(e)
                SET r.timestamp = $timestamp, 
                    r.context = $context,
                    r.sentiment = $sentiment,
                    r.stock_discussed = $stock_discussed
                """,
                company_id=entity_id,
                episode_id=episode_id,
                timestamp=entity.timestamp,
                context=entity.context,
                sentiment=entity.sentiment.value,
                stock_discussed=stock_discussed
            )
        
        elif entity.type == EntityType.TOPIC:
            session.run(
                """
                MERGE (n:Topic {id: $id})
                ON CREATE SET n.name = $value
                """,
                id=entity_id,
                value=entity.value
            )
            # Create DISCUSSES relationship
            session.run(
                """
                MATCH (e:Episode {id: $episode_id})
                MATCH (t:Topic {id: $topic_id})
                MERGE (e)-[r:DISCUSSES]->(t)
                SET r.timestamp = $timestamp
                """,
                episode_id=episode_id,
                topic_id=entity_id,
                timestamp=entity.timestamp
            )
        
        elif entity.type in [EntityType.PRODUCT, EntityType.LOCATION]:
            label = entity.type.value.title()
            session.run(
                f"""
                MERGE (n:{label} {{id: $id}})
                ON CREATE SET n.name = $value
                """,
                id=entity_id,
                value=entity.value
            )
            self._create_mentioned_in(session, entity_id, episode_id, entity)
        
        elif entity.type == EntityType.QUOTE:
            # Quotes are stored as properties on relationships, not as nodes
            if entity.speaker:
                speaker_id = f"person_{entity.speaker.lower().replace(' ', '_')}"
                session.run(
                    """
                    MATCH (p:Person {id: $person_id})
                    MATCH (e:Episode {id: $episode_id})
                    MERGE (p)-[r:QUOTED_IN]->(e)
                    SET r.quote = $quote, r.timestamp = $timestamp, r.context = $context
                    """,
                    person_id=speaker_id,
                    episode_id=episode_id,
                    quote=entity.value,
                    timestamp=entity.timestamp,
                    context=entity.context
                )
    
    def _create_mentioned_in(
        self,
        session: Session,
        entity_id: str,
        episode_id: str,
        entity: Entity
    ) -> None:
        """Create a MENTIONED_IN relationship."""
        session.run(
            """
            MATCH (n {id: $entity_id})
            MATCH (e:Episode {id: $episode_id})
            MERGE (n)-[r:MENTIONED_IN]->(e)
            SET r.timestamp = $timestamp, 
                r.context = $context,
                r.sentiment = $sentiment
            """,
            entity_id=entity_id,
            episode_id=episode_id,
            timestamp=entity.timestamp,
            context=entity.context,
            sentiment=entity.sentiment.value
        )
    
    def _parse_book_info(self, value: str) -> Tuple[str, str]:
        """Parse book title and author from value."""
        # Try to parse "Title by Author" format
        if " by " in value.lower():
            parts = value.rsplit(" by ", 1)
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        return value, ""
    
    def _parse_movie_info(self, value: str) -> Tuple[str, str]:
        """Parse movie title and director from value."""
        # Try to parse "Title (Director: Name)" format
        import re
        match = re.match(r'(.+?)\s*\(Director:\s*(.+?)\)', value)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return value, ""
    
    def _parse_music_info(self, value: str) -> Tuple[str, str]:
        """Parse music title and artist from value."""
        if " by " in value.lower():
            parts = value.rsplit(" by ", 1)
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        return value, ""
    
    def add_cross_reference(
        self,
        person1: str,
        person2: str,
        context: str,
        timestamp: float = None
    ) -> None:
        """Add or update a cross-reference between two people."""
        person1_id = f"person_{person1.lower().replace(' ', '_')}"
        person2_id = f"person_{person2.lower().replace(' ', '_')}"
        
        with self.session() as session:
            # Update or create the REFERENCES relationship
            session.run(
                """
                MATCH (p1:Person {id: $person1_id})
                MATCH (p2:Person {id: $person2_id})
                MERGE (p1)-[r:REFERENCES]->(p2)
                ON CREATE SET r.count = 1, r.contexts = [$context]
                ON MATCH SET r.count = r.count + 1, 
                             r.contexts = r.contexts + $context
                """,
                person1_id=person1_id,
                person2_id=person2_id,
                context=context
            )
    
    def find_common_guests(
        self,
        podcast1: str,
        podcast2: str
    ) -> List[str]:
        """Find guests who appeared on both podcasts."""
        with self.session() as session:
            result = session.run(
                """
                MATCH (g:Person)-[:APPEARED_ON]->(e1:Episode)-[:BELONGS_TO]->(p1:Podcast)
                WHERE toLower(p1.name) CONTAINS toLower($podcast1)
                WITH g
                MATCH (g)-[:APPEARED_ON]->(e2:Episode)-[:BELONGS_TO]->(p2:Podcast)
                WHERE toLower(p2.name) CONTAINS toLower($podcast2)
                RETURN DISTINCT g.name as guest
                """,
                podcast1=podcast1,
                podcast2=podcast2
            )
            return [record["guest"] for record in result]
    
    def trace_concept_across_podcasts(
        self,
        concept: str,
        podcasts: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Trace a concept/topic across multiple podcasts chronologically."""
        with self.session() as session:
            if podcasts:
                result = session.run(
                    """
                    MATCH (e:Episode)-[r:DISCUSSES]->(t:Topic)
                    WHERE toLower(t.name) CONTAINS toLower($concept)
                    MATCH (e)-[:BELONGS_TO]->(p:Podcast)
                    WHERE p.name IN $podcasts
                    RETURN p.name as podcast, e.title as episode, 
                           e.publish_date as date, e.video_id as video_id,
                           r.timestamp as timestamp
                    ORDER BY e.publish_date, r.timestamp
                    """,
                    concept=concept,
                    podcasts=podcasts
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Episode)-[r:DISCUSSES]->(t:Topic)
                    WHERE toLower(t.name) CONTAINS toLower($concept)
                    MATCH (e)-[:BELONGS_TO]->(p:Podcast)
                    RETURN p.name as podcast, e.title as episode,
                           e.publish_date as date, e.video_id as video_id,
                           r.timestamp as timestamp
                    ORDER BY e.publish_date, r.timestamp
                    """,
                    concept=concept
                )
            
            return [dict(record) for record in result]
    
    def get_sentiment_timeline(
        self,
        entity: str,
        podcast: str = None
    ) -> List[Dict[str, Any]]:
        """Get sentiment timeline for an entity across episodes."""
        with self.session() as session:
            if podcast:
                result = session.run(
                    """
                    MATCH (n)-[r:MENTIONED_IN]->(e:Episode)-[:BELONGS_TO]->(p:Podcast)
                    WHERE toLower(n.name) CONTAINS toLower($entity)
                    AND toLower(p.name) CONTAINS toLower($podcast)
                    RETURN e.publish_date as date, r.sentiment as sentiment,
                           r.context as context, e.title as episode
                    ORDER BY e.publish_date
                    """,
                    entity=entity,
                    podcast=podcast
                )
            else:
                result = session.run(
                    """
                    MATCH (n)-[r:MENTIONED_IN]->(e:Episode)
                    WHERE toLower(n.name) CONTAINS toLower($entity)
                    RETURN e.publish_date as date, r.sentiment as sentiment,
                           r.context as context, e.title as episode
                    ORDER BY e.publish_date
                    """,
                    entity=entity
                )
            
            return [dict(record) for record in result]
    
    def verify_entity_exists(
        self,
        entity_name: str,
        entity_type: str = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if an entity exists in the graph."""
        with self.session() as session:
            if entity_type:
                result = session.run(
                    f"""
                    MATCH (n:{entity_type})
                    WHERE toLower(n.name) = toLower($name)
                       OR toLower(n.title) = toLower($name)
                    RETURN n, labels(n) as labels
                    LIMIT 1
                    """,
                    name=entity_name
                )
            else:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE toLower(n.name) = toLower($name)
                       OR toLower(n.title) = toLower($name)
                    RETURN n, labels(n) as labels
                    LIMIT 1
                    """,
                    name=entity_name
                )
            
            record = result.single()
            if record:
                return True, record["labels"][0] if record["labels"] else None
            return False, None
    
    def verify_relationship_exists(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> bool:
        """Check if a relationship exists between two entities."""
        with self.session() as session:
            result = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE (toLower(a.name) CONTAINS toLower($subject)
                       OR toLower(a.title) CONTAINS toLower($subject))
                AND (toLower(b.name) CONTAINS toLower($object)
                     OR toLower(b.title) CONTAINS toLower($object))
                AND type(r) = $predicate
                RETURN count(r) > 0 as exists
                """,
                subject=subject,
                predicate=predicate.upper(),
                object=obj
            )
            record = result.single()
            return record["exists"] if record else False
    
    def execute_cypher(
        self,
        cypher: str,
        parameters: Dict[str, Any] = None
    ) -> CypherResult:
        """Execute a raw Cypher query."""
        parameters = parameters or {}
        
        with self.session() as session:
            try:
                result = session.run(cypher, **parameters)
                records = [dict(record) for record in result]
                summary = result.consume()
                
                return CypherResult(
                    records=records,
                    summary={
                        "counters": summary.counters.__dict__ if summary.counters else {},
                        "query_type": summary.query_type
                    }
                )
            except Exception as e:
                logger.error(f"Cypher execution error: {e}")
                return CypherResult(error=str(e))
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        stats = {}
        
        with self.session() as session:
            # Count nodes by label
            for label in ["Episode", "Podcast", "Person", "Book", "Movie", 
                         "Music", "Company", "Product", "Location", "Topic"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                record = result.single()
                stats[label.lower() + "_count"] = record["count"] if record else 0
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            record = result.single()
            stats["relationship_count"] = record["count"] if record else 0
        
        return stats
