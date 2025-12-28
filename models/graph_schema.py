"""
Graph schema definitions for the Podcast Knowledge Graph.
Defines node types, relationship types, and Cypher query structures.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    PERSON = "Person"
    BOOK = "Book"
    MOVIE = "Movie"
    MUSIC = "Music"
    COMPANY = "Company"
    PRODUCT = "Product"
    LOCATION = "Location"
    TOPIC = "Topic"
    EPISODE = "Episode"
    PODCAST = "Podcast"


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    APPEARED_ON = "APPEARED_ON"
    MENTIONED_IN = "MENTIONED_IN"
    DISCUSSED_IN = "DISCUSSED_IN"
    RECOMMENDED_BY = "RECOMMENDED_BY"
    REFERENCED_IN = "REFERENCED_IN"
    DISCUSSES = "DISCUSSES"
    BELONGS_TO = "BELONGS_TO"
    REFERENCES = "REFERENCES"


class NodeSchema(BaseModel):
    """Schema definition for a graph node."""
    type: NodeType
    properties: Dict[str, Any]
    
    def to_cypher_create(self) -> str:
        """Generate Cypher CREATE statement for this node."""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        return f"CREATE (n:{self.type.value} {{{props_str}}})"
    
    def to_cypher_merge(self) -> str:
        """Generate Cypher MERGE statement for this node."""
        id_prop = self.properties.get("id", "")
        props_str = ", ".join([f"n.{k} = ${k}" for k in self.properties.keys() if k != "id"])
        return f"MERGE (n:{self.type.value} {{id: $id}}) ON CREATE SET {props_str} ON MATCH SET {props_str}"


class RelationshipSchema(BaseModel):
    """Schema definition for a graph relationship."""
    type: RelationshipType
    from_node_type: NodeType
    to_node_type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    def to_cypher_create(self) -> str:
        """Generate Cypher for creating this relationship."""
        props_str = ""
        if self.properties:
            props_str = " {" + ", ".join([f"{k}: ${k}" for k in self.properties.keys()]) + "}"
        return f"MATCH (a:{self.from_node_type.value}), (b:{self.to_node_type.value}) " \
               f"WHERE a.id = $from_id AND b.id = $to_id " \
               f"MERGE (a)-[r:{self.type.value}{props_str}]->(b)"


class GraphQuery(BaseModel):
    """Represents a graph query to be executed."""
    cypher: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    
    def with_params(self, **kwargs) -> "GraphQuery":
        """Return a copy of this query with additional parameters."""
        new_params = {**self.parameters, **kwargs}
        return GraphQuery(
            cypher=self.cypher,
            parameters=new_params,
            description=self.description
        )


class CypherResult(BaseModel):
    """Result from a Cypher query execution."""
    records: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if the query was successful."""
        return self.error is None
    
    @property
    def count(self) -> int:
        """Get the number of records returned."""
        return len(self.records)


# Predefined Cypher queries for common operations
class PredefinedQueries:
    """Collection of predefined Cypher queries."""
    
    # Schema creation
    CREATE_CONSTRAINTS = """
    CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;
    CREATE CONSTRAINT book_id IF NOT EXISTS FOR (b:Book) REQUIRE b.id IS UNIQUE;
    CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE;
    CREATE CONSTRAINT music_id IF NOT EXISTS FOR (m:Music) REQUIRE m.id IS UNIQUE;
    CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE;
    CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE;
    CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE;
    CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE;
    CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE;
    CREATE CONSTRAINT podcast_id IF NOT EXISTS FOR (p:Podcast) REQUIRE p.id IS UNIQUE;
    """
    
    CREATE_INDEXES = """
    CREATE INDEX episode_date IF NOT EXISTS FOR (e:Episode) ON (e.publish_date);
    CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name);
    CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
    CREATE INDEX episode_video IF NOT EXISTS FOR (e:Episode) ON (e.video_id);
    """
    
    # Node operations
    MERGE_PERSON = """
    MERGE (p:Person {id: $id})
    ON CREATE SET p.name = $name
    ON MATCH SET p.name = $name
    RETURN p
    """
    
    MERGE_BOOK = """
    MERGE (b:Book {id: $id})
    ON CREATE SET b.title = $title, b.author = $author
    ON MATCH SET b.title = $title, b.author = $author
    RETURN b
    """
    
    MERGE_EPISODE = """
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
    RETURN e
    """
    
    MERGE_PODCAST = """
    MERGE (p:Podcast {id: $id})
    ON CREATE SET p.name = $name
    RETURN p
    """
    
    # Relationship operations
    CREATE_APPEARED_ON = """
    MATCH (p:Person {id: $person_id})
    MATCH (e:Episode {id: $episode_id})
    MERGE (p)-[r:APPEARED_ON]->(e)
    SET r.role = $role
    RETURN r
    """
    
    CREATE_MENTIONED_IN = """
    MATCH (n {id: $entity_id})
    MATCH (e:Episode {id: $episode_id})
    MERGE (n)-[r:MENTIONED_IN]->(e)
    SET r.timestamp = $timestamp, r.context = $context, r.sentiment = $sentiment
    RETURN r
    """
    
    CREATE_DISCUSSED_IN = """
    MATCH (b:Book {id: $book_id})
    MATCH (e:Episode {id: $episode_id})
    MERGE (b)-[r:DISCUSSED_IN]->(e)
    SET r.timestamp = $timestamp, r.context = $context, r.speaker = $speaker, r.recommended = $recommended
    RETURN r
    """
    
    CREATE_BELONGS_TO = """
    MATCH (e:Episode {id: $episode_id})
    MATCH (p:Podcast {id: $podcast_id})
    MERGE (e)-[r:BELONGS_TO]->(p)
    RETURN r
    """
    
    # Query operations
    FIND_COMMON_GUESTS = """
    MATCH (g:Person)-[:APPEARED_ON]->(e1:Episode)-[:BELONGS_TO]->(p1:Podcast {name: $podcast1})
    MATCH (g)-[:APPEARED_ON]->(e2:Episode)-[:BELONGS_TO]->(p2:Podcast {name: $podcast2})
    WHERE g.name <> p1.name AND g.name <> p2.name
    RETURN DISTINCT g.name as guest
    """
    
    FIND_BOOKS_BY_RECOMMENDER = """
    MATCH (p:Person {name: $name})-[:RECOMMENDED_BY]-(b:Book)
    RETURN b.title as title, b.author as author
    """
    
    FIND_BOOKS_DISCUSSED_IN_EPISODE = """
    MATCH (b:Book)-[r:DISCUSSED_IN]->(e:Episode {video_id: $video_id})
    RETURN b.title as title, b.author as author, r.speaker as speaker, r.recommended as recommended
    """
    
    ENTITY_EXISTS = """
    MATCH (n {name: $name})
    RETURN count(n) > 0 as exists, labels(n) as labels
    """
    
    RELATIONSHIP_EXISTS = """
    MATCH (a {name: $subject})-[r]->(b {name: $object})
    RETURN type(r) as relationship, r as properties
    """
    
    PERSON_APPEARED_ON_PODCAST = """
    MATCH (p:Person {name: $person_name})-[:APPEARED_ON]->(e:Episode)-[:BELONGS_TO]->(pod:Podcast {name: $podcast_name})
    RETURN e.title as episode, e.publish_date as date, e.video_id as video_id
    """
    
    SENTIMENT_TIMELINE = """
    MATCH (n {name: $entity_name})-[r:MENTIONED_IN]->(e:Episode)-[:BELONGS_TO]->(p:Podcast {name: $podcast_name})
    RETURN e.publish_date as date, r.sentiment as sentiment, r.context as context
    ORDER BY e.publish_date
    """
    
    COUNT_CROSS_REFERENCES = """
    MATCH (p1:Person {name: $person1})-[r:REFERENCES]->(p2:Person)
    RETURN p2.name as referenced_person, r.count as count
    ORDER BY r.count DESC
    """
    
    TRACE_CONCEPT = """
    MATCH (e:Episode)-[r:DISCUSSES]->(t:Topic {name: $topic})
    MATCH (e)-[:BELONGS_TO]->(p:Podcast)
    WHERE p.name IN $podcasts
    RETURN p.name as podcast, e.title as episode, e.publish_date as date, r.timestamp as timestamp
    ORDER BY e.publish_date
    """
    
    FIRST_MENTION = """
    MATCH (n {name: $entity_name})-[r:MENTIONED_IN]->(e:Episode)
    RETURN e.title as episode, e.video_id as video_id, e.publish_date as date, r.timestamp as timestamp
    ORDER BY e.publish_date, r.timestamp
    LIMIT 1
    """
