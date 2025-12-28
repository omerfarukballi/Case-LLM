"""
Query Engine Service for the Podcast Knowledge Graph System.
Handles hybrid queries combining graph and semantic search with hallucination prevention.
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Literal
from enum import Enum
import logging

from openai import AsyncOpenAI

from models.entities import QueryResult, VerificationResult
from services.graph_builder import GraphBuilder
from services.vector_store import VectorStore
from config import get_settings, get_logger, CYPHER_SCHEMA_STRING

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Types of queries the engine can handle."""
    GRAPH = "graph"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    VERIFY = "verify"


# Intent classification prompt
INTENT_CLASSIFICATION_PROMPT = """You are a query intent classifier for a podcast knowledge graph system.

Classify the following query into one of these categories:
1. GRAPH - Queries about relationships, lists, connections, counts (e.g., "Who appeared on podcast X?", "List all books recommended by Y", "Common guests between podcasts")
2. SEMANTIC - Queries requiring finding specific content, quotes, explanations (e.g., "What did X say about Y?", "Find the quote about Z", "Explain the discussion about W")
3. HYBRID - Queries requiring both graph structure and semantic content (e.g., "Trace concept X across podcasts", "Compare views of A and B on topic C", "How has sentiment on X changed?")
4. VERIFY - Queries checking if something is true, asking to verify claims (e.g., "Did X interview Y?", "Verify if X said Z", "Is it true that...")

Query: "{query}"

Respond with ONLY one word: GRAPH, SEMANTIC, HYBRID, or VERIFY
"""

# Cypher generation prompt
CYPHER_GENERATION_PROMPT = """Convert this natural language query to a Cypher query for Neo4j.

Graph Schema:
{schema}

Query: "{query}"

Rules:
1. Use MATCH for finding patterns
2. Use WHERE for filtering with toLower() for case-insensitive search
3. Use RETURN for results
4. For "common guests" use intersection pattern
5. For "sentiment over time" ORDER BY e.publish_date
6. Include LIMIT clause when appropriate
7. Use CONTAINS for partial name matching

Examples:
- "List all books recommended by David Senra" → 
  MATCH (b:Book)-[:RECOMMENDED_BY]->(p:Person)
  WHERE toLower(p.name) CONTAINS toLower("David Senra")
  RETURN b.title as title, b.author as author

- "Who appeared on both Dwarkesh and Lex podcasts?" →
  MATCH (g:Person)-[:APPEARED_ON]->(e1:Episode)-[:BELONGS_TO]->(p1:Podcast)
  WHERE toLower(p1.name) CONTAINS "dwarkesh"
  WITH g
  MATCH (g)-[:APPEARED_ON]->(e2:Episode)-[:BELONGS_TO]->(p2:Podcast)
  WHERE toLower(p2.name) CONTAINS "lex"
  RETURN DISTINCT g.name as guest

- "What topics were discussed in video X?" →
  MATCH (e:Episode {{video_id: "X"}})-[:DISCUSSES]->(t:Topic)
  RETURN t.name as topic, e.title as episode

Return ONLY the Cypher query, no explanation. If you cannot convert the query, return "CANNOT_CONVERT".
"""

# Answer synthesis prompt
ANSWER_SYNTHESIS_PROMPT = """You are an assistant that synthesizes answers from podcast knowledge graph data.

Query: "{query}"

Context from sources:
{context}

Rules:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Include specific details like timestamps, speakers, and episode names when available
4. DO NOT make up or hallucinate information not in the context
5. For verification queries, clearly state if evidence was found or not
6. If multiple sources agree/disagree, mention that

Provide a clear, concise answer:
"""

# Claim verification prompt
CLAIM_VERIFICATION_PROMPT = """You are a fact-checker for a podcast knowledge graph. Your job is to verify claims against available evidence.

Claim to verify: "{claim}"

Available evidence from the knowledge graph:
{evidence}

Graph verification results:
- Subject exists: {subject_exists}
- Object exists: {object_exists}
- Relationship exists: {relationship_exists}

Instructions:
1. Carefully analyze if the claim is supported by the evidence
2. Look for direct contradictions
3. Consider if absence of evidence means the claim is false
4. Be conservative - if unsure, say "Cannot verify"

Respond in this JSON format:
{{
    "verified": true/false/null,
    "confidence": 0.0-1.0,
    "reason": "Explanation of why claim is verified/refuted/uncertain",
    "supporting_evidence": ["List of evidence supporting the claim"],
    "contradicting_evidence": ["List of evidence contradicting the claim"]
}}
"""


class QueryEngine:
    """
    Hybrid query engine combining graph and semantic search.
    
    Features:
    - Intent classification for query routing
    - Natural language to Cypher conversion
    - Graph-based queries
    - Semantic vector search
    - Hybrid queries combining both
    - Hallucination prevention through verification
    """
    
    def __init__(
        self,
        graph_builder: GraphBuilder,
        vector_store: VectorStore
    ):
        self.settings = get_settings()
        self.graph = graph_builder
        self.vectors = vector_store
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
    
    async def query(
        self,
        user_query: str,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for queries.
        
        Args:
            user_query: The user's natural language query
            filters: Optional filters (video_id, date_range, podcast, etc.)
            
        Returns:
            QueryResult dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        filters = filters or {}
        
        try:
            # Step 1: Classify query intent
            query_type = await self.classify_intent(user_query)
            logger.info(f"Query classified as: {query_type}")
            
            # Step 2: Route to appropriate handler
            if query_type == QueryType.GRAPH:
                result = await self.execute_graph_query(user_query)
            elif query_type == QueryType.SEMANTIC:
                result = await self.execute_semantic_query(user_query, filters)
            elif query_type == QueryType.HYBRID:
                result = await self.execute_hybrid_query(user_query, filters)
            elif query_type == QueryType.VERIFY:
                result = await self.verify_claim(user_query)
            else:
                result = await self.execute_hybrid_query(user_query, filters)
            
            # Add execution time
            result["execution_time"] = time.time() - start_time
            result["type"] = query_type.value if isinstance(query_type, QueryType) else query_type
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "query": user_query,
                "type": "error",
                "answer": f"An error occurred: {str(e)}",
                "results": [],
                "sources": [],
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def classify_intent(self, query: str) -> QueryType:
        """Classify the intent of a query."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Faster for simple classification
                messages=[
                    {"role": "system", "content": "You are a query classifier. Respond with only one word."},
                    {"role": "user", "content": INTENT_CLASSIFICATION_PROMPT.format(query=query)}
                ],
                temperature=0,
                max_tokens=10
            )
            
            intent = response.choices[0].message.content.strip().upper()
            
            # Map to QueryType
            intent_map = {
                "GRAPH": QueryType.GRAPH,
                "SEMANTIC": QueryType.SEMANTIC,
                "HYBRID": QueryType.HYBRID,
                "VERIFY": QueryType.VERIFY
            }
            
            return intent_map.get(intent, QueryType.HYBRID)
            
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}, defaulting to HYBRID")
            return QueryType.HYBRID
    
    async def execute_graph_query(self, query: str) -> Dict[str, Any]:
        """Execute a graph-based query."""
        # Generate Cypher query
        cypher = await self.generate_cypher(query)
        
        if cypher == "CANNOT_CONVERT" or not cypher:
            # Fallback to hybrid if can't convert
            return await self.execute_hybrid_query(query, {})
        
        # Execute Cypher
        result = self.graph.execute_cypher(cypher)
        
        if result.error:
            logger.warning(f"Cypher execution failed: {result.error}")
            # Try hybrid as fallback
            return await self.execute_hybrid_query(query, {})
        
        # Format results
        records = result.records
        
        # Synthesize answer from results
        answer = await self._synthesize_answer(query, str(records), [])
        
        return {
            "query": query,
            "answer": answer,
            "results": records,
            "sources": self._extract_sources_from_records(records),
            "cypher_query": cypher,
            "verified": True  # Graph results are trusted
        }
    
    async def execute_semantic_query(
        self,
        query: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a semantic search query."""
        # Build metadata filter
        metadata_filter = {}
        if filters.get("video_id"):
            metadata_filter["video_id"] = filters["video_id"]
        if filters.get("podcast"):
            metadata_filter["podcast_name"] = filters["podcast"]
        
        # Search vector store
        if filters.get("start_date") and filters.get("end_date"):
            search_results = self.vectors.search_by_timerange(
                query=query,
                start_date=filters["start_date"],
                end_date=filters["end_date"],
                n_results=10
            )
        else:
            search_results = self.vectors.search(
                query=query,
                n_results=10,
                filter_metadata=metadata_filter if metadata_filter else None
            )
        
        # Build context from results
        context_parts = []
        sources = []
        
        for result in search_results.get("results", []):
            metadata = result.get("metadata", {})
            text = result.get("text", "")
            
            context_parts.append(
                f"[{metadata.get('podcast_name', 'Unknown')} - {metadata.get('start_time', 0):.1f}s]: {text}"
            )
            
            sources.append({
                "video_id": metadata.get("video_id", ""),
                "podcast": metadata.get("podcast_name", ""),
                "start_time": metadata.get("start_time", 0),
                "end_time": metadata.get("end_time", 0),
                "speaker": metadata.get("speaker", ""),
                "text": text[:200] + "..." if len(text) > 200 else text,
                "similarity": result.get("similarity", 0)
            })
        
        context = "\n\n".join(context_parts)
        
        # Synthesize answer
        answer = await self._synthesize_answer(query, context, sources)
        
        return {
            "query": query,
            "answer": answer,
            "results": search_results.get("results", []),
            "sources": sources,
            "verified": False  # Semantic results need verification
        }
    
    async def execute_hybrid_query(
        self,
        query: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a hybrid query combining graph and semantic search."""
        # Execute both query types in parallel
        graph_task = self._safe_graph_query(query)
        semantic_task = self.execute_semantic_query(query, filters)
        
        graph_result, semantic_result = await asyncio.gather(
            graph_task, semantic_task, return_exceptions=True
        )
        
        # Combine results
        combined_context = []
        combined_sources = []
        
        # Add graph results if available
        if isinstance(graph_result, dict) and graph_result.get("results"):
            for record in graph_result["results"][:5]:
                combined_context.append(f"[Graph]: {str(record)}")
            combined_sources.extend(graph_result.get("sources", []))
        
        # Add semantic results
        if isinstance(semantic_result, dict):
            for source in semantic_result.get("sources", [])[:5]:
                combined_context.append(
                    f"[{source.get('podcast', 'Unknown')} @ {source.get('start_time', 0):.1f}s]: {source.get('text', '')}"
                )
            combined_sources.extend(semantic_result.get("sources", []))
        
        context = "\n\n".join(combined_context)
        
        # Synthesize combined answer
        answer = await self._synthesize_answer(query, context, combined_sources)
        
        return {
            "query": query,
            "answer": answer,
            "results": (
                graph_result.get("results", []) if isinstance(graph_result, dict) else []
            ) + (
                semantic_result.get("results", []) if isinstance(semantic_result, dict) else []
            ),
            "sources": combined_sources,
            "cypher_query": graph_result.get("cypher_query") if isinstance(graph_result, dict) else None,
            "verified": False
        }
    
    async def verify_claim(self, query: str) -> Dict[str, Any]:
        """
        Verify a claim against the knowledge graph.
        
        This is the hallucination prevention system.
        """
        # Parse the claim to extract subject, predicate, object
        claim_parts = await self._parse_claim(query)
        
        subject = claim_parts.get("subject", "")
        predicate = claim_parts.get("predicate", "")
        obj = claim_parts.get("object", "")
        
        # Verification checks
        subject_exists = False
        object_exists = False
        relationship_exists = False
        evidence = []
        
        # Check if subject exists
        if subject:
            exists, label = self.graph.verify_entity_exists(subject)
            subject_exists = exists
            if exists:
                evidence.append(f"'{subject}' exists in the knowledge graph as {label}")
        
        # Check if object exists
        if obj:
            exists, label = self.graph.verify_entity_exists(obj)
            object_exists = exists
            if exists:
                evidence.append(f"'{obj}' exists in the knowledge graph as {label}")
        
        # Check if relationship exists
        if subject and obj and predicate:
            relationship_exists = self.graph.verify_relationship_exists(
                subject, predicate, obj
            )
            if relationship_exists:
                evidence.append(f"Relationship between '{subject}' and '{obj}' exists")
        
        # Search for semantic evidence
        semantic_results = self.vectors.search(query, n_results=5)
        for result in semantic_results.get("results", []):
            evidence.append(f"Semantic match: {result.get('text', '')[:150]}...")
        
        # Use LLM to synthesize verification
        verification = await self._verify_with_llm(
            claim=query,
            evidence=evidence,
            subject_exists=subject_exists,
            object_exists=object_exists,
            relationship_exists=relationship_exists
        )
        
        # Build appropriate answer
        if verification["verified"] is False:
            if not subject_exists and subject:
                answer = f"No record found. '{subject}' does not appear in the knowledge graph."
            elif not object_exists and obj:
                answer = f"No record found. '{obj}' does not appear in the knowledge graph."
            elif not relationship_exists:
                answer = f"No evidence found to support this claim. {verification.get('reason', '')}"
            else:
                answer = verification.get("reason", "Cannot verify this claim.")
        elif verification["verified"] is True:
            answer = f"Verified. {verification.get('reason', '')}"
        else:
            answer = f"Cannot verify. {verification.get('reason', 'Insufficient evidence.')}"
        
        return {
            "query": query,
            "type": "verify",
            "answer": answer,
            "verified": verification["verified"],
            "confidence": verification["confidence"],
            "reason": verification.get("reason", ""),
            "evidence": evidence,
            "sources": [
                {
                    "text": result.get("text", ""),
                    "video_id": result.get("metadata", {}).get("video_id", ""),
                    "start_time": result.get("metadata", {}).get("start_time", 0)
                }
                for result in semantic_results.get("results", [])
            ],
            "results": []
        }
    
    async def _parse_claim(self, claim: str) -> Dict[str, str]:
        """Parse a claim into subject, predicate, object structure."""
        prompt = f"""Parse this claim/question into its components:
        
Claim: "{claim}"

Extract:
- subject: The main entity (person, thing) being asked about
- predicate: The action/relationship (e.g., "interviewed", "said", "recommended")
- object: The secondary entity or concept

Respond in JSON format:
{{"subject": "...", "predicate": "...", "object": "..."}}

If a component is not clear, use an empty string.
"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {"subject": "", "predicate": "", "object": ""}
    
    async def _verify_with_llm(
        self,
        claim: str,
        evidence: List[str],
        subject_exists: bool,
        object_exists: bool,
        relationship_exists: bool
    ) -> Dict[str, Any]:
        """Use LLM to verify a claim."""
        prompt = CLAIM_VERIFICATION_PROMPT.format(
            claim=claim,
            evidence="\n".join(f"- {e}" for e in evidence) if evidence else "No evidence found.",
            subject_exists=subject_exists,
            object_exists=object_exists,
            relationship_exists=relationship_exists
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a careful fact-checker. Be conservative with verification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert string "true"/"false" to boolean if needed
            verified = result.get("verified")
            if isinstance(verified, str):
                verified = verified.lower() == "true"
            
            return {
                "verified": verified,
                "confidence": float(result.get("confidence", 0.5)),
                "reason": result.get("reason", ""),
                "supporting_evidence": result.get("supporting_evidence", []),
                "contradicting_evidence": result.get("contradicting_evidence", [])
            }
            
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return {
                "verified": None,
                "confidence": 0.0,
                "reason": f"Verification failed: {str(e)}"
            }
    
    async def generate_cypher(self, natural_language: str) -> str:
        """Convert natural language to Cypher query."""
        prompt = CYPHER_GENERATION_PROMPT.format(
            schema=CYPHER_SCHEMA_STRING,
            query=natural_language
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a Neo4j Cypher expert. Return only valid Cypher queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            cypher = response.choices[0].message.content.strip()
            
            # Clean up the response
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            
            # Validate basic structure
            if not any(keyword in cypher.upper() for keyword in ["MATCH", "RETURN", "CREATE"]):
                return "CANNOT_CONVERT"
            
            return cypher
            
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            return "CANNOT_CONVERT"
    
    async def _synthesize_answer(
        self,
        query: str,
        context: str,
        sources: List[Dict]
    ) -> str:
        """Synthesize a natural language answer from query results."""
        if not context or context.strip() == "[]":
            return "No relevant information found in the knowledge graph for this query."
        
        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            query=query,
            context=context
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.gpt_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on podcast knowledge graph data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"Found {len(sources)} relevant sources but could not synthesize answer."
    
    async def _safe_graph_query(self, query: str) -> Dict[str, Any]:
        """Execute graph query with error handling."""
        try:
            return await self.execute_graph_query(query)
        except Exception as e:
            logger.warning(f"Graph query failed: {e}")
            return {"results": [], "sources": []}
    
    def _extract_sources_from_records(
        self,
        records: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract source citations from graph records."""
        sources = []
        
        for record in records:
            source = {}
            for key, value in record.items():
                if "video_id" in key.lower():
                    source["video_id"] = value
                elif "episode" in key.lower():
                    source["episode"] = value
                elif "date" in key.lower():
                    source["date"] = value
                elif "timestamp" in key.lower():
                    source["start_time"] = value
            
            if source:
                sources.append(source)
        
        return sources
    
    # Specific query methods for UAT scenarios
    
    async def find_books_by_recommender(
        self,
        recommender: str,
        exclude_subject: str = None,
        year: str = None
    ) -> Dict[str, Any]:
        """Find books recommended by a person. (UAT-01)"""
        cypher = """
        MATCH (b:Book)-[:RECOMMENDED_BY]->(p:Person)
        WHERE toLower(p.name) CONTAINS toLower($name)
        """
        
        if exclude_subject:
            cypher += " AND NOT toLower(b.title) CONTAINS toLower($exclude)"
        
        cypher += " RETURN b.title as title, b.author as author"
        
        params = {"name": recommender}
        if exclude_subject:
            params["exclude"] = exclude_subject
        
        result = self.graph.execute_cypher(cypher, params)
        
        return {
            "query": f"Books recommended by {recommender}",
            "type": "graph",
            "results": result.records,
            "sources": []
        }
    
    async def find_common_guests(
        self,
        podcast1: str,
        podcast2: str,
        year: str = None
    ) -> Dict[str, Any]:
        """Find common guests between podcasts. (UAT-04)"""
        guests = self.graph.find_common_guests(podcast1, podcast2)
        
        return {
            "query": f"Common guests between {podcast1} and {podcast2}",
            "type": "graph",
            "answer": f"Found {len(guests)} common guests: {', '.join(guests)}" if guests else "No common guests found.",
            "results": [{"guest": g} for g in guests],
            "sources": []
        }
    
    async def get_sentiment_analysis(
        self,
        entity: str,
        podcast: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Get sentiment timeline for an entity. (UAT-05, UAT-30)"""
        timeline = self.graph.get_sentiment_timeline(entity, podcast)
        
        # Filter by date if specified
        if start_date or end_date:
            filtered = []
            for item in timeline:
                date = item.get("date", "")
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                filtered.append(item)
            timeline = filtered
        
        # Calculate sentiment shift if we have pre/post data
        pre_sentiment = []
        post_sentiment = []
        
        if start_date:
            for item in timeline:
                if item.get("date", "") < start_date:
                    pre_sentiment.append(item.get("sentiment"))
                else:
                    post_sentiment.append(item.get("sentiment"))
        
        return {
            "query": f"Sentiment for {entity}",
            "type": "graph",
            "timeline": timeline,
            "pre_sentiment": pre_sentiment,
            "post_sentiment": post_sentiment,
            "results": timeline,
            "sources": []
        }
    
    async def trace_concept(
        self,
        concept: str,
        podcasts: List[str] = None
    ) -> Dict[str, Any]:
        """Trace a concept across podcasts chronologically. (UAT-21)"""
        trace = self.graph.trace_concept_across_podcasts(concept, podcasts)
        
        return {
            "query": f"Trace '{concept}' across podcasts",
            "type": "hybrid",
            "timeline": trace,
            "results": trace,
            "sources": [
                {
                    "podcast": t.get("podcast"),
                    "episode": t.get("episode"),
                    "date": t.get("date"),
                    "video_id": t.get("video_id")
                }
                for t in trace
            ]
        }
    
    def verify_entity_exists(self, entity_name: str, entity_type: str = None) -> bool:
        """Check if an entity exists. (For hallucination prevention)"""
        exists, _ = self.graph.verify_entity_exists(entity_name, entity_type)
        return exists
    
    def verify_date_in_range(self, date: str, video_id: str) -> bool:
        """Verify if a date is within the episode's date range."""
        result = self.graph.execute_cypher(
            "MATCH (e:Episode {video_id: $video_id}) RETURN e.publish_date as date",
            {"video_id": video_id}
        )
        
        if result.records:
            episode_date = result.records[0].get("date", "")
            return date == episode_date
        return False
    
    def verify_speaker_exists(self, speaker: str, video_id: str) -> bool:
        """Verify if a speaker exists in an episode."""
        result = self.graph.execute_cypher(
            """
            MATCH (p:Person)-[:APPEARED_ON]->(e:Episode {video_id: $video_id})
            WHERE toLower(p.name) CONTAINS toLower($speaker)
            RETURN count(p) > 0 as exists
            """,
            {"video_id": video_id, "speaker": speaker}
        )
        
        if result.records:
            return result.records[0].get("exists", False)
        return False
