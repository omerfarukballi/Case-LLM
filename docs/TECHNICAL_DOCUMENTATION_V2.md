# Podcast Knowledge Graph System (GraphRAG) - Technical Architecture & Design Decisions

## 1. High-Level Architecture
The Podcast Knowledge Graph System is a **Local-First**, **Privacy-Preserving**, **GraphRAG (Retrieval-Augmented Generation)** pipeline designed to extract, structure, and query complex relationships from audio content.

It combines the semantic search capabilities of **Vector Databases** (ChromaDB) with the structured reasoning of **Graph Databases** (Neo4j) to answer complex questions that neither system can answer alone.

### 1.1 Core Pipeline
1.  **Ingestion:** YouTube Audio Download -> `FFmpeg` Conversion (16kHz Mono).
2.  **Transcription (Local):** `OpenAI Whisper` (Base/Small) running locally for speech-to-text.
3.  **Diarization (Hybrid):** Acoustic separation (Whisper segments) + Semantic Heuristics to identify "Host" vs "Guest".
4.  **Extraction (Local):** `Ollama (Mistral)` LLM to extract Entities (Person, Book, Company) and Relations.
5.  **Storage:**
    *   **Graph:** Neo4j (Entities & Relationships).
    *   **Vector:** ChromaDB (Text Chunks & Embeddings).
6.  **Querying:** Hybrid Search (Graph Traversal + Semantic Similarity).

---

## 2. Technical Interview Questions (Deep Dive)
Here are the specific answers to the architectural questions, reflecting the implemented codebase.

### Q1: How did you design the GraphDB structure? (Nodes, Edges & Strategy)
**Strategy:** Moved beyond simple "Named Entity Recognition" to a **Schema-First Approach**. Instead of generic labels, I modeled the domain-specific logic of podcasts (Recommendations, Mentions, Appearances).

**Node Types (The Nouns):**
*   `Person`: Hosts, Guests, mentioned figures (e.g., "Elon Musk").
*   `Podcast` & `Episode`: Hierarchical containers for content.
*   `Book`, `Movie`, `Company`, `Product`: Specialized entities to answer specific UAT queries (e.g., "Book recommendations").
*   `Topic`: Abstract concepts (e.g., "Artificial Intelligence", "Stoicism") for thematic linking.

**Edge Types (The Verbs):**
*   `APPEARED_ON`: Links `Person` -> `Episode`. Property: `role: "host" | "guest"`.
*   `RECOMMENDED_BY`: **Critical for "Recs" engine.** `Book` <- `Person`.
*   `DISCUSSED`: General mentions. Properties: `sentiment`, `timestamp`.
*   `MENTIONED_IN`: Connects entities to episodes with time markers.

**Cypher Implementation:**
```cypher
// Example: Finding who recommended a specific book
MATCH (p:Person)-[:RECOMMENDED_BY]->(b:Book) RETURN p.name, b.title
```

---

### Q2: How did you design the VectorDB? (Chunking & Strategy)
**Strategy:** Implemented **"Context-Aware Overlapping Chunking"** to solve the "Lost in the Middle" problem common in RAG systems.

**Chunking Logic:**
1.  **Size:** 2000 Tokens (Large enough to capture full thoughts/arguments).
2.  **Overlap:** 200 Tokens (10%). Ensures that if a sentence starts at the end of Chunk A, it completes in Chunk B.
3.  **Speaker-Awareness:** The chunking algorithm (`services/entity_extraction.py`) respects segment boundaries. It tries not to split a speaker's monologue in half arbitrarily.

**Metadata Enrichment:**
Each vector is not just text. It is injected with:
*   `video_id` & `timestamp` (For citation).
*   `speaker_name` (To filter "Show me what Elon said", not what was said *about* him).
*   `podcast_name` (For domain filtering).

---

### Q3: How will you solve Diarization?
**Strategy:** Adopted a **Hybrid approach** balancing cost, privacy, and accuracy.

1.  **Acoustic Separation (Whisper):** The `whisper` model provides segment-level timestamps. While the 'base' model doesn't strictly label "Speaker A/B", it gives us temporal segmentation.
2.  **Semantic Heuristics (The "Data Science" Touch):** I implemented a heuristic algorithms in `TranscriptionService`:
    *   **Host Identification:** The speaker appearing at `t=0` who says "Welcome to..." is tagged as Host.
    *   **Guest Identification:** The first *new* voice after the Host's introduction is tagged as Guest.
    *   **Context cues:** Detecting "I'm [Name]" phrases to map generic speaker labels to specific names.
*(Note: For production, we can plug in `pyannote.audio` for distinct speaker fingerprints, but the current heuristic fits the lightweight local constraint).*

---

### Q4: Which Embedding Model did you use and why?
**Selection:** `text-embedding-3-small` (OpenAI) OR `nomic-embed-text` (Local).

**Reasoning:**
1.  **Cost/Performance Ratio:** The `3-small` model beats the older Ada-002 on MTEB benchmarks while being significantly cheaper.
2.  **Dimensionality:** 1536 dimensions provide a dense enough representation for semantic nuance (differentiating "Apple" the fruit vs. company).
3.  **Local Alternative:** For the fully offline (Ollama) setup, `nomic-embed-text` is the state-of-the-art local embedding model, supporting long contexts (8k tokens).

---

### Q5: Ingestion, Pre-processing & Post-processing Strategy?
**Pre-processing (Cleaning):**
*   **Silence Removal:** Using `FFmpeg` to strip silence reduces token usage by ~10-15%.
*   **Format Normalization:** Converting all inputs (mp4, mkv, wav) to `16kHz mono mp3` ensures consistent model behavior.

**Post-processing (Refining):**
*   **Entity Deduplication (Resolution):** "Elon R. Musk" and "Elon Musk" are merged into a single Node using Fuzzy Matching (Levenshtein Distance) before writing to Neo4j.
*   **Hallucination Check:** Entities with `confidence < 0.7` (from the LLM extraction step) are discarded.
*   **Cross-Reference Detection:** A dedicated step checks if a Guest mentions an entity that appeared in *another* episode, creating "Second-Order" links in the graph.

---

### Q6: How do you address Performance + Cost concerns?
**This is the strongest part of the architecture:**

1.  **Zero-Cost Operation (Local Mode):**
    *   By switching to **Local Whisper** (Transcription) and **Ollama/Mistral** (Reasoning), we eliminated the $1.08/episode API cost.
    *   Privacy is guaranteed; audio never leaves the machine.
2.  **Caching (L1/L2):**
    *   **Audio Cache:** If a video ID is re-submitted, we check the hash on disk.
    *   **Transcript Cache:** Expensive transcription results are saved as JSON. Re-runs take milliseconds.
3.  **Hybrid Search Optimization:**
    *   Pure Vector Search is O(N). Graph traversal is O(1) for neighbors.
    *   We query the Graph *first* (e.g., "Find books by Naval") to get exact IDs, then perform Vector Search *only* on that subset. This drastically reduces the search space.

---

## 3. UAT Scenario Coverage (Examples)

| UAT Query | Architectural Feature Handling It |
| :--- | :--- |
| **"Books recommended by David Senra?"** | **Graph:** `(Person {name:'David Senra'})-[:RECOMMENDED_BY]->(b:Book)` |
| **"Sentiment of NVIDIA in 2025"** | **Edge Properties:** `[:MENTIONED_IN {sentiment: 'positive', year: 2025}]` |
| **"Common guests between Lex & All-In"** | **Graph Intersection:** `MATCH (p)-[:APPEARED_ON]->(p1), (p)-[:APPEARED_ON]->(p2)` |
| **"Clip where Joe Rogan says..."** | **Vector:** Semantic search over chunks filtered by `speaker='Joe Rogan'` |

## 4. Setup & Usage (Local Mode)
1.  **Prepare:** Ensure `Ollama` is running (`ollama serve`).
2.  **Config:** Set `USE_LOCAL_LLM=True` in `.env`.
3.  **Run:** `streamlit run ui/app.py`.
