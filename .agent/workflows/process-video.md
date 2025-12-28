---
description: Process a YouTube podcast video through the knowledge graph pipeline
---

# Process Podcast Video Workflow

This workflow guides you through processing a YouTube video and adding it to the knowledge graph.

## Prerequisites

1. Ensure Neo4j is running:
   ```bash
   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourpassword neo4j:latest
   ```

2. Ensure `.env` is configured with your API keys.

## Steps

// turbo-all

1. Activate the virtual environment:
   ```bash
   cd /Users/omerfarukballi/Desktop/Dosyalar/Case\ LLM/podcast-kg
   source venv/bin/activate
   ```

2. Process a single video:
   ```bash
   python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --podcast "Podcast Name" --hosts "Host1,Host2"
   ```

3. Check processing status:
   ```bash
   python main.py --stats
   ```

4. Query the knowledge graph:
   ```bash
   python main.py --query "List all books mentioned in podcasts"
   ```

5. Launch the Streamlit UI:
   ```bash
   streamlit run ui/app.py
   ```

## Batch Processing

For processing multiple videos:

1. Edit `data/videos.json` with your video configurations
2. Run batch processing:
   ```bash
   python main.py --batch data/videos.json
   ```

## Running Tests

```bash
pytest tests/test_uat.py -v
```
