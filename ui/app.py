"""
Streamlit UI for Podcast Knowledge Graph System

A professional interface for:
- Processing YouTube podcast videos
- Querying the knowledge graph
- Visualizing entities and relationships
- Viewing citations with timestamps
"""

import streamlit as st
import asyncio
import time
import json
from pathlib import Path
import sys
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import PodcastKnowledgeSystem, parse_video_url
from ui.components import (
    render_graph_visualization,
    render_results_table,
    render_sources_list,
    render_statistics_cards
)

# Page configuration
st.set_page_config(
    page_title="Podcast Knowledge Graph",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #2d2d44 0%, #1f1f35 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Query input styling */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        font-size: 1.1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Results card */
    .result-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }
    
    /* Source citation */
    .source-citation {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Example button */
    .example-btn {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-btn:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #1a1a2e;
    }
    
    /* Success/Error messages */
    .success-msg {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #22c55e;
    }
    
    .error-msg {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)


# Initialize system
@st.cache_resource
def init_system():
    """Initialize the knowledge graph system."""
    try:
        return PodcastKnowledgeSystem(auto_connect=True)
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None


# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "processing" not in st.session_state:
    st.session_state.processing = False


def set_query(query: str):
    """Set the current query."""
    st.session_state.current_query = query


# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🎙️ Podcast Knowledge Graph</h1>', unsafe_allow_html=True)
    st.markdown("*AI-powered podcast analysis with knowledge graph and semantic search*")
    
    # Initialize system
    system = init_system()
    
    # Sidebar
    with st.sidebar:
        st.header("📥 Data Ingestion")
        
        # Video processing
        st.subheader("Process Video")
        video_input = st.text_input(
            "YouTube URL or Video ID",
            placeholder="https://youtube.com/watch?v=... or video_id"
        )
        podcast_name = st.text_input("Podcast Name", placeholder="e.g., All-In, Dwarkesh Patel")
        
        col1, col2 = st.columns(2)
        with col1:
            hosts = st.text_input("Hosts", placeholder="Host1, Host2")
        with col2:
            guests = st.text_input("Guests", placeholder="Guest1, Guest2")
        
        if st.button("🚀 Process Video", use_container_width=True):
            if video_input and podcast_name and system:
                try:
                    video_id = parse_video_url(video_input)
                    
                    with st.spinner("Processing video... This may take a few minutes."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(step, progress, message):
                            progress_bar.progress(progress)
                            status_text.text(f"{step}: {message}")
                        
                        result = asyncio.run(system.process_video(
                            video_id=video_id,
                            podcast_name=podcast_name,
                            hosts=[h.strip() for h in hosts.split(",")] if hosts else [],
                            guests=[g.strip() for g in guests.split(",")] if guests else [],
                            progress_callback=update_progress
                        ))
                        
                        if result.get("status") == "success":
                            st.success(f"✅ Processed: {result.get('entity_count', 0)} entities extracted!")
                        else:
                            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a video URL and podcast name")
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Database Stats")
        if system:
            stats = system.get_statistics()
            graph_stats = stats.get("graph", {})
            vector_stats = stats.get("vectors", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Episodes", graph_stats.get("episode_count", 0))
                st.metric("People", graph_stats.get("person_count", 0))
            with col2:
                st.metric("Topics", graph_stats.get("topic_count", 0))
                st.metric("Chunks", vector_stats.get("total_chunks", 0))
            
            with st.expander("View All Stats"):
                st.json(stats)
        
        st.divider()
        
        # Batch processing
        st.subheader("📁 Batch Processing")
        uploaded_file = st.file_uploader("Upload videos.json", type="json")
        if uploaded_file and st.button("Process Batch"):
            try:
                videos = json.load(uploaded_file)
                with st.spinner(f"Processing {len(videos)} videos..."):
                    results = asyncio.run(system.batch_process(videos))
                    success_count = sum(1 for r in results if r.get("status") == "success")
                    st.success(f"Processed {success_count}/{len(videos)} videos successfully")
            except Exception as e:
                st.error(f"Batch processing failed: {e}")
    
    # Main content area
    # Example queries
    st.subheader("💡 Try These Queries")
    
    example_queries = [
        "List all books recommended in podcasts",
        "Who are the common guests across podcasts?",
        "Find discussions about AI and technology",
        "What topics were discussed the most?",
        "Did anyone discuss Bitcoin or cryptocurrency?",
        "Verify: Did Lex Fridman interview Satoshi Nakamoto?",
        "What companies were mentioned positively?",
        "Find quotes about entrepreneurship"
    ]
    
    # Create columns for example buttons
    cols = st.columns(4)
    for i, example in enumerate(example_queries):
        with cols[i % 4]:
            if st.button(example[:35] + "..." if len(example) > 35 else example, 
                        key=f"example_{i}", 
                        use_container_width=True):
                set_query(example)
                st.rerun()
    
    st.divider()
    
    # Query input
    query = st.text_input(
        "🔍 Ask a question about the podcasts",
        value=st.session_state.current_query,
        placeholder="e.g., List all books recommended by David Senra, excluding Steve Jobs biographies",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.current_query = ""
            st.rerun()
    
    # Execute query
    if (search_clicked or query != st.session_state.current_query) and query and system:
        st.session_state.current_query = query
        
        with st.spinner("Searching..."):
            start_time = time.time()
            result = system.query(query)
            latency = time.time() - start_time
            
            # Add to history
            st.session_state.query_history.insert(0, {
                "query": query,
                "result": result,
                "time": latency
            })
            st.session_state.query_history = st.session_state.query_history[:10]  # Keep last 10
        
        # Display results
        st.divider()
        
        # Result header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Query Type", result.get("type", "unknown").upper())
        with col2:
            st.metric("Query Time", f"{latency:.2f}s")
        with col3:
            source_count = len(result.get("sources", []))
            st.metric("Sources", source_count)
        with col4:
            verified = result.get("verified")
            if verified is True:
                st.metric("Verified", "✅ Yes")
            elif verified is False:
                st.metric("Verified", "❌ No")
            else:
                st.metric("Verified", "—")
        
        # Answer
        st.subheader("📝 Answer")
        st.markdown(f"""
        <div class="result-card">
            {result.get("answer", "No answer available")}
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for additional information
        tab1, tab2, tab3, tab4 = st.tabs(["📚 Sources", "📊 Results", "🔍 Query Details", "📈 Graph"])
        
        with tab1:
            sources = result.get("sources", [])
            if sources:
                render_sources_list(sources)
            else:
                st.info("No source citations available for this query.")
        
        with tab2:
            results = result.get("results", [])
            if results:
                render_results_table(results)
            else:
                st.info("No structured results available.")
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Query Classification**")
                st.code(result.get("type", "unknown"), language=None)
                
                if result.get("cypher_query"):
                    st.markdown("**Generated Cypher Query**")
                    st.code(result.get("cypher_query"), language="cypher")
            
            with col2:
                st.markdown("**Execution Details**")
                st.json({
                    "execution_time": f"{result.get('execution_time', 0):.3f}s",
                    "verified": result.get("verified"),
                    "confidence": result.get("confidence", "N/A"),
                    "source_count": len(sources),
                    "result_count": len(results)
                })
        
        with tab4:
            if system and results:
                render_graph_visualization(results, system)
            else:
                st.info("Graph visualization will appear when results contain entity relationships.")
    
    # Query history
    if st.session_state.query_history:
        with st.expander("📜 Query History"):
            for i, item in enumerate(st.session_state.query_history):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"🔍 {item['query'][:60]}...", key=f"history_{i}"):
                        set_query(item['query'])
                        st.rerun()
                with col2:
                    st.caption(f"{item['time']:.2f}s")


if __name__ == "__main__":
    main()
