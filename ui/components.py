"""
UI Components for Podcast Knowledge Graph Streamlit App

Reusable components for visualizations and data display.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json


def render_graph_visualization(results: List[Dict], system: Any = None):
    """
    Render an interactive graph visualization using NetworkX and Plotly.
    
    Args:
        results: Query results containing entity data
        system: The PodcastKnowledgeSystem for fetching additional data
    """
    try:
        import networkx as nx
        import plotly.graph_objects as go
        
        # Create graph
        G = nx.Graph()
        
        # Extract entities and relationships from results
        node_colors = {
            "Person": "#667eea",
            "Book": "#f59e0b",
            "Movie": "#ef4444",
            "Company": "#22c55e",
            "Topic": "#8b5cf6",
            "Episode": "#06b6d4",
            "Podcast": "#ec4899",
            "default": "#94a3b8"
        }
        
        nodes_added = set()
        edges = []
        
        for result in results:
            if isinstance(result, dict):
                # Try to extract graph-like data
                for key, value in result.items():
                    if value and isinstance(value, str):
                        node_type = key.title() if key in ["guest", "person", "book", "topic", "episode"] else "default"
                        
                        if value not in nodes_added:
                            G.add_node(value, type=node_type)
                            nodes_added.add(value)
        
        if len(G.nodes()) == 0:
            st.info("No graph data available to visualize.")
            return
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(255,255,255,0.3)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors_list = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_type = G.nodes[node].get('type', 'default')
            node_colors_list.append(node_colors.get(node_type, node_colors['default']))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            textfont=dict(size=10, color='white'),
            marker=dict(
                size=20,
                color=node_colors_list,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("**Node Types:**")
        legend_cols = st.columns(len(node_colors) - 1)
        for i, (node_type, color) in enumerate(list(node_colors.items())[:-1]):
            with legend_cols[i]:
                st.markdown(f"<span style='color:{color};'>●</span> {node_type}", unsafe_allow_html=True)
                
    except ImportError:
        st.warning("Install networkx and plotly for graph visualization: pip install networkx plotly")
    except Exception as e:
        st.error(f"Graph visualization error: {e}")


def render_results_table(results: List[Dict]):
    """
    Render query results as a formatted table.
    
    Args:
        results: List of result dictionaries
    """
    if not results:
        st.info("No results to display.")
        return
    
    # Try to create a DataFrame
    try:
        import pandas as pd
        
        # Flatten nested dicts
        flat_results = []
        for result in results:
            if isinstance(result, dict):
                flat_result = {}
                for key, value in result.items():
                    if isinstance(value, (dict, list)):
                        flat_result[key] = json.dumps(value)[:100]
                    else:
                        flat_result[key] = value
                flat_results.append(flat_result)
            else:
                flat_results.append({"value": result})
        
        df = pd.DataFrame(flat_results)
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        # Fallback to JSON display
        for i, result in enumerate(results):
            with st.expander(f"Result {i + 1}"):
                st.json(result)


def render_sources_list(sources: List[Dict]):
    """
    Render source citations with clickable YouTube links.
    
    Args:
        sources: List of source dictionaries with video_id, timestamp, etc.
    """
    if not sources:
        st.info("No sources available.")
        return
    
    for i, source in enumerate(sources):
        video_id = source.get("video_id", "")
        timestamp = source.get("start_time", source.get("timestamp", 0))
        podcast = source.get("podcast", source.get("podcast_name", "Unknown"))
        speaker = source.get("speaker", "")
        text = source.get("text", "")
        similarity = source.get("similarity", 0)
        
        # Create YouTube link
        if video_id:
            youtube_link = f"https://youtube.com/watch?v={video_id}&t={int(timestamp)}"
        else:
            youtube_link = None
        
        # Render source card
        st.markdown(f"""
        <div class="source-citation">
            <strong>📍 Source {i + 1}</strong>
            {f' - <a href="{youtube_link}" target="_blank">📺 Watch at {timestamp:.1f}s</a>' if youtube_link else ''}
            <br>
            <small>
                🎙️ {podcast}
                {f' | 🗣️ {speaker}' if speaker else ''}
                {f' | 📊 {similarity:.0%} match' if similarity else ''}
            </small>
            {f'<br><em>"{text[:200]}..."</em>' if text else ''}
        </div>
        """, unsafe_allow_html=True)
    
    # Download sources
    if st.button("📥 Export Sources as JSON"):
        st.download_button(
            label="Download",
            data=json.dumps(sources, indent=2),
            file_name="sources.json",
            mime="application/json"
        )


def render_statistics_cards(stats: Dict[str, Any]):
    """
    Render statistics as styled cards.
    
    Args:
        stats: Statistics dictionary
    """
    graph_stats = stats.get("graph", {})
    vector_stats = stats.get("vectors", {})
    
    # Create metric cards
    st.markdown("""
    <style>
    .stat-card {
        background: linear-gradient(135deg, #2d2d44 0%, #1f1f35 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{graph_stats.get('episode_count', 0)}</p>
            <p class="stat-label">Episodes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        entity_count = sum(
            graph_stats.get(k, 0) 
            for k in ["person_count", "book_count", "movie_count", "company_count", "topic_count"]
        )
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{entity_count}</p>
            <p class="stat-label">Entities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{graph_stats.get('relationship_count', 0)}</p>
            <p class="stat-label">Relationships</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{vector_stats.get('total_chunks', 0)}</p>
            <p class="stat-label">Text Chunks</p>
        </div>
        """, unsafe_allow_html=True)


def render_sentiment_timeline(timeline_data: List[Dict]):
    """
    Render a sentiment timeline using Plotly.
    
    Args:
        timeline_data: List of dicts with date, sentiment, context
    """
    if not timeline_data:
        st.info("No timeline data available.")
        return
    
    try:
        import plotly.graph_objects as go
        import pandas as pd
        
        df = pd.DataFrame(timeline_data)
        
        # Map sentiment to numeric values
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        df["sentiment_value"] = df["sentiment"].map(sentiment_map)
        
        # Create figure
        fig = go.Figure()
        
        # Add sentiment line
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["sentiment_value"],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>" +
                "<b>Sentiment:</b> %{customdata}<br>" +
                "<extra></extra>"
            ),
            customdata=df["sentiment"]
        ))
        
        # Update layout
        fig.update_layout(
            title="Sentiment Over Time",
            xaxis_title="Date",
            yaxis_title="Sentiment",
            yaxis=dict(
                tickmode='array',
                tickvals=[-1, 0, 1],
                ticktext=['Negative', 'Neutral', 'Positive']
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.warning("Install plotly for timeline visualization: pip install plotly")
    except Exception as e:
        st.error(f"Timeline visualization error: {e}")


def render_entity_network(entities: List[Dict[str, Any]]):
    """
    Render entity relationship network using PyVis.
    
    Args:
        entities: List of entity dictionaries
    """
    try:
        from pyvis.network import Network
        import tempfile
        
        # Create network
        net = Network(
            height="500px",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white"
        )
        
        # Node color mapping
        colors = {
            "PERSON": "#667eea",
            "BOOK": "#f59e0b",
            "MOVIE": "#ef4444",
            "COMPANY": "#22c55e",
            "TOPIC": "#8b5cf6",
        }
        
        # Add nodes
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            value = entity.get("value", "Unknown")
            
            net.add_node(
                value,
                label=value,
                color=colors.get(entity_type, "#94a3b8"),
                title=f"{entity_type}: {value}"
            )
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
            net.save_graph(f.name)
            with open(f.name, 'r') as html_file:
                html_content = html_file.read()
        
        st.components.v1.html(html_content, height=500, scrolling=True)
        
    except ImportError:
        st.warning("Install pyvis for network visualization: pip install pyvis")
    except Exception as e:
        st.error(f"Network visualization error: {e}")


def render_query_type_badge(query_type: str):
    """Render a colored badge for query type."""
    colors = {
        "graph": "#22c55e",
        "semantic": "#3b82f6",
        "hybrid": "#8b5cf6",
        "verify": "#f59e0b"
    }
    
    color = colors.get(query_type.lower(), "#94a3b8")
    
    st.markdown(f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    ">{query_type}</span>
    """, unsafe_allow_html=True)


def render_verification_result(result: Dict[str, Any]):
    """Render verification result with visual indicators."""
    verified = result.get("verified")
    reason = result.get("reason", "")
    confidence = result.get("confidence", 0)
    
    if verified is True:
        icon = "✅"
        color = "#22c55e"
        status = "VERIFIED"
    elif verified is False:
        icon = "❌"
        color = "#ef4444"
        status = "NOT VERIFIED"
    else:
        icon = "❓"
        color = "#f59e0b"
        status = "UNCERTAIN"
    
    st.markdown(f"""
    <div style="
        background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
        border: 1px solid {color};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    ">
        <h1 style="font-size: 3rem; margin: 0;">{icon}</h1>
        <h3 style="color: {color}; margin: 0.5rem 0;">{status}</h3>
        {f'<p>Confidence: {confidence:.0%}</p>' if confidence else ''}
        {f'<p style="color: #888;">{reason}</p>' if reason else ''}
    </div>
    """, unsafe_allow_html=True)
