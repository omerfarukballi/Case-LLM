#!/bin/bash

# Podcast Knowledge Graph System - Setup Script

echo "🎙️ Podcast Knowledge Graph System Setup"
echo "========================================"

# Check Python version
echo "📦 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "   Python version: $python_version"

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
echo ""
echo "⚙️ Checking configuration..."
if [ ! -f .env ]; then
    echo "   Creating .env from .env.example..."
    cp .env.example .env
    echo "   ⚠️  Please edit .env with your API keys!"
else
    echo "   ✓ .env file exists"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data/cache
mkdir -p chroma_db

# Check Neo4j connection
echo ""
echo "🔌 Checking Neo4j..."
if docker ps | grep -q neo4j; then
    echo "   ✓ Neo4j container is running"
else
    echo "   ⚠️  Neo4j is not running. Start with:"
    echo "   docker run -d --name neo4j \\"
    echo "     -p 7474:7474 -p 7687:7687 \\"
    echo "     -e NEO4J_AUTH=neo4j/yourpassword \\"
    echo "     neo4j:latest"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys (OPENAI_API_KEY, ASSEMBLYAI_API_KEY, NEO4J_PASSWORD)"
echo "2. Start Neo4j if not running"
echo "3. Process a video: python main.py --url 'https://youtube.com/watch?v=VIDEO_ID' --podcast 'Podcast Name'"
echo "4. Or start the UI: streamlit run ui/app.py"
