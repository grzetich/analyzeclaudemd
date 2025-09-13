#!/bin/bash
# Development Environment Setup Script

set -e

echo "🔧 Setting up development environment..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "✅ Docker found"
    DOCKER_AVAILABLE=true
else
    echo "⚠️  Docker not found - will use native Python setup"
    DOCKER_AVAILABLE=false
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚡ Please edit .env and add your GITHUB_PAT token"
fi

# Option 1: Docker setup (recommended)
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo "🐳 Setting up Docker environment..."
    
    # Build and start containers
    docker-compose up --build -d
    
    echo "✅ Docker setup complete!"
    echo "🌐 Application running at:"
    echo "   - Development: http://localhost:5000"
    echo "   - Production-like: docker-compose --profile prod up (port 5001)"
    echo ""
    echo "📱 Useful commands:"
    echo "   - View logs: docker-compose logs -f"
    echo "   - Restart: docker-compose restart"
    echo "   - Stop: docker-compose down"
    
else
    # Option 2: Native Python setup
    echo "🐍 Setting up native Python environment..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d" " -f2 | cut -d"." -f1-2)
    echo "Python version: $python_version"
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Download NLTK data
    echo "Downloading NLTK data..."
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
    
    echo "✅ Native Python setup complete!"
    echo "🌐 To start the application:"
    echo "   source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
    echo "   python app.py"
fi

echo ""
echo "🎉 Development environment is ready!"