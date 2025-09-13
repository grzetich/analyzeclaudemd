# Development Environment Setup

This document explains how to set up a local development environment that mirrors production.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Add your GitHub token to .env
# Edit .env and replace 'your_github_personal_access_token_here' with your actual token

# 3. Start development environment
docker-compose up --build

# 4. Access the application
# http://localhost:5000 - Development mode with live reload
```

### Option 2: Native Python

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your GitHub token

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# 5. Run application
python app.py
```

## 🐳 Docker Commands

```bash
# Development (live reload, debug mode)
docker-compose up

# Production-like environment
docker-compose --profile prod up

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up --build
```

## 🔧 Environment Differences

| Feature | Development | Production |
|---------|-------------|------------|
| **Python Version** | 3.11 | 3.11 (Render.com) |
| **WSGI Server** | Gunicorn (reload) | Gunicorn (optimized) |
| **Debug Mode** | ON | OFF |
| **Live Reload** | ✅ | ❌ |
| **Log Level** | DEBUG | INFO |
| **Workers** | 1 | 2+ |

## 📁 Directory Structure

```
analyzeclaude/
├── cache/              # Analysis results (mounted in Docker)
├── logs/               # Application logs (mounted in Docker)
├── templates/          # HTML templates
├── static/            # CSS/JS/images
├── scripts/           # Development scripts
├── Dockerfile         # Container definition
├── docker-compose.yml # Multi-container setup
├── .env.example       # Environment template
├── requirements.txt   # Python dependencies
└── app.py            # Main application
```

## 🧪 Testing Production Changes Safely

### 1. Test in Docker First
```bash
# Test with production-like settings
docker-compose --profile prod up
```

### 2. Environment-Specific Code
```python
import os

# Use environment variables for configuration
if os.getenv('DEV_MODE', 'false').lower() == 'true':
    # Development-only code
    USE_MOCK_DATA = True
else:
    # Production code
    USE_MOCK_DATA = False
```

### 3. Safe Mock Data Loading
Instead of modifying production code, use environment flags:

```python
def load_analysis_history():
    """Load historical analysis results."""
    if os.getenv('USE_MOCK_DATA', 'false').lower() == 'true':
        return load_mock_data()
    
    # Original production code
    try:
        if db:
            return db.get_analysis_history()
        else:
            if os.path.exists(ANALYSIS_HISTORY_PATH):
                with open(ANALYSIS_HISTORY_PATH, 'r') as f:
                    return json.load(f)
    except Exception as e:
        logging.error(f"Error loading analysis history: {e}")
    return []
```

## 🔍 Debugging

### View Container Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f claude-analyzer
```

### Access Container Shell
```bash
docker-compose exec claude-analyzer /bin/bash
```

### Monitor Resource Usage
```bash
docker stats
```

## 🚀 Deployment

The Docker setup matches production:
- Same Python version (3.11)
- Same dependencies (requirements.txt)
- Same WSGI server (Gunicorn)
- Same environment variables

Changes tested in Docker should work identically in production.

## 🛟 Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000
# Kill the process
kill -9 <PID>
```

### Docker Issues
```bash
# Clean rebuild
docker-compose down
docker system prune
docker-compose up --build
```

### Python Environment Issues
```bash
# Clean virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Flask Development Guide](https://flask.palletsprojects.com/en/2.3.x/tutorial/deploy/)
- [Render.com Deployment Guide](https://render.com/docs/deploy-flask)