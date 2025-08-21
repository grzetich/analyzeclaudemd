# Claude.md Topic Analyzer & Visualizer

## Overview

This Flask web application analyzes `claude.md` files from GitHub repositories using topic modeling (LDA) and creates interactive visualizations to discover common themes and patterns in Claude documentation across projects.

## Key Features

- **GitHub Integration**: Searches for and downloads `claude.md` files from public repositories
- **Text Processing**: Uses NLTK for tokenization, stopword removal, and lemmatization
- **Topic Modeling**: Implements Latent Dirichlet Allocation (LDA) using scikit-learn
- **Visualization**: Generates custom HTML visualizations showing discovered topics and key words
- **Rate Limiting**: Handles GitHub API rate limits with retry logic

## Technical Architecture

### Backend (Flask)
- `/` - Main interface page
- `/analyze` - Triggers data collection and LDA analysis
- `/visualization` - Serves generated topic visualization

### Dependencies
- **Flask 3.0.0** - Web framework
- **scikit-learn 1.5.0** - Machine learning (LDA implementation)
- **NLTK 3.9.1** - Natural language processing
- **requests 2.32.3** - GitHub API calls
- **numpy/scipy/pandas** - Scientific computing stack

### Data Pipeline
1. **Collection**: GitHub API search for `filename:claude.md`
2. **Preprocessing**: Text cleaning, tokenization, lemmatization
3. **Modeling**: LDA with CountVectorizer feature extraction
4. **Visualization**: Custom HTML with topic words and weights

## Deployment

### Render.com Configuration
- **Runtime**: Python 3.11+ (compatible with scikit-learn)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Environment Variable**: `GITHUB_PAT` (GitHub Personal Access Token)

### Python Compatibility
- **Python 3.13 Ready**: Uses scikit-learn instead of gensim for compatibility
- **Pre-compiled Wheels**: All dependencies available as wheels on Linux
- **No C Compilation**: Avoids build issues on deployment platforms

## Usage Instructions

1. Set `GITHUB_PAT` environment variable with GitHub Personal Access Token
2. Click "Start Analysis" to fetch claude.md files (limited to 500 for MVP)
3. Wait for LDA processing to complete
4. Click "View Visualization" to see discovered topics

## Rate Limits & Constraints

- **GitHub API**: 5,000 requests/hour with PAT, 30 requests/minute for code search
- **Collection Limit**: 500 files maximum to manage API limits and processing time
- **Free Tier Friendly**: Optimized for Render.com free tier resource constraints

## Recent Updates

**Version 2.0** (Latest):
- Replaced gensim with scikit-learn for Python 3.13 compatibility
- Custom HTML visualization instead of pyLDAvis
- Improved build reliability on Render.com
- Faster deployment with pre-compiled wheels

**Version 1.0**:
- Original implementation with gensim and pyLDAvis
- Had Python 3.13 compatibility issues on deployment

## Research Applications

This tool enables analysis of:
- Common Claude usage patterns across repositories
- Documentation trends in AI-assisted development
- Topic evolution in Claude integration practices
- Best practices identification in Claude.md files

Perfect for researchers studying AI tool adoption, developers understanding Claude usage patterns, or teams wanting to benchmark their Claude documentation against the broader ecosystem.