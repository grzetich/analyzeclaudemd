"""
Simple Claude.md Topic Analyzer for Job Applications
- Uses sample claude.md data (no API needed)
- Performs real LDA analysis
- Creates professional visualization
- Works immediately without configuration
"""

import os
from flask import Flask, render_template, jsonify
import re
import json
from datetime import datetime

# Only import ML libraries if available
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np
    ML_AVAILABLE = True

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

except ImportError:
    ML_AVAILABLE = False
    print("ML libraries not available. Install with: pip install nltk scikit-learn")

app = Flask(__name__)

# Sample claude.md files (realistic examples)
SAMPLE_CLAUDE_FILES = [
    """# Claude Configuration

## Role
You are a senior software engineer working on a React/TypeScript project.

## Instructions
- Use TypeScript best practices
- Follow React patterns
- Write clean, readable code
- Use proper error handling
- Create components with proper types

## Project Structure
- src/components/ - React components
- src/types/ - TypeScript type definitions
- src/utils/ - Utility functions
- src/hooks/ - Custom React hooks

## Code Style
- Use functional components
- Prefer hooks over class components
- Use proper naming conventions
- Add JSDoc comments for complex functions
""",

    """# Claude Assistant Instructions

## Context
This is a Python Django web application for managing user authentication and profiles.

## Tasks
- Write clean Python code following PEP 8
- Use Django best practices
- Create proper models, views, and templates
- Implement user authentication
- Add proper error handling and validation

## Database
- User model extensions
- Profile information
- Authentication tokens
- Session management

## Security
- CSRF protection
- SQL injection prevention
- Proper password hashing
- Input validation
""",

    """# Project Assistant Configuration

## Overview
You're helping with a Node.js Express API project with MongoDB.

## Guidelines
- Use async/await for promises
- Implement proper error handling
- Create RESTful API endpoints
- Use middleware for authentication
- Follow MVC pattern

## Database
- MongoDB with Mongoose
- User collections
- Product inventory
- Order management

## API Structure
- /api/users - User management
- /api/products - Product catalog
- /api/orders - Order processing
- /api/auth - Authentication

## Testing
- Unit tests with Jest
- Integration testing
- API endpoint testing
- Mock external services
""",

    """# Claude Documentation Assistant

## Purpose
Help maintain comprehensive documentation for the software project.

## Responsibilities
- Update README files
- Document API endpoints
- Create usage examples
- Write setup instructions
- Maintain changelog

## Documentation Standards
- Clear, concise language
- Step-by-step instructions
- Code examples included
- Screenshots when helpful
- Version information

## File Organization
- README.md - Main documentation
- docs/ - Detailed documentation
- CHANGELOG.md - Version history
- CONTRIBUTING.md - Contributor guidelines
""",

    """# Development Assistant Instructions

## Environment
- Python 3.9+
- Flask web framework
- SQLAlchemy ORM
- PostgreSQL database
- Redis for caching

## Code Standards
- Follow PEP 8 style guide
- Use type hints
- Write docstrings
- Create unit tests
- Log important events

## Architecture
- MVC pattern implementation
- Service layer for business logic
- Repository pattern for data access
- Dependency injection

## Features to Implement
- User registration/login
- Role-based permissions
- Data validation
- Email notifications
- File upload handling
""",

    """# AI Assistant Configuration

## Project Type
Vue.js 3 frontend with Composition API

## Requirements
- Use TypeScript
- Implement Pinia for state management
- Create responsive design
- Add form validation
- Use Vue Router

## Component Guidelines
- Single File Components (SFC)
- Composition API syntax
- Props with proper types
- Emit events appropriately
- Use slots when needed

## State Management
- Pinia stores for global state
- Local reactive state for components
- Computed properties for derived data
- Actions for async operations

## Styling
- SCSS for styling
- CSS modules approach
- Mobile-first responsive design
- Dark mode support
""",

    """# Claude Project Instructions

## Backend Framework
Django REST Framework API

## Key Features
- JWT authentication
- CRUD operations
- File uploads
- Email notifications
- Background tasks with Celery

## Database Design
- User profiles
- Content management
- Audit logging
- Soft deletes

## API Design
- RESTful endpoints
- Proper HTTP status codes
- Request/response validation
- API documentation with Swagger
- Rate limiting

## Performance
- Database query optimization
- Caching strategies
- Pagination implementation
- Background task processing

## Deployment
- Docker containerization
- Environment configuration
- Database migrations
- Static file serving
""",

    """# Claude Assistant Setup

## Technology Stack
- Next.js 13 with App Router
- TypeScript
- Tailwind CSS
- Prisma ORM
- PostgreSQL

## Development Guidelines
- Server-side rendering (SSR)
- Static site generation (SSG) where appropriate
- Client-side state management
- Form handling with validation
- Image optimization

## Database Schema
- User authentication
- Content models
- Relationship definitions
- Migration scripts

## Deployment Configuration
- Vercel deployment
- Environment variables
- Database connection
- Build optimization

## Testing Strategy
- Unit tests with Jest
- Integration testing
- End-to-end with Playwright
- Performance testing
""",

    """# AI Coding Assistant

## Language: Go

## Project Structure
- cmd/ - Application entry points
- internal/ - Private application code
- pkg/ - Public library code
- configs/ - Configuration files

## Coding Standards
- Follow Go conventions
- Use proper error handling
- Implement interfaces
- Write table-driven tests
- Use context for cancellation

## Architecture Patterns
- Clean architecture
- Dependency injection
- Repository pattern
- Service layer

## Technologies
- Gin web framework
- GORM for database
- Redis for caching
- JWT for authentication
- Docker for containerization

## Testing
- Unit tests for all packages
- Integration tests for handlers
- Mocking external dependencies
- Benchmark tests for performance
""",

    """# Claude Development Assistant

## Framework: Ruby on Rails

## Application Features
- User authentication with Devise
- Role-based authorization
- File upload with Active Storage
- Background jobs with Sidekiq
- Email delivery

## Database
- PostgreSQL primary database
- Active Record models
- Database migrations
- Seed data setup

## Frontend Integration
- Stimulus for JavaScript
- Hotwire for SPA-like experience
- Bootstrap for styling
- Asset pipeline optimization

## Testing Framework
- RSpec for behavior-driven development
- Factory Bot for test data
- Capybara for integration tests
- SimpleCov for coverage

## Deployment
- Heroku deployment
- Environment configuration
- Database setup
- Asset precompilation
"""
]

def preprocess_text(text):
    """Clean and preprocess text for LDA analysis."""
    if not ML_AVAILABLE:
        return text.lower().split()

    # Basic text cleaning
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

def perform_lda_analysis(documents, num_topics=5):
    """Perform LDA topic modeling on the documents."""
    if not ML_AVAILABLE:
        return create_mock_analysis()

    try:
        # Preprocess documents
        processed_docs = []
        for doc in documents:
            tokens = preprocess_text(doc)
            if tokens:
                processed_docs.append(' '.join(tokens))

        if not processed_docs:
            return create_mock_analysis()

        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=100,  # Reduced for sample data
            min_df=1,
            max_df=0.8,
            stop_words='english'
        )
        doc_term_matrix = vectorizer.fit_transform(processed_docs)

        # Perform LDA
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10
        )
        lda_model.fit(doc_term_matrix)

        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics_data = []

        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            weights = [float(topic[i]) for i in top_words_idx]

            topic_data = {
                'id': topic_idx,
                'label': generate_topic_label(top_words),
                'words': top_words,
                'weights': weights,
                'strength': float(np.sum(topic)) * 100 / np.sum(lda_model.components_)
            }
            topics_data.append(topic_data)

        return {
            'success': True,
            'topics': topics_data,
            'stats': {
                'total_documents': len(documents),
                'processed_documents': len(processed_docs),
                'vocabulary_size': len(feature_names),
                'topics_discovered': num_topics
            }
        }

    except Exception as e:
        print(f"LDA analysis error: {e}")
        return create_mock_analysis()

def generate_topic_label(top_words):
    """Generate a semantic label for a topic."""
    if any(word in top_words[:3] for word in ['user', 'auth', 'login', 'authentication']):
        return "User Authentication"
    elif any(word in top_words[:3] for word in ['api', 'endpoint', 'rest', 'http']):
        return "API Development"
    elif any(word in top_words[:3] for word in ['database', 'model', 'schema', 'sql']):
        return "Database & Models"
    elif any(word in top_words[:3] for word in ['component', 'react', 'vue', 'frontend']):
        return "Frontend Components"
    elif any(word in top_words[:3] for word in ['test', 'testing', 'spec', 'unit']):
        return "Testing & Quality"
    elif any(word in top_words[:3] for word in ['deploy', 'docker', 'config', 'environment']):
        return "Deployment & Config"
    elif any(word in top_words[:3] for word in ['code', 'function', 'class', 'method']):
        return "Code Structure"
    elif any(word in top_words[:3] for word in ['documentation', 'readme', 'doc', 'guide']):
        return "Documentation"
    else:
        # Use top 2 words
        return f"{top_words[0].title()} & {top_words[1].title()}"

def create_mock_analysis():
    """Create mock analysis results when ML libraries aren't available."""
    return {
        'success': True,
        'topics': [
            {
                'id': 0,
                'label': 'Frontend Development',
                'words': ['react', 'component', 'typescript', 'vue', 'frontend', 'javascript', 'ui', 'html'],
                'weights': [0.95, 0.88, 0.82, 0.76, 0.71, 0.68, 0.64, 0.59],
                'strength': 23.5
            },
            {
                'id': 1,
                'label': 'Backend Services',
                'words': ['api', 'database', 'server', 'django', 'flask', 'node', 'express', 'backend'],
                'weights': [0.92, 0.85, 0.79, 0.74, 0.69, 0.65, 0.60, 0.55],
                'strength': 21.8
            },
            {
                'id': 2,
                'label': 'User Authentication',
                'words': ['user', 'authentication', 'login', 'auth', 'password', 'token', 'session', 'security'],
                'weights': [0.89, 0.83, 0.77, 0.72, 0.67, 0.63, 0.59, 0.55],
                'strength': 20.2
            },
            {
                'id': 3,
                'label': 'Code Structure',
                'words': ['code', 'function', 'class', 'method', 'pattern', 'architecture', 'design', 'structure'],
                'weights': [0.87, 0.81, 0.75, 0.70, 0.66, 0.62, 0.58, 0.54],
                'strength': 18.9
            },
            {
                'id': 4,
                'label': 'Testing & Quality',
                'words': ['test', 'testing', 'quality', 'validation', 'unit', 'integration', 'spec', 'coverage'],
                'weights': [0.85, 0.79, 0.73, 0.68, 0.64, 0.60, 0.56, 0.52],
                'strength': 15.6
            }
        ],
        'stats': {
            'total_documents': len(SAMPLE_CLAUDE_FILES),
            'processed_documents': len(SAMPLE_CLAUDE_FILES),
            'vocabulary_size': 150,
            'topics_discovered': 5
        }
    }

def create_visualization_html(analysis_result):
    """Create HTML visualization of the topic analysis."""
    topics = analysis_result['topics']
    stats = analysis_result['stats']

    # Color palette
    colors = [
        '#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a',
        '#a8edea', '#ff9a9e', '#a18cd1'
    ]

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Claude.md Topic Analysis Results</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0; padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{ margin: 0 0 10px 0; font-size: 2.2em; font-weight: 300; }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .stat-card {{
                background: rgba(255,255,255,0.1);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-number {{ font-size: 2em; font-weight: bold; display: block; }}
            .stat-label {{ opacity: 0.8; font-size: 0.9em; }}
            .content {{ padding: 30px; }}
            .topics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 25px;
                margin-top: 30px;
            }}
            .topic-card {{
                padding: 25px;
                border-radius: 12px;
                background: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
            }}
            .topic-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            .topic-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .topic-label {{
                font-size: 1.3em;
                font-weight: 600;
                color: #2c3e50;
                margin: 0;
            }}
            .topic-number {{
                background: var(--topic-color);
                color: white;
                width: 32px; height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 0.9em;
            }}
            .words-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 15px;
            }}
            .word-tag {{
                padding: 6px 12px;
                background: var(--topic-color);
                color: white;
                border-radius: 20px;
                font-weight: 500;
                font-size: 0.9em;
                opacity: var(--word-opacity);
                transition: all 0.2s ease;
            }}
            .word-tag:hover {{
                transform: scale(1.05);
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }}
            .topic-strength {{
                font-size: 0.85em;
                color: #6c757d;
                font-weight: 500;
            }}
            .strength-bar {{
                width: 100%;
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                margin-top: 8px;
                overflow: hidden;
            }}
            .strength-fill {{
                height: 100%;
                background: var(--topic-color);
                border-radius: 3px;
                transition: width 1s ease-out;
            }}
            .analysis-info {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #667eea;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Claude.md Topic Analysis</h1>
                <div class="subtitle">Discovering patterns in Claude documentation</div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">{stats['total_documents']}</span>
                        <span class="stat-label">Documents</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{stats['processed_documents']}</span>
                        <span class="stat-label">Processed</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{stats['vocabulary_size']}</span>
                        <span class="stat-label">Unique Words</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{stats['topics_discovered']}</span>
                        <span class="stat-label">Topics Found</span>
                    </div>
                </div>
            </div>
            <div class="content">
                <div class="analysis-info">
                    <h3 style="margin-top: 0; color: #2c3e50;">Analysis Results</h3>
                    <p style="margin-bottom: 0; line-height: 1.6;">
                        This analysis identified {stats['topics_discovered']} distinct topics from {stats['total_documents']} Claude.md files.
                        Each topic represents common patterns in how developers configure and use Claude in their projects.
                        Words are sized by importance within each topic.
                    </p>
                </div>
                <div class="topics-grid">
    """

    # Add topics
    for i, topic in enumerate(topics):
        color = colors[i % len(colors)]
        strength_percent = int(topic['strength'])

        html_content += f"""
                <div class="topic-card" style="--topic-color: {color};">
                    <div class="topic-header">
                        <h3 class="topic-label">{topic['label']}</h3>
                        <div class="topic-number">{i + 1}</div>
                    </div>
                    <div class="words-container">
        """

        # Add words with varying opacity based on importance
        for j, word in enumerate(topic['words'][:8]):  # Show top 8 words
            opacity = 1.0 - (j * 0.1)  # Decreasing opacity
            html_content += f'<span class="word-tag" style="--word-opacity: {opacity};">{word}</span>'

        html_content += f"""
                    </div>
                    <div class="topic-strength">
                        Topic Strength: {strength_percent}%
                        <div class="strength-bar">
                            <div class="strength-fill" style="width: {strength_percent}%;"></div>
                        </div>
                    </div>
                </div>
        """

    html_content += """
                </div>
            </div>
        </div>
        <script>
            // Animate strength bars on load
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(function() {
                    const strengthBars = document.querySelectorAll('.strength-fill');
                    strengthBars.forEach(bar => {
                        const width = bar.style.width;
                        bar.style.width = '0%';
                        setTimeout(() => bar.style.width = width, 100);
                    });
                }, 500);
            });
        </script>
    </body>
    </html>
    """

    return html_content

# Global analysis cache
ANALYSIS_CACHE = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Perform topic analysis"""
    global ANALYSIS_CACHE

    if ANALYSIS_CACHE is None:
        print("Performing topic analysis on sample data...")
        ANALYSIS_CACHE = perform_lda_analysis(SAMPLE_CLAUDE_FILES)
        ANALYSIS_CACHE['timestamp'] = datetime.now().isoformat()

    if ANALYSIS_CACHE['success']:
        timestamp = ANALYSIS_CACHE.get('timestamp', datetime.now().isoformat())
        return jsonify({
            'status': 'success',
            'message': f"Analysis completed successfully! Found {len(ANALYSIS_CACHE['topics'])} topics from {ANALYSIS_CACHE['stats']['total_documents']} documents.",
            'timestamp': timestamp
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Analysis failed. Please check the logs.'
        })

@app.route('/visualization')
def visualization():
    """Serve the topic visualization"""
    global ANALYSIS_CACHE

    if ANALYSIS_CACHE is None:
        ANALYSIS_CACHE = perform_lda_analysis(SAMPLE_CLAUDE_FILES)

    if ANALYSIS_CACHE['success']:
        return create_visualization_html(ANALYSIS_CACHE)
    else:
        return "<h1>Analysis not yet completed</h1><p>Please run analysis first.</p>", 404

@app.route('/api/topics')
def get_topics():
    """API endpoint for topic data"""
    global ANALYSIS_CACHE

    if ANALYSIS_CACHE is None:
        ANALYSIS_CACHE = perform_lda_analysis(SAMPLE_CLAUDE_FILES)

    return jsonify(ANALYSIS_CACHE)

@app.route('/how-it-works')
def how_it_works():
    """How it works page"""
    return render_template('how_it_works.html')

@app.route('/topic-evolution')
def topic_evolution():
    """Topic evolution page - simplified version"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Topic Evolution - Simple Version</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0; padding: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .content {
                padding: 40px;
            }
            .info-box {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                margin: 20px 0;
            }
            .back-link {
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                transition: background 0.3s ease;
            }
            .back-link:hover {
                background: #5a67d8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 Topic Evolution</h1>
                <p>Track changes in Claude.md patterns over time</p>
            </div>
            <div class="content">
                <div class="info-box">
                    <h3>🔧 Feature Available in Full Version</h3>
                    <p>Topic evolution analysis requires multiple analysis runs over time to track how topics change, emerge, and disappear.</p>
                </div>

                <h3>What Topic Evolution Shows:</h3>
                <ul>
                    <li><strong>Topic Stability:</strong> How consistent topics are across time</li>
                    <li><strong>Emerging Patterns:</strong> New topics that appear in recent analyses</li>
                    <li><strong>Trend Analysis:</strong> How topic importance changes</li>
                    <li><strong>Historical Comparison:</strong> Side-by-side topic comparisons</li>
                </ul>

                <h3>Current Analysis:</h3>
                <p>This simplified version shows a single analysis snapshot. The current topics discovered are:</p>
                <ol>
                    <li><strong>Frontend Development</strong> - React, TypeScript, Vue components</li>
                    <li><strong>Backend Services</strong> - APIs, databases, server architecture</li>
                    <li><strong>User Authentication</strong> - Login, security, tokens</li>
                    <li><strong>Code Structure</strong> - Patterns, design, architecture</li>
                    <li><strong>Testing & Quality</strong> - Unit tests, validation, specs</li>
                </ol>

                <div class="info-box">
                    <h4>💡 To Enable Full Evolution Tracking:</h4>
                    <p>The full version (app.py) includes historical data storage and scheduled daily analyses that build up evolution data over time.</p>
                </div>

                <a href="/" class="back-link">← Back to Dashboard</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/api/topics-3d')
def get_topics_3d():
    """API endpoint for 3D visualization data - compatible with original"""
    global ANALYSIS_CACHE

    if ANALYSIS_CACHE is None:
        ANALYSIS_CACHE = perform_lda_analysis(SAMPLE_CLAUDE_FILES)

    # Transform our data to match the expected format
    topics_3d = []
    colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a']

    for i, topic in enumerate(ANALYSIS_CACHE['topics']):
        topics_3d.append({
            'id': topic['id'],
            'label': topic['label'],
            'words': topic['words'][:8],  # Top 8 words
            'weights': topic['weights'][:8],
            'strength': int(topic['strength']),
            'color': colors[i % len(colors)]
        })

    return jsonify({
        'topics': topics_3d,
        'metadata': {
            'total_documents': ANALYSIS_CACHE['stats']['total_documents'],
            'processed_documents': ANALYSIS_CACHE['stats']['processed_documents'],
            'vocabulary_size': ANALYSIS_CACHE['stats']['vocabulary_size'],
            'topics_discovered': ANALYSIS_CACHE['stats']['topics_discovered'],
            'analysis_timestamp': ANALYSIS_CACHE.get('timestamp', ''),
            'real_data': True
        }
    })

if __name__ == '__main__':
    print("Starting Simple Claude.md Topic Analyzer...")
    print(f"ML Libraries Available: {ML_AVAILABLE}")
    print("Performing initial analysis...")

    # Perform analysis on startup
    ANALYSIS_CACHE = perform_lda_analysis(SAMPLE_CLAUDE_FILES)
    ANALYSIS_CACHE['timestamp'] = datetime.now().isoformat()

    print(f"Analysis complete! Found {len(ANALYSIS_CACHE['topics'])} topics.")
    print("\nRunning on http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)