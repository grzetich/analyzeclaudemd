"""
Mock version of app.py for local testing without heavy C++ dependencies.
This version replaces scikit-learn LDA with a simple mock implementation.
"""
import os
import sys
import requests
import json
import re
import time
from datetime import datetime
from flask import Flask, render_template, jsonify

# Mock implementations to avoid C++ dependencies
class MockCountVectorizer:
    def __init__(self, **kwargs):
        self.feature_names = []
        
    def fit_transform(self, documents):
        # Simple word counting without numpy
        vocab = set()
        for doc in documents:
            words = doc.split()
            vocab.update(words)
        
        self.feature_names = list(vocab)[:50]  # Limit to 50 features
        return MockMatrix(len(documents), len(self.feature_names))
    
    def get_feature_names_out(self):
        return self.feature_names

class MockMatrix:
    def __init__(self, rows, cols):
        self.shape = (rows, cols)

class MockLDA:
    def __init__(self, **kwargs):
        self.n_components = kwargs.get('n_components', 5)
        self.components_ = None
        
    def fit(self, matrix):
        # Create mock topic-word distributions
        self.components_ = []
        words = ['claude', 'assistant', 'project', 'code', 'file', 'test', 'documentation', 
                'function', 'class', 'method', 'api', 'database', 'server', 'client', 'user']
        
        for i in range(self.n_components):
            # Create fake weights for words
            component = {}
            for j, word in enumerate(words):
                component[j] = 1.0 + (i * 0.5) + (j * 0.1)  # Fake weights
            self.components_.append(MockComponent(component))
    
class MockComponent:
    def __init__(self, weights_dict):
        self.weights = weights_dict
        
    def argsort(self):
        # Return indices sorted by weight (descending)
        sorted_items = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return MockIndices([item[0] for item in sorted_items])
    
    def sum(self):
        return sum(self.weights.values())
    
    def __getitem__(self, idx):
        return self.weights.get(idx, 0.0)

class MockIndices:
    def __init__(self, indices):
        self.indices = indices
    
    def __getitem__(self, slice_obj):
        return MockIndices(self.indices[slice_obj])

# Mock memory management
class MockMemoryManager:
    def __init__(self):
        pass
    
    def get_memory_stats(self):
        return {'rss': 150.0, 'vms': 500.0, 'percent': 0.5, 'available': 8000.0}
    
    def log_memory_usage(self, context=""):
        stats = self.get_memory_stats()
        print(f"Memory ({context}): RSS {stats['rss']}MB")
        return stats
    
    def cleanup_resources(self, context=""):
        print(f"Mock cleanup ({context})")

# Mock NLTK functions
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Mock NLTK
    def word_tokenize(text):
        return text.split()
    
    class MockLemmatizer:
        def lemmatize(self, word):
            return word.lower()
    
    stopwords = type('MockStopwords', (), {
        'words': lambda lang: ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    })()

app = Flask(__name__)

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_PAT")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "Accept": "application/vnd.github.v3.text-match+json"
}

# Mock global variables
memory_manager = MockMemoryManager()
stop_words = set(stopwords.words('english') if NLTK_AVAILABLE else ['the', 'and', 'or', 'but'])
lemmatizer = MockLemmatizer() if not NLTK_AVAILABLE else WordNetLemmatizer()

def preprocess_text(text):
    """Simple text preprocessing without heavy dependencies"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    if NLTK_AVAILABLE:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def mock_lda_analysis(documents, num_topics=5):
    """Mock LDA analysis that works without scikit-learn"""
    print(f"Mock LDA analysis with {len(documents)} documents, {num_topics} topics")
    
    # Simple topic generation based on common words
    processed_docs = []
    for doc in documents:
        tokens = preprocess_text(doc)
        if tokens:
            processed_docs.append(' '.join(tokens))
    
    if not processed_docs:
        return False, None
    
    # Mock topic data
    topics_data = []
    topic_labels = [
        "AI Assistant Configuration",
        "Project Structure", 
        "Code Guidelines",
        "Testing & Quality",
        "Documentation"
    ]
    
    word_sets = [
        ['claude', 'assistant', 'ai', 'prompt', 'system', 'role', 'behavior', 'instruction'],
        ['project', 'file', 'directory', 'structure', 'folder', 'path', 'organization'],
        ['code', 'function', 'class', 'method', 'style', 'format', 'convention'],
        ['test', 'testing', 'spec', 'validation', 'check', 'verify', 'quality'],
        ['documentation', 'readme', 'docs', 'guide', 'tutorial', 'example']
    ]
    
    for i in range(min(num_topics, len(topic_labels))):
        words = word_sets[i % len(word_sets)]
        weights = [1.0 - (j * 0.1) for j in range(len(words))]  # Decreasing weights
        
        topics_data.append({
            'topic_id': i,
            'top_words': words,
            'weights': weights,
            'topic_strength': sum(weights),
            'top_10_strength': sum(weights[:10]),
            'label': topic_labels[i]
        })
    
    return True, topics_data

def create_mock_visualization(topics_data, output_path):
    """Create a simple HTML visualization"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mock Claude.md Topic Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .topic {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }}
            .topic h3 {{ margin-top: 0; color: #333; }}
            .words {{ margin: 10px 0; }}
            .word {{ display: inline-block; margin: 2px; padding: 5px 10px; 
                     background: #007bff; color: white; border-radius: 3px; font-size: 14px; }}
        </style>
    </head>
    <body>
        <h1>Mock Claude.md Topic Analysis</h1>
        <p><strong>Note:</strong> This is a mock analysis for local testing without C++ dependencies.</p>
        <p>Total Topics: {len(topics_data)}</p>
    """
    
    for topic in topics_data:
        html_content += f"""
        <div class="topic">
            <h3>Topic {topic['topic_id'] + 1}: {topic['label']}</h3>
            <div class="words">
        """
        for word in topic['top_words'][:8]:  # Show top 8 words
            html_content += f'<span class="word">{word}</span>'
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

# Flask routes
@app.route('/')
def index():
    return """
    <html>
    <body>
        <h1>Mock Claude.md Analyzer</h1>
        <p>Lightweight version for local testing</p>
        <p><a href="/test-mock-analysis">Test Mock Analysis</a></p>
        <p><a href="/visualization">View Mock Visualization</a></p>
    </body>
    </html>
    """

@app.route('/test-mock-analysis')
def test_mock_analysis():
    test_docs = [
        'claude assistant configuration prompt role system behavior instructions',
        'project file directory structure organization folder path management',
        'code function class method style format convention programming',
        'test testing spec validation check verify quality assurance',
        'documentation readme docs guide tutorial example usage help'
    ]
    
    success, topics_data = mock_lda_analysis(test_docs, num_topics=3)
    
    if success:
        # Create visualization
        viz_path = os.path.join('templates', 'mock_visualization.html')
        create_mock_visualization(topics_data, viz_path)
        
        return jsonify({
            "status": "success",
            "message": f"Mock analysis completed with {len(topics_data)} topics",
            "topics": topics_data
        })
    else:
        return jsonify({"status": "error", "message": "Mock analysis failed"})

@app.route('/visualization')
def get_visualization():
    viz_path = os.path.join('templates', 'mock_visualization.html')
    if os.path.exists(viz_path):
        with open(viz_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Mock visualization not generated yet. <a href='/test-mock-analysis'>Run test analysis first</a>", 404

if __name__ == '__main__':
    print("Starting Mock Claude.md Analyzer (no C++ dependencies)")
    print("This version uses simplified implementations for local testing")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))