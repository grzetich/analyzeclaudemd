import os
import sys
import requests

# Ensure UTF-8 output on Windows
if os.name == 'nt':  # Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import base64
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
import warnings

from flask import Flask, render_template, send_from_directory, jsonify
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file (for local development)

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# GitHub API configuration
GITHUB_TOKEN = os.getenv("GITHUB_PAT")
if not GITHUB_TOKEN:
    print("WARNING: GITHUB_PAT environment variable not set. GitHub API calls will be severely rate-limited.")
    # For unauthenticated requests, GitHub's rate limit is 60 requests per hour.
    # Authenticated requests (with PAT) get 5,000 requests per hour.
    # Code search has a specific limit of 30 requests per minute with PAT.
    # If not set, the collection step might fail quickly.

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "Accept": "application/vnd.github.v3.text-match+json"
}
SEARCH_QUERY = "filename:claude.md"
BASE_URL = "https://api.github.com/search/code"

# NLTK downloads (run once on startup or when container builds)
def download_nltk_data():
    """Download required NLTK data with error handling"""
    # Set NLTK data path for deployment environments
    if os.name != 'nt':
        nltk_data_dir = '/tmp/nltk_data'
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
    
    datasets = [
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('tokenizers/punkt', 'punkt')
    ]
    
    for path, name in datasets:
        try:
            nltk.data.find(path)
            print(f"NLTK {name} already available")
        except (LookupError, OSError):
            try:
                print(f"Downloading NLTK {name}...")
                nltk.download(name, quiet=True, download_dir='/tmp/nltk_data' if os.name != 'nt' else None)
                print(f"Successfully downloaded NLTK {name}")
            except Exception as e:
                print(f"Failed to download NLTK {name}: {e}")

# Download NLTK data on startup
download_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Global variable to store visualization HTML (cache it after first run)
VIS_HTML_PATH = "/tmp/lda_visualization.html" if os.name != 'nt' else "templates/lda_visualization.html"

# --- Helper Functions (from previous responses) ---

def get_claude_md_files(query, headers, max_files=100): # Limiting for MVP and rate limits
    """
    Searches GitHub for claude.md files and retrieves their content.
    Handles pagination and basic rate limit adherence.
    """
    all_file_contents = []
    page = 1
    per_page = 100 # Max per_page for code search is 100

    print(f"Starting GitHub file collection (max {max_files} files)...")

    while len(all_file_contents) < max_files:
        params = {
            "q": query,
            "page": page,
            "per_page": per_page
        }
          
        response = requests.get(BASE_URL, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])

            if not items:
                print("No more items found on GitHub search.")
                break # No more items, exit loop

            for item in items:
                if len(all_file_contents) >= max_files:
                    break # Stop if max_files limit is reached

                download_url = item.get("download_url")
                if download_url:
                    try:
                        file_response = requests.get(download_url, headers=headers)
                        if file_response.status_code == 200:
                            all_file_contents.append(file_response.text)
                            print(f"  Downloaded: {item['repository']['full_name']}/{item['path']}")
                        else:
                            print(f"  Failed to download {item['path']} from {item['repository']['full_name']}: {file_response.status_code}")
                    except Exception as e:
                        print(f"  Error downloading {item['path']} from {item['repository']['full_name']}: {e}")
                else:
                    print(f"  No download_url found for {item['path']} in {item['repository']['full_name']}. Skipping raw content fetch.")

            # Check if there are more pages
            if "next" in response.links and len(all_file_contents) < max_files:
                page += 1
                # GitHub API best practice: wait a bit between requests to avoid hitting secondary limits
                time.sleep(1) # Small delay
            else:
                break # No more pages

        elif response.status_code == 403:
            if "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                if remaining == 0:
                    reset_time = int(response.headers["X-RateLimit-Reset"])
                    current_time = int(time.time())
                    sleep_duration = max(0, reset_time - current_time + 5) # Add 5 seconds buffer
                    print(f"Rate limit hit ({remaining} requests remaining). Sleeping for {sleep_duration} seconds until {time.ctime(reset_time)}.")
                    time.sleep(sleep_duration)
                    continue # Try again after sleeping
                else:
                    print(f"Error 403 (Forbidden) but {remaining} requests remaining. Check token permissions or other limits: {response.text}")
            else:
                print(f"Error 403 (Forbidden) without rate limit info. Check token or API details: {response.text}")
            break # Break on persistent 403
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break
      
    print(f"Finished collection. Total {len(all_file_contents)} files collected.")
    return all_file_contents

def preprocess_text(text):
    """
    Cleans and preprocesses text for LDA.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def perform_lda_and_visualize(documents, num_topics=5):
    """
    Performs LDA on the given documents and creates a simple HTML visualization.
    """
    if not documents:
        print("No documents provided for LDA analysis.")
        return False

    # Preprocess documents into strings for CountVectorizer
    processed_docs = []
    for doc in documents:
        tokens = preprocess_text(doc)
        if tokens:
            processed_docs.append(' '.join(tokens))
    
    if not processed_docs:
        print("No valid documents after preprocessing for LDA analysis.")
        return False

    print(f"Starting LDA training with {len(processed_docs)} documents...")
    try:
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=1000,
            min_df=5,
            max_df=0.5,
            stop_words='english'
        )
        doc_term_matrix = vectorizer.fit_transform(processed_docs)
        
        # Perform LDA
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=100,
            max_iter=10,
            learning_method='online'
        )
        lda_model.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Create simple HTML visualization
        create_simple_visualization(lda_model, feature_names, num_topics)
        print(f"LDA visualization saved to {VIS_HTML_PATH}")
        return True
    except Exception as e:
        print(f"Error during LDA modeling or visualization: {e}")
        return False

def create_simple_visualization(lda_model, feature_names, num_topics):
    """
    Creates a simple HTML visualization of LDA topics.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LDA Topic Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; }
            .topic { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa; }
            .topic-title { color: #2c3e50; font-size: 18px; font-weight: bold; margin-bottom: 10px; }
            .word { display: inline-block; margin: 2px 5px; padding: 3px 8px; background-color: #3498db; color: white; border-radius: 3px; font-size: 12px; }
            .word-weight { opacity: 0.8; font-size: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Topic Analysis Results</h1>
            <p>Discovered topics and their most representative words:</p>
    """
    
    # Add topics
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        weights = [topic[i] for i in top_words_idx]
        
        html_content += f'<div class="topic"><div class="topic-title">Topic {topic_idx + 1}</div>'
        for word, weight in zip(top_words, weights):
            html_content += f'<span class="word">{word} <span class="word-weight">({weight:.3f})</span></span>'
        html_content += '</div>'
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    try:
        # Ensure directory exists for the file
        os.makedirs(os.path.dirname(VIS_HTML_PATH), exist_ok=True)
        with open(VIS_HTML_PATH, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except Exception as e:
        print(f"Error writing visualization file: {e}")
        raise

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Triggers the data collection, analysis, and visualization.
    Returns a status message.
    """
    if not GITHUB_TOKEN:
        return jsonify({"status": "error", "message": "GitHub PAT is not set. Please configure GITHUB_PAT environment variable."}), 400

    print("Analysis triggered...")
    collected_documents = get_claude_md_files(SEARCH_QUERY, HEADERS, max_files=500) # Limiting to 500 files for MVP on Render's free tier
      
    if not collected_documents:
        return jsonify({"status": "warning", "message": "No claude.md files found or an error occurred during collection. Check logs for details."})

    success = perform_lda_and_visualize(collected_documents, num_topics=5) # Can make num_topics configurable later

    if success:
        return jsonify({"status": "success", "message": "Analysis complete! Refresh the page or click 'View Visualization' to see results."})
    else:
        return jsonify({"status": "error", "message": "Analysis failed. Check server logs for details."}), 500

@app.route('/visualization')
def get_visualization():
    """Serves the generated LDA visualization HTML."""
    if os.path.exists(VIS_HTML_PATH):
        try:
            with open(VIS_HTML_PATH, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading visualization file: {e}")
            return f"Error loading visualization: {e}", 500
    else:
        return "Visualization not yet generated. Please trigger analysis first.", 404

# --- Run the Flask App ---
if __name__ == '__main__':
    # When running locally, Flask uses default server.
    # When deploying on Render, Gunicorn will run the app.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))