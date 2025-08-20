import os
import requests
import base64
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models as gensim_models
import warnings

from flask import Flask, render_template, send_from_directory, jsonify
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file (for local development)

app = Flask(__name__)

# Suppress pyLDAvis deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Global variable to store visualization HTML (cache it after first run)
VIS_HTML_PATH = "templates/lda_visualization.html"

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
    Performs LDA on the given documents and saves the pyLDAvis visualization.
    """
    if not documents:
        print("No documents provided for LDA analysis.")
        return False

    processed_docs = [preprocess_text(doc) for doc in documents]
      
    # Filter out empty processed documents
    processed_docs = [doc for doc in processed_docs if doc]
    if not processed_docs:
        print("No valid documents after preprocessing for LDA analysis.")
        return False

    dictionary = corpora.Dictionary(processed_docs)
    # Filter out tokens that appear in less than no_below documents (absolute number)
    # or in more than no_above fraction of total corpus documents (fraction)
    # and keep only the top keep_n most frequent tokens.
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
      
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    if not corpus:
        print("Corpus is empty after dictionary filtering.")
        return False

    print(f"Starting LDA training with {len(processed_docs)} documents and {len(dictionary)} unique tokens...")
    try:
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            chunksize=2000, # Larger chunksize for potentially larger corpus
            passes=10,
            alpha='auto',
            per_word_topics=True
        )

        vis = gensim_models.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, VIS_HTML_PATH)
        print(f"LDA visualization saved to {VIS_HTML_PATH}")
        return True
    except Exception as e:
        print(f"Error during LDA modeling or visualization: {e}")
        return False

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
    """Serves the generated pyLDAvis HTML."""
    if os.path.exists(VIS_HTML_PATH):
        return send_from_directory('templates', 'lda_visualization.html')
    else:
        return "Visualization not yet generated. Please trigger analysis first.", 404

# --- Run the Flask App ---
if __name__ == '__main__':
    # When running locally, Flask uses default server.
    # When deploying on Render, Gunicorn will run the app.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))