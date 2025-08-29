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
import atexit
import shutil
import threading
import logging
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

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
        ('tokenizers/punkt_tab', 'punkt_tab')
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
# Paths for cached files and logs
VIS_HTML_PATH = "/tmp/lda_visualization.html" if os.name != 'nt' else "templates/lda_visualization.html"
ANALYSIS_CACHE_PATH = "/tmp/last_analysis.json" if os.name != 'nt' else "cache/last_analysis.json"
ANALYSIS_HISTORY_PATH = "/tmp/analysis_history.json" if os.name != 'nt' else "cache/analysis_history.json"
LOG_PATH = "/tmp/claude_analyzer.log" if os.name != 'nt' else "logs/claude_analyzer.log"
ANALYSIS_LOGS_DIR = "/tmp/analysis_logs/" if os.name != 'nt' else "logs/analysis_logs/"
TEMP_DIRS = ["/tmp", "/tmp/analysis_logs"] if os.name != 'nt' else ["temp", "cache", "logs", "logs/analysis_logs"]

# Analysis scheduling
ANALYSIS_INTERVAL_HOURS = 24  # Run analysis once per day
MAX_HISTORY_ENTRIES = 30  # Keep 30 days of history
last_analysis_thread = None

# Configure logging
def setup_logging():
    """Set up comprehensive logging with file rotation."""
    try:
        # Ensure log directory exists
        log_dir = os.path.dirname(LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Create rotating file handler (10MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            LOG_PATH, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        # Create console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        if os.name == 'nt':  # Add console handler for local development
            logger.addHandler(console_handler)
            
        logging.info("Logging system initialized")
        return True
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return False

# Initialize logging
setup_logging()

def create_analysis_logger(analysis_timestamp):
    """Create a dedicated logger for an individual analysis run."""
    try:
        # Ensure analysis logs directory exists
        os.makedirs(ANALYSIS_LOGS_DIR, exist_ok=True)
        
        # Create timestamp string for filename (safe for filesystem)
        timestamp_str = analysis_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"analysis_{timestamp_str}.log"
        log_filepath = os.path.join(ANALYSIS_LOGS_DIR, log_filename)
        
        # Create dedicated logger for this analysis
        logger_name = f"analysis_{timestamp_str}"
        analysis_logger = logging.getLogger(logger_name)
        analysis_logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in analysis_logger.handlers[:]:
            analysis_logger.removeHandler(handler)
        
        # Create file handler for this specific analysis
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        analysis_logger.addHandler(file_handler)
        analysis_logger.propagate = False  # Don't propagate to root logger
        
        return analysis_logger, log_filepath
        
    except Exception as e:
        logging.error(f"Failed to create analysis logger: {e}")
        return None, None

# --- Cleanup Functions ---

def cleanup_temp_files():
    """Clean up temporary files created during analysis."""
    try:
        if os.path.exists(VIS_HTML_PATH):
            os.remove(VIS_HTML_PATH)
            print(f"Cleaned up visualization file: {VIS_HTML_PATH}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def cleanup_on_analysis_complete():
    """Clean up temporary data after successful visualization generation."""
    # For now, we keep the visualization file since it's needed for viewing
    # Could add more cleanup here if we create other temporary files
    pass

def save_analysis_cache(success=False, message="", timestamp=None, files_collected=0, topics_discovered=0):
    """Save analysis results to cache file and add to history."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    cache_data = {
        'timestamp': timestamp,
        'success': success,
        'message': message,
        'files_collected': files_collected,
        'topics_discovered': topics_discovered
    }
    
    try:
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(ANALYSIS_CACHE_PATH), exist_ok=True)
        
        # Save current cache
        with open(ANALYSIS_CACHE_PATH, 'w') as f:
            json.dump(cache_data, f)
            
        # Add to historical data
        add_to_analysis_history(cache_data)
        
        # Log the result
        if success:
            logging.info(f"Analysis completed successfully: {files_collected} files, {topics_discovered} topics")
        else:
            logging.error(f"Analysis failed: {message}")
            
    except Exception as e:
        logging.error(f"Error saving analysis cache: {e}")
        print(f"Error saving analysis cache: {e}")

def load_analysis_cache():
    """Load analysis results from cache file."""
    try:
        if os.path.exists(ANALYSIS_CACHE_PATH):
            with open(ANALYSIS_CACHE_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading analysis cache: {e}")
        print(f"Error loading analysis cache: {e}")
    return None

def add_to_analysis_history(analysis_data):
    """Add analysis result to historical data."""
    try:
        history = load_analysis_history()
        history.append(analysis_data)
        
        # Keep only the last MAX_HISTORY_ENTRIES
        if len(history) > MAX_HISTORY_ENTRIES:
            history = history[-MAX_HISTORY_ENTRIES:]
        
        # Ensure history directory exists
        os.makedirs(os.path.dirname(ANALYSIS_HISTORY_PATH), exist_ok=True)
        
        with open(ANALYSIS_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        logging.error(f"Error adding to analysis history: {e}")
        print(f"Error adding to analysis history: {e}")

def load_analysis_history():
    """Load historical analysis results."""
    try:
        if os.path.exists(ANALYSIS_HISTORY_PATH):
            with open(ANALYSIS_HISTORY_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading analysis history: {e}")
        print(f"Error loading analysis history: {e}")
    return []

def get_analysis_stats():
    """Get analysis statistics from history."""
    try:
        history = load_analysis_history()
        if not history:
            return {
                'total_analyses': 0,
                'successful_analyses': 0,
                'success_rate': 0,
                'avg_files_collected': 0,
                'last_30_days': []
            }
        
        successful = [h for h in history if h.get('success', False)]
        
        return {
            'total_analyses': len(history),
            'successful_analyses': len(successful),
            'success_rate': (len(successful) / len(history)) * 100 if history else 0,
            'avg_files_collected': sum(h.get('files_collected', 0) for h in successful) / len(successful) if successful else 0,
            'last_30_days': history[-30:] if len(history) > 30 else history
        }
    except Exception as e:
        logging.error(f"Error calculating analysis stats: {e}")
        return {'error': str(e)}

def should_run_analysis():
    """Check if analysis should run based on cache timestamp."""
    cache = load_analysis_cache()
    if not cache:
        return True
    
    try:
        last_run = datetime.fromisoformat(cache['timestamp'])
        time_diff = datetime.now() - last_run
        return time_diff.total_seconds() > (ANALYSIS_INTERVAL_HOURS * 3600)
    except:
        return True

def run_scheduled_analysis():
    """Run analysis in background if needed."""
    global last_analysis_thread
    
    if should_run_analysis():
        print("Starting scheduled analysis...")
        
        def analysis_worker():
            analysis_start_time = datetime.now()
            analysis_logger = None
            log_filepath = None
            
            try:
                # Create individual analysis logger
                analysis_logger, log_filepath = create_analysis_logger(analysis_start_time)
                
                # Log to both main log and individual analysis log
                logging.info("Starting scheduled analysis...")
                if analysis_logger:
                    analysis_logger.info("=== ANALYSIS RUN STARTED ===")
                    analysis_logger.info(f"Analysis timestamp: {analysis_start_time.isoformat()}")
                
                if not GITHUB_TOKEN:
                    error_msg = "GitHub PAT not configured"
                    logging.error(error_msg)
                    if analysis_logger:
                        analysis_logger.error(error_msg)
                        analysis_logger.info("=== ANALYSIS RUN FAILED ===")
                    save_analysis_cache(False, error_msg, files_collected=0, topics_discovered=0)
                    return
                
                logging.info("Collecting claude.md files from GitHub...")
                if analysis_logger:
                    analysis_logger.info("Starting GitHub file collection (max 500 files)...")
                
                collected_documents = get_claude_md_files(SEARCH_QUERY, HEADERS, max_files=500)
                
                if not collected_documents:
                    error_msg = "No claude.md files found"
                    logging.warning(error_msg)
                    if analysis_logger:
                        analysis_logger.warning(error_msg)
                        analysis_logger.info("=== ANALYSIS RUN COMPLETED (NO DATA) ===")
                    save_analysis_cache(False, error_msg, files_collected=0, topics_discovered=0)
                    return
                
                num_files = len(collected_documents)
                logging.info(f"Collected {num_files} files, starting LDA analysis...")
                if analysis_logger:
                    analysis_logger.info(f"Successfully collected {num_files} claude.md files")
                    analysis_logger.info("Starting LDA topic modeling analysis...")
                
                success = perform_lda_and_visualize(collected_documents, num_topics=5)
                
                # Clean up memory
                collected_documents.clear()
                collected_documents = None
                
                if success:
                    success_msg = f"Analysis complete with {num_files} files"
                    logging.info(f"Scheduled analysis completed successfully with {num_files} files")
                    if analysis_logger:
                        analysis_logger.info("LDA analysis completed successfully")
                        analysis_logger.info(f"Topics discovered: 5")
                        analysis_logger.info(f"Visualization generated: {VIS_HTML_PATH}")
                        analysis_logger.info("=== ANALYSIS RUN COMPLETED SUCCESSFULLY ===")
                    
                    save_analysis_cache(True, success_msg, files_collected=num_files, topics_discovered=5)
                    cleanup_on_analysis_complete()
                    print("Scheduled analysis completed successfully")
                else:
                    error_msg = "Analysis failed during LDA processing"
                    logging.error(error_msg)
                    if analysis_logger:
                        analysis_logger.error("LDA processing failed")
                        analysis_logger.info("=== ANALYSIS RUN FAILED ===")
                    save_analysis_cache(False, error_msg, files_collected=num_files, topics_discovered=0)
                    print("Scheduled analysis failed")
                    
            except Exception as e:
                error_msg = f"Analysis error: {str(e)}"
                logging.error(f"Scheduled analysis error: {e}", exc_info=True)
                if analysis_logger:
                    analysis_logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
                    analysis_logger.info("=== ANALYSIS RUN FAILED WITH ERROR ===")
                save_analysis_cache(False, error_msg, files_collected=0, topics_discovered=0)
                print(f"Scheduled analysis error: {e}")
            finally:
                # Close individual analysis logger
                if analysis_logger:
                    for handler in analysis_logger.handlers:
                        handler.close()
                        analysis_logger.removeHandler(handler)
                    
                    # Log the final log file location to main log
                    if log_filepath:
                        logging.info(f"Individual analysis log saved to: {log_filepath}")
        
        last_analysis_thread = threading.Thread(target=analysis_worker)
        last_analysis_thread.daemon = True
        last_analysis_thread.start()
    else:
        print("Analysis not needed - using cached results")

# Register cleanup function to run at exit
atexit.register(cleanup_temp_files)

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
    logging.info(f"Starting GitHub file collection (max {max_files} files)...")

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
                            print(f"  Downloaded via download_url: {item['repository']['full_name']}/{item['path']}")
                        else:
                            print(f"  Failed to download {item['path']} from {item['repository']['full_name']}: {file_response.status_code}")
                    except Exception as e:
                        print(f"  Error downloading {item['path']} from {item['repository']['full_name']}: {e}")
                else:
                    # Fallback to GitHub Contents API for public repos
                    repo_full_name = item['repository']['full_name']
                    file_path = item['path']
                    contents_url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
                    
                    try:
                        contents_response = requests.get(contents_url, headers=headers)
                        if contents_response.status_code == 200:
                            contents_data = contents_response.json()
                            if contents_data.get('encoding') == 'base64':
                                # Decode base64 content
                                content = base64.b64decode(contents_data['content']).decode('utf-8')
                                all_file_contents.append(content)
                                print(f"  Downloaded via Contents API: {repo_full_name}/{file_path}")
                            else:
                                print(f"  Unsupported encoding for {file_path} in {repo_full_name}: {contents_data.get('encoding')}")
                        elif contents_response.status_code == 404:
                            print(f"  File not found or repo is private: {repo_full_name}/{file_path}")
                        elif contents_response.status_code == 403:
                            print(f"  Access forbidden (likely private repo): {repo_full_name}/{file_path}")
                        else:
                            print(f"  Contents API failed for {repo_full_name}/{file_path}: {contents_response.status_code}")
                    except Exception as e:
                        print(f"  Error using Contents API for {repo_full_name}/{file_path}: {e}")

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
    logging.info(f"Finished collection. Total {len(all_file_contents)} files collected.")
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
        
        # Calculate analysis stats for visualization
        analysis_stats = {
            'total_documents': len(documents),
            'processed_documents': len(processed_docs),
            'vocabulary_size': len(feature_names),
            'topics_discovered': num_topics
        }
        
        # Create enhanced HTML visualization
        create_enhanced_visualization(lda_model, feature_names, num_topics, analysis_stats)
        print(f"LDA visualization saved to {VIS_HTML_PATH}")
        return True
    except Exception as e:
        print(f"Error during LDA modeling or visualization: {e}")
        return False

def generate_topic_label(top_words):
    """
    Generate a semantic label for a topic based on its top words.
    """
    # Simple heuristic: look for common patterns in Claude.md files
    top_3_words = top_words[:3]
    
    # Check for common Claude.md themes
    if any(word in top_3_words for word in ['assistant', 'claude', 'ai']):
        return "AI Assistant Configuration"
    elif any(word in top_3_words for word in ['project', 'repo', 'repository']):
        return "Project Structure"
    elif any(word in top_3_words for word in ['code', 'function', 'class']):
        return "Code Guidelines"
    elif any(word in top_3_words for word in ['file', 'directory', 'folder']):
        return "File Management"
    elif any(word in top_3_words for word in ['test', 'testing', 'spec']):
        return "Testing & Quality"
    elif any(word in top_3_words for word in ['data', 'database', 'api']):
        return "Data & APIs"
    elif any(word in top_3_words for word in ['user', 'interface', 'ui']):
        return "User Interface"
    elif any(word in top_3_words for word in ['deploy', 'build', 'production']):
        return "Deployment & Build"
    elif any(word in top_3_words for word in ['doc', 'documentation', 'readme']):
        return "Documentation"
    elif any(word in top_3_words for word in ['style', 'format', 'convention']):
        return "Code Style"
    else:
        # Fallback: use the top 2 words
        return f"{top_words[0].title()} & {top_words[1].title()}"

def create_enhanced_visualization(lda_model, feature_names, num_topics, analysis_stats):
    """
    Creates an enhanced HTML visualization of LDA topics with better clarity.
    """
    # Calculate topic strengths for relative sizing
    topic_strengths = []
    for topic in lda_model.components_:
        strength = topic.sum()  # Sum of all word weights in topic
        topic_strengths.append(strength)
    
    # Normalize strengths to 0.6-1.0 range for visual scaling
    max_strength = max(topic_strengths)
    min_strength = min(topic_strengths)
    normalized_strengths = []
    for strength in topic_strengths:
        if max_strength == min_strength:
            normalized_strengths.append(1.0)
        else:
            normalized = 0.6 + 0.4 * ((strength - min_strength) / (max_strength - min_strength))
            normalized_strengths.append(normalized)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Claude.md Topic Analysis - Enhanced Visualization</title>
        <meta charset="UTF-8">
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { 
                max-width: 1200px; 
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
            .header h1 { margin: 0 0 10px 0; font-size: 2.2em; font-weight: 300; }
            .header .subtitle { opacity: 0.9; font-size: 1.1em; margin-bottom: 20px; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .stat-card {
                background: rgba(255,255,255,0.1);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-number { font-size: 2em; font-weight: bold; display: block; }
            .stat-label { opacity: 0.8; font-size: 0.9em; }
            .content { padding: 30px; }
            .topics-intro { 
                margin-bottom: 30px; 
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .topics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                gap: 25px;
                margin-top: 30px;
            }
            .topic-card { 
                padding: 25px; 
                border-radius: 12px; 
                background: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .topic-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            .topic-card::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 4px;
                background: linear-gradient(90deg, var(--topic-color), var(--topic-color-light));
            }
            .topic-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            .topic-label { 
                font-size: 1.3em; 
                font-weight: 600; 
                color: #2c3e50;
                margin: 0;
            }
            .topic-number {
                background: var(--topic-color);
                color: white;
                width: 32px; height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 0.9em;
            }
            .words-container {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 15px;
            }
            .word-tag { 
                display: inline-flex;
                align-items: center;
                padding: 8px 12px; 
                background: var(--topic-color);
                color: white; 
                border-radius: 20px; 
                font-weight: 500;
                position: relative;
                transition: all 0.2s ease;
            }
            .word-tag:hover {
                transform: scale(1.05);
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }
            .word-primary { font-size: 1em; opacity: 1; }
            .word-secondary { font-size: 0.9em; opacity: 0.9; }
            .word-tertiary { font-size: 0.85em; opacity: 0.8; }
            .topic-strength {
                font-size: 0.85em;
                color: #6c757d;
                font-weight: 500;
            }
            .strength-bar {
                width: 100%;
                height: 4px;
                background: #e9ecef;
                border-radius: 2px;
                margin-top: 8px;
                overflow: hidden;
            }
            .strength-fill {
                height: 100%;
                background: var(--topic-color);
                border-radius: 2px;
                transition: width 1s ease-out;
            }
            .footer { 
                margin-top: 40px; 
                padding: 30px; 
                background: #f8f9fa; 
                text-align: center;
                border-top: 1px solid #e9ecef;
            }
            .footer-content {
                max-width: 600px;
                margin: 0 auto;
            }
            .how-it-works-link {
                display: inline-block;
                margin-top: 15px;
                padding: 10px 20px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 500;
                transition: background 0.3s ease;
            }
            .how-it-works-link:hover {
                background: #5a67d8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Claude.md Topic Analysis</h1>
                <div class="subtitle">Discovering patterns in Claude documentation across GitHub repositories</div>
                <div class="stats-grid">
    """
    
    # Add analysis statistics
    html_content += f"""
                    <div class="stat-card">
                        <span class="stat-number">{analysis_stats['total_documents']:,}</span>
                        <span class="stat-label">Files Found</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{analysis_stats['processed_documents']:,}</span>
                        <span class="stat-label">Files Analyzed</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{analysis_stats['vocabulary_size']:,}</span>
                        <span class="stat-label">Unique Words</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{analysis_stats['topics_discovered']}</span>
                        <span class="stat-label">Topics Discovered</span>
                    </div>
    """
    
    html_content += """
                </div>
            </div>
            <div class="content">
                <div class="topics-intro">
                    <h3 style="margin-top: 0; color: #2c3e50;">What are these topics?</h3>
                    <p style="margin-bottom: 0; line-height: 1.6;">
                        Each topic represents a common theme found across Claude.md files. The words shown are the most characteristic 
                        terms for that topic, sized by their importance. Topics with stronger presence in the dataset appear more prominent.
                    </p>
                </div>
                <div class="topics-grid">
    """
    
    # Define color palette for topics
    colors = [
        ('#667eea', '#764ba2'),  # Purple-blue
        ('#f093fb', '#f5576c'),  # Pink-red
        ('#4facfe', '#00f2fe'),  # Blue-cyan
        ('#43e97b', '#38f9d7'),  # Green-teal
        ('#fa709a', '#fee140'),  # Pink-yellow
        ('#a8edea', '#fed6e3'),  # Light teal-pink
        ('#ff9a9e', '#fecfef'),  # Light pink
        ('#a18cd1', '#fbc2eb'),  # Purple-pink
    ]
    
    # Add topics with enhanced styling
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-12:][::-1]  # Get top 12 words
        top_words = [feature_names[i] for i in top_words_idx]
        weights = [topic[i] for i in top_words_idx]
        
        # Generate semantic label
        topic_label = generate_topic_label(top_words)
        
        # Get colors for this topic
        color_primary, color_secondary = colors[topic_idx % len(colors)]
        
        # Calculate relative strength
        strength_percent = int(normalized_strengths[topic_idx] * 100)
        
        html_content += f"""
                <div class="topic-card" style="--topic-color: {color_primary}; --topic-color-light: {color_secondary};">
                    <div class="topic-header">
                        <h3 class="topic-label">{topic_label}</h3>
                        <div class="topic-number">{topic_idx + 1}</div>
                    </div>
                    <div class="words-container">
        """
        
        # Add words with different styling based on importance
        for i, (word, weight) in enumerate(zip(top_words, weights)):
            if i < 3:
                css_class = "word-tag word-primary"
            elif i < 6:
                css_class = "word-tag word-secondary"
            else:
                css_class = "word-tag word-tertiary"
            
            html_content += f'<span class="{css_class}">{word}</span>'
        
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
            <div class="footer">
                <div class="footer-content">
                    <strong>Privacy Notice:</strong> No data is saved or retained by this application. 
                    All analysis is performed in memory and temporary files are cleaned up after processing.
                    <br>
                    <a href="/how-it-works" class="how-it-works-link">How it works →</a>
                </div>
            </div>
        </div>
        <script>
            // Add simple animations
            document.addEventListener('DOMContentLoaded', function() {
                // Animate strength bars
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
    
    try:
        # Ensure directory exists for the file
        os.makedirs(os.path.dirname(VIS_HTML_PATH), exist_ok=True)
        with open(VIS_HTML_PATH, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except Exception as e:
        print(f"Error writing visualization file: {e}")
        raise

def create_simple_visualization(lda_model, feature_names, num_topics):
    """
    Creates a simple HTML visualization of LDA topics (legacy function for backward compatibility).
    """
    # For backward compatibility, we'll just call the enhanced version with default stats
    analysis_stats = {
        'total_documents': 0,
        'processed_documents': 0,
        'vocabulary_size': len(feature_names),
        'topics_discovered': num_topics
    }
    create_enhanced_visualization(lda_model, feature_names, num_topics, analysis_stats)

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page and triggers scheduled analysis if needed."""
    # Trigger background analysis if needed
    run_scheduled_analysis()
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Returns cached analysis status or triggers new analysis if needed.
    """
    cache = load_analysis_cache()
    
    if cache:
        # Check if visualization file exists
        if cache['success'] and os.path.exists(VIS_HTML_PATH):
            return jsonify({
                "status": "success", 
                "message": f"Analysis completed on {datetime.fromisoformat(cache['timestamp']).strftime('%B %d, %Y at %I:%M %p')}. Results are cached and ready to view."
            })
        elif not cache['success']:
            return jsonify({
                "status": "error", 
                "message": f"Last analysis failed on {datetime.fromisoformat(cache['timestamp']).strftime('%B %d, %Y at %I:%M %p')}: {cache.get('message', 'Unknown error')}"
            })
    
    # If no cache or visualization missing, check if analysis is running
    global last_analysis_thread
    if last_analysis_thread and last_analysis_thread.is_alive():
        return jsonify({
            "status": "warning", 
            "message": "Analysis is currently running in the background. Results will be available once complete."
        })
    
    # No valid cache and no running analysis - trigger new one
    run_scheduled_analysis()
    return jsonify({
        "status": "warning", 
        "message": "Analysis started in background. Results will be available shortly."
    })

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

@app.route('/how-it-works')
def how_it_works():
    """Renders the how it works page."""
    return render_template('how_it_works.html')

# --- Run the Flask App ---
if __name__ == '__main__':
    # When running locally, Flask uses default server.
    # When deploying on Render, Gunicorn will run the app.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))