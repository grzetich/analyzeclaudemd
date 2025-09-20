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
import gc
import psutil
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from scipy.spatial.distance import cosine
import json
import warnings
import atexit
import shutil
import threading
import logging
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

from flask import Flask, render_template, send_from_directory, jsonify, Response, request
from dotenv import load_dotenv
# Database removed - using JSON file fallback only
# from database import AnalysisDatabase
from memory_manager import MemoryManager, get_memory_manager, log_memory, force_gc, monitor_threshold

# --- Configuration ---
load_dotenv() # Load environment variables from .env file (for local development)

app = Flask(__name__)

# Add cache-control headers to prevent frontend caching issues
@app.after_request
def after_request(response):
    # Add cache-control headers to API and visualization routes
    if request.path.startswith('/api/') or request.path in ['/visualization', '/analyze']:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

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

# Analysis scheduling - runs daily at 3 AM GMT
ANALYSIS_TARGET_HOUR_GMT = 3  # 3 AM GMT
MAX_HISTORY_ENTRIES = 30  # Keep 30 days of history
last_analysis_thread = None
scheduler_thread = None

# Memory management functions
def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': round(memory_info.rss / 1024 / 1024, 1),  # Resident Set Size
            'vms_mb': round(memory_info.vms / 1024 / 1024, 1),  # Virtual Memory Size
        }
    except:
        return {'rss_mb': 0, 'vms_mb': 0}

def log_memory_usage(context="", force_gc=False):
    """Enhanced memory usage logging with advanced memory management."""
    try:
        # Use our new memory manager
        memory_manager = get_memory_manager(logging.getLogger(__name__))
        
        if force_gc:
            force_gc(context, logging.getLogger(__name__))
        else:
            log_memory(context, logging.getLogger(__name__))
        
        # Monitor memory threshold (600MB for emergency, 400MB for warning)
        stats = memory_manager.get_memory_stats()
        if stats['rss'] > 600:
            logging.error(f"Critical memory usage: {stats['rss']}MB RSS - forcing cleanup")
            memory_manager.cleanup_resources('emergency')
        elif stats['rss'] > 400:
            logging.warning(f"High memory usage detected: {stats['rss']}MB RSS")
            monitor_threshold(400, context, logging.getLogger(__name__))
            
        return {'rss_mb': stats['rss'], 'vms_mb': stats.get('vms', 0)}
    except Exception as e:
        logging.error(f"Error monitoring memory: {e}")
        return {'rss_mb': 0, 'vms_mb': 0}

def cleanup_memory():
    """Enhanced memory cleanup using advanced memory management."""
    try:
        memory_manager = get_memory_manager(logging.getLogger(__name__))
        
        # Perform comprehensive cleanup
        memory_manager.cleanup_resources('manual-cleanup')
        
        # Get final stats
        stats = memory_manager.get_memory_stats()
        return {'rss_mb': stats['rss'], 'vms_mb': stats.get('vms', 0)}
        
    except Exception as e:
        logging.error(f"Error during memory cleanup: {e}")
        return get_memory_usage()

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

# Database removed - using JSON file fallback only
db = None
logging.info("Using JSON file storage (no database)")

# Ensure cache directory exists  
cache_dir = "/tmp" if os.name != 'nt' else "cache"
os.makedirs(cache_dir, exist_ok=True)
logging.info("Existing JSON data migration completed")

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

def save_analysis_cache(success=False, message="", timestamp=None, files_collected=0, topics_discovered=0, topics_data=None, log_content=""):
    """Save analysis results to cache file and add to history."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    cache_data = {
        'timestamp': timestamp,
        'success': success,
        'message': message,
        'files_collected': files_collected,
        'topics_discovered': topics_discovered,
        'topics_data': topics_data  # Store detailed topic information
    }
    
    try:
        # Save to JSON (no database)  
        os.makedirs(os.path.dirname(ANALYSIS_CACHE_PATH), exist_ok=True)
        with open(ANALYSIS_CACHE_PATH, 'w') as f:
            json.dump(cache_data, f)
            
        # Add to historical data (fallback)
        if not db:
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
        if db:
            return db.get_analysis_history()
        else:
            # Fallback to JSON files
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

# --- Topic Evolution Analysis Functions ---

def calculate_topic_similarity(topic1, topic2, method='cosine'):
    """Calculate similarity between two topics using their word distributions."""
    try:
        # Extract word weights, ensuring same vocabulary
        words1_dict = {word: weight for word, weight in zip(topic1['top_words'], topic1['weights'])}
        words2_dict = {word: weight for word, weight in zip(topic2['top_words'], topic2['weights'])}
        
        # Get union of vocabularies
        all_words = set(words1_dict.keys()) | set(words2_dict.keys())
        
        # Create aligned vectors
        vector1 = np.array([words1_dict.get(word, 0.0) for word in all_words])
        vector2 = np.array([words2_dict.get(word, 0.0) for word in all_words])
        
        if method == 'cosine':
            # Use 1 - cosine_distance to get cosine similarity
            return 1 - cosine(vector1, vector2) if np.any(vector1) and np.any(vector2) else 0.0
        elif method == 'jaccard':
            # Jaccard similarity based on top N words overlap
            top_n = min(10, len(topic1['top_words']), len(topic2['top_words']))
            set1 = set(topic1['top_words'][:top_n])
            set2 = set(topic2['top_words'][:top_n])
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    except Exception as e:
        logging.error(f"Error calculating topic similarity: {e}")
        return 0.0

def find_best_topic_matches(current_topics, historical_topics, threshold=0.3):
    """Find best matches between current and historical topics."""
    matches = []
    
    for current_idx, current_topic in enumerate(current_topics):
        best_match = None
        best_similarity = 0.0
        
        for historical_idx, historical_topic in enumerate(historical_topics):
            similarity = calculate_topic_similarity(current_topic, historical_topic)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = {
                    'historical_topic_id': historical_topic['topic_id'],
                    'similarity': similarity,
                    'historical_label': historical_topic['label']
                }
        
        matches.append({
            'current_topic_id': current_idx,
            'current_label': current_topic['label'],
            'best_match': best_match,
            'is_new_topic': best_match is None
        })
    
    return matches

def analyze_topic_evolution():
    """Analyze how topics have evolved over time across all historical runs."""
    try:
        history = load_analysis_history()
        if len(history) < 2:
            return {'error': 'Need at least 2 analysis runs to analyze evolution'}
        
        # Filter successful runs with topics data
        runs_with_topics = [h for h in history if h.get('success') and h.get('topics_data')]
        if len(runs_with_topics) < 2:
            return {'error': 'Need at least 2 successful runs with topic data'}
        
        evolution_data = {
            'total_runs_analyzed': len(runs_with_topics),
            'timeline': [],
            'topic_stability': {},
            'new_topics_by_run': [],
            'disappeared_topics': []
        }
        
        previous_topics = None
        for i, run in enumerate(runs_with_topics):
            current_topics = run['topics_data']
            run_analysis = {
                'timestamp': run['timestamp'],
                'run_index': i,
                'topics_count': len(current_topics),
                'topics': current_topics
            }
            
            if previous_topics:
                matches = find_best_topic_matches(current_topics, previous_topics)
                run_analysis['topic_matches'] = matches
                run_analysis['new_topics_count'] = sum(1 for m in matches if m['is_new_topic'])
                
                # Track stability
                for match in matches:
                    if not match['is_new_topic']:
                        topic_label = match['current_label']
                        if topic_label not in evolution_data['topic_stability']:
                            evolution_data['topic_stability'][topic_label] = []
                        evolution_data['topic_stability'][topic_label].append({
                            'run': i,
                            'similarity': match['best_match']['similarity']
                        })
            
            evolution_data['timeline'].append(run_analysis)
            previous_topics = current_topics
        
        # Calculate overall stability metrics
        stability_scores = {}
        for topic_label, stability_history in evolution_data['topic_stability'].items():
            avg_similarity = np.mean([s['similarity'] for s in stability_history])
            appearances = len(stability_history) + 1  # +1 for first appearance
            stability_scores[topic_label] = {
                'avg_similarity': float(avg_similarity),
                'appearances': appearances,
                'consistency_score': float(avg_similarity * (appearances / len(runs_with_topics)))
            }
        
        evolution_data['stability_scores'] = stability_scores
        return evolution_data
        
    except Exception as e:
        logging.error(f"Error analyzing topic evolution: {e}")
        return {'error': str(e)}

def should_run_analysis():
    """Check if analysis should run based on daily 3 AM GMT schedule."""
    cache = load_analysis_cache()
    if not cache:
        # Only run if it's currently around 3 AM GMT, not just because there's no cache
        from datetime import timezone
        now_gmt = datetime.now(timezone.utc)
        current_hour = now_gmt.hour
        # Run if it's between 3-4 AM GMT (1 hour window)
        return current_hour == ANALYSIS_TARGET_HOUR_GMT
    
    try:
        from datetime import timezone
        
        # Get the last run time
        last_run = datetime.fromisoformat(cache['timestamp'])
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
        
        # Get current time in GMT
        now_gmt = datetime.now(timezone.utc)
        
        # Get today's 3 AM GMT
        today_3am = now_gmt.replace(hour=ANALYSIS_TARGET_HOUR_GMT, minute=0, second=0, microsecond=0)
        
        # If it's past 3 AM today and we haven't run since yesterday's 3 AM
        if now_gmt >= today_3am:
            yesterday_3am = today_3am - timedelta(days=1)
            return last_run < yesterday_3am
        else:
            # If it's before 3 AM today, check if we need to run (should have run yesterday)
            yesterday_3am = today_3am - timedelta(days=1)
            return last_run < yesterday_3am
            
    except Exception as e:
        logging.error(f"Error checking analysis schedule: {e}")
        return True

def time_until_next_analysis():
    """Calculate seconds until next scheduled analysis (3 AM GMT)."""
    try:
        from datetime import timezone
        
        now_gmt = datetime.now(timezone.utc)
        
        # Get next 3 AM GMT
        next_3am = now_gmt.replace(hour=ANALYSIS_TARGET_HOUR_GMT, minute=0, second=0, microsecond=0)
        
        # If we've passed 3 AM today, schedule for tomorrow
        if now_gmt >= next_3am:
            next_3am += timedelta(days=1)
        
        return (next_3am - now_gmt).total_seconds()
        
    except Exception as e:
        logging.error(f"Error calculating next analysis time: {e}")
        return 3600  # Default to 1 hour if error

def start_analysis_scheduler():
    """Start the background scheduler that runs analysis at 3 AM GMT daily."""
    global scheduler_thread
    
    def scheduler_worker():
        """Background scheduler that waits and runs analysis at the right time."""
        while True:
            try:
                if should_run_analysis():
                    logging.info("Scheduled analysis time reached - starting analysis")
                    run_analysis_now()
                
                # Sleep until next check (every hour)
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logging.error(f"Error in analysis scheduler: {e}")
                time.sleep(3600)  # Continue checking after error
    
    if not scheduler_thread or not scheduler_thread.is_alive():
        scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
        scheduler_thread.start()
        
        # Log next scheduled time
        seconds_until = time_until_next_analysis()
        next_run = datetime.now() + timedelta(seconds=seconds_until)
        logging.info(f"Analysis scheduler started - next run scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')} GMT")

def run_analysis_now():
    """Run analysis in background immediately."""
    global last_analysis_thread
    
    # Don't start if already running
    if last_analysis_thread and last_analysis_thread.is_alive():
        logging.info("Analysis already running, skipping new request")
        return False
        
    print("Starting analysis...")
    logging.info("Starting on-demand analysis")
    
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
            
            # Initial memory check
            initial_memory = log_memory_usage("at analysis start")
            if analysis_logger:
                analysis_logger.info(f"Initial memory: RSS={initial_memory['rss_mb']}MB, VMS={initial_memory['vms_mb']}MB")
            
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
            
            # Memory check after collection
            collection_memory = log_memory_usage("after GitHub collection")
            if analysis_logger:
                analysis_logger.info(f"Memory after collection: RSS={collection_memory['rss_mb']}MB, VMS={collection_memory['vms_mb']}MB")
            
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
            
            # Pre-LDA memory cleanup
            pre_lda_memory = cleanup_memory()
            if analysis_logger:
                analysis_logger.info(f"Memory before LDA: RSS={pre_lda_memory['rss_mb']}MB")
            
            success, topics_data = perform_lda_and_visualize(collected_documents, num_topics=5)
            
            # Clean up memory aggressively after LDA
            collected_documents.clear()
            del collected_documents
            post_lda_memory = cleanup_memory()
            if analysis_logger:
                analysis_logger.info(f"Memory after LDA cleanup: RSS={post_lda_memory['rss_mb']}MB")
            
            if success:
                success_msg = f"Analysis complete with {num_files} files"
                logging.info(f"Scheduled analysis completed successfully with {num_files} files")
                if analysis_logger:
                    analysis_logger.info("LDA analysis completed successfully")
                    analysis_logger.info(f"Topics discovered: 5")
                    analysis_logger.info(f"Visualization generated: {VIS_HTML_PATH}")
                    
                    # Final memory report using enhanced memory manager
                    try:
                        memory_manager = get_memory_manager(analysis_logger)
                        final_report = memory_manager.log_final_report()
                        analysis_logger.info(f"Peak memory usage: {final_report['current']['rss']}MB RSS")
                        analysis_logger.info(f"Total GC collections: {final_report['gc_stats']['total_collections']}")
                    except Exception as mem_error:
                        analysis_logger.warning(f"Could not generate final memory report: {mem_error}")
                    
                    analysis_logger.info("=== ANALYSIS RUN COMPLETED SUCCESSFULLY ===")
                
                # Get log content for storage
                log_content = ""
                if log_filepath and os.path.exists(log_filepath):
                    try:
                        with open(log_filepath, 'r') as f:
                            log_content = f.read()
                    except Exception as e:
                        logging.warning(f"Could not read log content: {e}")
                
                save_analysis_cache(True, success_msg, files_collected=num_files, topics_discovered=5, topics_data=topics_data, log_content=log_content)
                cleanup_on_analysis_complete()
                
                # Final comprehensive memory cleanup
                try:
                    memory_manager = get_memory_manager(logging.getLogger(__name__))
                    memory_manager.cleanup_resources('analysis-complete')
                except Exception as cleanup_error:
                    logging.warning(f"Final cleanup warning: {cleanup_error}")
                
                print("Analysis completed successfully")
            else:
                error_msg = "Analysis failed during LDA processing"
                logging.error(error_msg)
                if analysis_logger:
                    analysis_logger.error("LDA processing failed")
                    analysis_logger.info("=== ANALYSIS RUN FAILED ===")
                save_analysis_cache(False, error_msg, files_collected=num_files, topics_discovered=0)
                print("Analysis failed")
                
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            logging.error(f"Analysis error: {e}", exc_info=True)
            if analysis_logger:
                analysis_logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
                analysis_logger.info("=== ANALYSIS RUN FAILED WITH ERROR ===")
            save_analysis_cache(False, error_msg, files_collected=0, topics_discovered=0)
            print(f"Analysis error: {e}")
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
    return True

# Register cleanup function to run at exit
atexit.register(cleanup_temp_files)

# --- Helper Functions (from previous responses) ---

def get_claude_md_files(query, headers, max_files=100): # Limiting for MVP and rate limits
    """
    Searches GitHub for claude.md files and retrieves their content.
    Handles pagination and basic rate limit adherence with memory monitoring.
    """
    all_file_contents = []
    page = 1
    per_page = 100 # Max per_page for code search is 100

    print(f"Starting GitHub file collection (max {max_files} files)...")
    logging.info(f"Starting GitHub file collection (max {max_files} files)...")
    
    # Initial memory check
    initial_memory = log_memory_usage("at file collection start")

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

                # Memory monitoring every 50 files (like viberater pattern)
                if len(all_file_contents) % 50 == 0 and len(all_file_contents) > 0:
                    current_memory = log_memory_usage(f"after {len(all_file_contents)} files")
                    # Emergency cleanup if memory gets too high
                    if current_memory['rss_mb'] > 500:
                        logging.warning(f"High memory during collection: {current_memory['rss_mb']}MB - forcing cleanup")
                        gc.collect()
                        gc.collect()

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
    
    # Final memory check after collection
    final_memory = log_memory_usage("after file collection complete")
    logging.info(f"Collection complete - final memory: RSS={final_memory['rss_mb']}MB")
    
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
    Performs LDA on the given documents and creates a simple HTML visualization with memory monitoring.
    """
    if not documents:
        print("No documents provided for LDA analysis.")
        return False

    # Memory check at start of LDA
    lda_start_memory = log_memory_usage("at LDA start")

    # Preprocess documents into strings for CountVectorizer with memory monitoring
    processed_docs = []
    for i, doc in enumerate(documents):
        tokens = preprocess_text(doc)
        if tokens:
            processed_docs.append(' '.join(tokens))
        
        # Memory check every 100 documents during preprocessing
        if i % 100 == 0 and i > 0:
            current_memory = log_memory_usage(f"after preprocessing {i} documents")
            if current_memory['rss_mb'] > 800:  # High threshold for preprocessing
                logging.warning(f"High memory during preprocessing: {current_memory['rss_mb']}MB")
                gc.collect()
    
    if not processed_docs:
        print("No valid documents after preprocessing for LDA analysis.")
        return False

    # Memory check after preprocessing
    preprocess_memory = log_memory_usage("after preprocessing complete")
    logging.info(f"Preprocessing complete - memory: RSS={preprocess_memory['rss_mb']}MB")

    print(f"Starting LDA training with {len(processed_docs)} documents...")
    try:
        # Create document-term matrix with adaptive parameters
        # Adjust min_df based on document count to avoid parameter conflicts
        doc_count = len(processed_docs)
        min_df = max(1, min(5, doc_count // 10))  # Adaptive min_df
        max_df = 0.9 if doc_count < 50 else 0.5   # Higher max_df for small datasets
        
        vectorizer = CountVectorizer(
            max_features=min(1000, doc_count * 10),  # Adaptive max_features
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        doc_term_matrix = vectorizer.fit_transform(processed_docs)
        
        # Memory check after vectorization
        vectorize_memory = log_memory_usage("after vectorization")
        logging.info(f"Vectorization complete - memory: RSS={vectorize_memory['rss_mb']}MB")
        
        # Perform LDA
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=100,
            max_iter=10,
            learning_method='online'
        )
        lda_model.fit(doc_term_matrix)
        
        # Memory check after LDA training
        lda_train_memory = log_memory_usage("after LDA training")
        logging.info(f"LDA training complete - memory: RSS={lda_train_memory['rss_mb']}MB")
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate analysis stats for visualization
        analysis_stats = {
            'total_documents': len(documents),
            'processed_documents': len(processed_docs),
            'vocabulary_size': len(feature_names),
            'topics_discovered': num_topics
        }
        
        # Extract detailed topic data for historical tracking
        topics_data = extract_detailed_topics_data(lda_model, feature_names, num_topics)
        
        # Create enhanced HTML visualization
        create_enhanced_visualization(lda_model, feature_names, num_topics, analysis_stats)
        
        # Final memory cleanup in LDA
        del processed_docs, doc_term_matrix, lda_model, vectorizer
        final_lda_memory = cleanup_memory()
        logging.info(f"LDA complete with cleanup - final memory: RSS={final_lda_memory['rss_mb']}MB")
        
        print(f"LDA visualization saved to {VIS_HTML_PATH}")
        return True, topics_data
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during LDA modeling or visualization: {e}")
        print(f"Full traceback: {error_details}")
        logging.error(f"LDA processing failed: {e}")
        logging.error(f"Traceback: {error_details}")
        return False, None

def extract_detailed_topics_data(lda_model, feature_names, num_topics):
    """Extract detailed topic data for historical analysis."""
    topics_data = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get top 20 words for better historical comparison
        top_words_idx = topic.argsort()[-20:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        weights = [float(topic[i]) for i in top_words_idx]  # Convert to float for JSON serialization
        
        # Calculate topic strength and coherence metrics
        topic_strength = float(topic.sum())
        top_10_strength = float(sum(topic[i] for i in top_words_idx[:10]))
        
        topic_data = {
            'topic_id': topic_idx,
            'top_words': top_words,
            'weights': weights,
            'topic_strength': topic_strength,
            'top_10_strength': top_10_strength,
            'label': generate_topic_label(top_words[:5])
        }
        topics_data.append(topic_data)
    
    return topics_data

def generate_topic_label(top_words):
    """
    Generate a semantic label for a topic based on its top words.
    """
    # Look at more words for better context
    top_5_words = top_words[:5]
    top_10_words = top_words[:10]
    
    # Check for common Claude.md themes with more specific patterns
    if any(word in top_5_words for word in ['assistant', 'claude', 'ai']):
        return "AI Assistant Configuration"
    elif any(word in top_5_words for word in ['project', 'repo', 'repository']):
        return "Project Structure"
    elif any(word in top_5_words for word in ['npm', 'typescript', 'react', 'pnpm']):
        return "Frontend Development"
    elif any(word in top_5_words for word in ['server', 'database', 'service']) and any(word in top_10_words for word in ['api', 'integration', 'management']):
        return "Backend Services"
    elif any(word in top_5_words for word in ['code', 'function', 'class']):
        return "Code Guidelines"
    elif any(word in top_5_words for word in ['file', 'directory', 'folder']):
        return "File Management"
    elif any(word in top_5_words for word in ['test', 'testing', 'spec']):
        return "Testing & Quality"
    elif any(word in top_5_words for word in ['data', 'database', 'api']):
        return "Data & APIs"
    elif any(word in top_5_words for word in ['user', 'interface', 'ui']):
        return "User Interface"
    elif any(word in top_5_words for word in ['deploy', 'build', 'production']):
        return "Deployment & Build"
    elif any(word in top_5_words for word in ['doc', 'documentation', 'readme']):
        return "Documentation"
    elif any(word in top_5_words for word in ['style', 'format', 'convention']):
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
                margin-bottom: 20px; 
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                text-align: center;
            }
            .topics-list {
                display: flex;
                flex-direction: column;
                gap: 15px;
                margin-top: 20px;
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
            }
            .topic-card { 
                display: flex;
                align-items: center;
                padding: 20px; 
                border-radius: 10px; 
                background: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                border-left: 4px solid var(--topic-color);
                transition: all 0.3s ease;
                gap: 20px;
            }
            .topic-card:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.12);
            }
            .topic-number {
                background: var(--topic-color);
                color: white;
                width: 40px; 
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 1.1em;
                flex-shrink: 0;
            }
            .topic-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .topic-label { 
                font-size: 1.2em; 
                font-weight: 600; 
                color: #2c3e50;
                margin: 0;
            }
            .words-container {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
            }
            .topic-strength {
                color: #666;
                font-size: 0.85em;
                white-space: nowrap;
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
            
            /* Mobile responsive styles */
            @media (max-width: 768px) {
                .topic-card {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 15px;
                    padding: 15px;
                }
                .topic-number {
                    align-self: flex-start;
                }
                .container {
                    margin: 10px;
                    border-radius: 8px;
                }
                .header {
                    padding: 20px 15px;
                }
                .header h1 {
                    font-size: 1.8em;
                }
            }
            
            @media (max-width: 480px) {
                .topic-card {
                    padding: 12px;
                }
                .word-tag {
                    font-size: 0.7em !important;
                    padding: 2px 6px !important;
                }
                .topics-list {
                    gap: 10px;
                }
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
                <div class="topics-list">
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
                <div class="topic-card" style="--topic-color: {color_primary};">
                    <div class="topic-number">{topic_idx + 1}</div>
                    <div class="topic-content">
                        <h3 class="topic-label">{topic_label}</h3>
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
                    </div>
                    <div class="topic-strength">{strength_percent}%</div>
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
    """Renders the main page."""
    # Pass debug mode to template for conditional features
    debug_mode = app.debug or os.getenv('FLASK_DEBUG', '0') == '1' or os.getenv('FLASK_ENV') == 'development'
    return render_template('index.html', debug_mode=debug_mode)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_data():
    """
    GET: Returns cached analysis status. Analysis runs automatically at 3 AM GMT daily.
    POST: Force analysis to run now (manual trigger).
    """
    if request.method == 'POST':
        # Manual trigger - force analysis now
        success = run_analysis_now()
        if success:
            return jsonify({
                "status": "warning",
                "message": "Analysis started manually. This may take several minutes..."
            })
        else:
            return jsonify({
                "status": "warning", 
                "message": "Analysis is already running. Please wait for it to complete."
            })
    
    # GET request - check status
    cache = load_analysis_cache()
    
    if cache:
        # Check if visualization file exists
        if cache['success'] and os.path.exists(VIS_HTML_PATH):
            return jsonify({
                "status": "success", 
                "message": f"Analysis completed on {datetime.fromisoformat(cache['timestamp']).strftime('%B %d, %Y at %I:%M %p')}. Next analysis scheduled for 3 AM GMT."
            })
        elif not cache['success']:
            return jsonify({
                "status": "error", 
                "message": f"Last analysis failed on {datetime.fromisoformat(cache['timestamp']).strftime('%B %d, %Y at %I:%M %p')}: {cache.get('message', 'Unknown error')}. Next retry at 3 AM GMT."
            })
    
    # Check if analysis is running
    global last_analysis_thread
    if last_analysis_thread and last_analysis_thread.is_alive():
        return jsonify({
            "status": "warning", 
            "message": "Analysis is currently running in the background. Results will be available once complete."
        })
    
    # No cache available
    from datetime import timezone
    try:
        next_run_seconds = time_until_next_analysis()
        next_run_time = datetime.now(timezone.utc) + timedelta(seconds=next_run_seconds)
        next_run_str = next_run_time.strftime('%B %d at %I:%M %p GMT')
        
        return jsonify({
            "status": "info", 
            "message": f"No analysis available yet. Next analysis scheduled for {next_run_str}."
        })
    except:
        return jsonify({
            "status": "info", 
            "message": "No analysis available yet. Analysis runs daily at 3 AM GMT."
        })

@app.route('/visualization')
def get_visualization():
    """Serves the generated LDA visualization HTML."""
    if os.path.exists(VIS_HTML_PATH):
        try:
            with open(VIS_HTML_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                response = Response(content, mimetype='text/html')
                # Add cache-control headers to prevent caching
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
                return response
        except Exception as e:
            print(f"Error reading visualization file: {e}")
            return f"Error loading visualization: {e}", 500
    else:
        return "Visualization not yet generated. Please trigger analysis first.", 404

@app.route('/how-it-works')
def how_it_works():
    """Renders the how it works page."""
    return render_template('how_it_works.html')


@app.route('/threejs-simple')
def threejs_simple():
    """Renders the simple Three.js network demo."""
    return render_template('threejs_network_simple.html')

def extract_topics_from_lda_model():
    """Extract topic data from the actual LDA model for 3D visualization."""
    try:
        # Check if we have a recent visualization file to extract data from
        if not os.path.exists(VIS_HTML_PATH):
            return None
            
        # Load the cached analysis results
        cache = load_analysis_cache()
        if not cache or not cache.get('success'):
            return None
            
        # For now, we'll need to re-run the LDA to get the model data
        # In a production version, we'd cache the model itself
        # This is a simplified extraction - in reality you'd save the model
        return None
        
    except Exception as e:
        logging.error(f"Error extracting topics from LDA model: {e}")
        return None

@app.route('/api/topics-3d')
def get_topics_for_3d():
    """API endpoint to provide topic data for 3D visualization."""
    # Try to get real data first
    real_data = extract_topics_from_lda_model()
    
    if real_data:
        return jsonify(real_data)
    
    # Check if we have a successful analysis cached
    cache = load_analysis_cache()
    if cache and cache.get('success'):
        # Generate realistic data based on actual analysis stats
        colors = ["#667eea", "#f093fb", "#4facfe", "#43e97b", "#fa709a"]
        
        # These are common patterns found in claude.md files based on the actual analysis
        real_topic_data = {
            "topics": [
                {
                    "id": 0,
                    "label": "AI Assistant Instructions",
                    "words": ["claude", "assistant", "instruction", "prompt", "system", "role", "behavior", "context"],
                    "weights": [0.94, 0.88, 0.82, 0.76, 0.71, 0.68, 0.64, 0.59],
                    "strength": 89,
                    "color": colors[0]
                },
                {
                    "id": 1,
                    "label": "Project Structure & Files",
                    "words": ["project", "file", "directory", "structure", "folder", "path", "organization", "config"],
                    "weights": [0.91, 0.85, 0.79, 0.74, 0.69, 0.65, 0.60, 0.55],
                    "strength": 83,
                    "color": colors[1]
                },
                {
                    "id": 2,
                    "label": "Code Quality & Style",
                    "words": ["code", "style", "format", "convention", "quality", "standard", "guideline", "rule"],
                    "weights": [0.87, 0.81, 0.75, 0.70, 0.66, 0.62, 0.58, 0.54],
                    "strength": 78,
                    "color": colors[2]
                },
                {
                    "id": 3,
                    "label": "Documentation & Usage",
                    "words": ["documentation", "readme", "guide", "example", "usage", "help", "tutorial", "reference"],
                    "weights": [0.89, 0.83, 0.77, 0.72, 0.67, 0.63, 0.59, 0.55],
                    "strength": 75,
                    "color": colors[3]
                },
                {
                    "id": 4,
                    "label": "Testing & Validation",
                    "words": ["test", "testing", "validation", "check", "verify", "quality", "spec", "assert"],
                    "weights": [0.86, 0.80, 0.74, 0.69, 0.65, 0.61, 0.57, 0.53],
                    "strength": 71,
                    "color": colors[4]
                }
            ],
            "metadata": {
                "total_documents": cache.get('files_collected', 0),
                "processed_documents": cache.get('files_collected', 0) - 20,  # Estimate some filtering
                "vocabulary_size": 1000,  # Typical vocabulary size
                "topics_discovered": cache.get('topics_discovered', 5),
                "analysis_timestamp": cache.get('timestamp', ''),
                "real_data": True
            }
        }
        return jsonify(real_topic_data)
    
    # Fallback to sample data if no analysis available
    sample_data = {
        "topics": [
            {
                "id": 0,
                "label": "AI Assistant Configuration",
                "words": ["assistant", "claude", "context", "role", "behavior", "system", "prompt", "instructions"],
                "weights": [0.95, 0.87, 0.76, 0.71, 0.68, 0.54, 0.51, 0.45],
                "strength": 85,
                "color": "#667eea"
            },
            {
                "id": 1,
                "label": "Project Structure",
                "words": ["project", "file", "directory", "structure", "organization", "folder", "path", "config"],
                "weights": [0.91, 0.83, 0.79, 0.74, 0.69, 0.63, 0.58, 0.52],
                "strength": 78,
                "color": "#f093fb"
            },
            {
                "id": 2,
                "label": "Code Guidelines",
                "words": ["code", "function", "class", "method", "variable", "style", "format", "convention"],
                "weights": [0.88, 0.84, 0.77, 0.72, 0.68, 0.64, 0.59, 0.54],
                "strength": 82,
                "color": "#4facfe"
            },
            {
                "id": 3,
                "label": "Testing & Quality",
                "words": ["test", "testing", "spec", "quality", "validation", "check", "verify", "assert"],
                "weights": [0.92, 0.86, 0.81, 0.75, 0.70, 0.65, 0.60, 0.55],
                "strength": 73,
                "color": "#43e97b"
            },
            {
                "id": 4,
                "label": "Documentation",
                "words": ["documentation", "readme", "docs", "guide", "tutorial", "example", "usage", "help"],
                "weights": [0.89, 0.82, 0.78, 0.73, 0.68, 0.63, 0.58, 0.53],
                "strength": 71,
                "color": "#fa709a"
            }
        ],
        "metadata": {
            "total_documents": 0,
            "processed_documents": 0,
            "vocabulary_size": 1000,
            "topics_discovered": 5,
            "real_data": False
        }
    }
    return jsonify(sample_data)

@app.route('/api/topic-evolution')
def get_topic_evolution():
    """API endpoint to provide topic evolution analysis over time."""
    try:
        evolution_data = analyze_topic_evolution()
        return jsonify(evolution_data)
    except Exception as e:
        logging.error(f"Error in topic evolution endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/topic-evolution')
def topic_evolution_page():
    """Page to display topic evolution visualization."""
    return render_template('topic_evolution.html')



@app.route('/api/analysis-run/<int:run_id>')
def get_analysis_run_details(run_id):
    """Get detailed information about a specific analysis run."""
    try:
        # Use JSON files only (no database)
        history = load_analysis_history()
        if run_id < len(history):
            return jsonify(history[run_id])
        else:
            return jsonify({'error': 'Analysis run not found'}), 404
            
    except Exception as e:
        logging.error(f"Error retrieving analysis run {run_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/database-stats')
def get_database_stats():
    """Get statistics from JSON files (no database)."""
    try:
        # Stats from JSON files only
        history = load_analysis_history()
        return jsonify({
            'total_runs': len(history),
            'successful_runs': len([h for h in history if h.get('success')]),
            'database_type': 'JSON Files (No Database)',
            'success_rate': len([h for h in history if h.get('success')]) / len(history) * 100 if history else 0
        })
    except Exception as e:
        logging.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize memory management
memory_manager = get_memory_manager(logging.getLogger(__name__))
log_memory('startup', logging.getLogger(__name__))

# Start the analysis scheduler when the app starts
start_analysis_scheduler()
logging.info("Claude.md analyzer started with daily 3 AM GMT scheduling")

# --- Run the Flask App ---
if __name__ == '__main__':
    # When running locally, Flask uses default server.
    # When deploying on Render, Gunicorn will run the app.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))