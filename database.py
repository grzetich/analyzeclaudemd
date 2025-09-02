import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

import sqlite3

class AnalysisDatabase:
    """Database handler for persistent analysis storage on Render.com with PostgreSQL fallback to SQLite."""
    
    def __init__(self):
        # Use PostgreSQL on Render.com, SQLite locally
        self.database_url = os.getenv('DATABASE_URL')
        self.use_postgres = POSTGRES_AVAILABLE and bool(self.database_url)
        
        if not self.use_postgres:
            self.db_path = "analysis.db"
            
        self.init_database()
        
    def get_connection(self):
        """Get database connection based on environment."""
        if self.use_postgres:
            return psycopg2.connect(self.database_url)
        else:
            return sqlite3.connect(self.db_path)
            
    def init_database(self):
        """Initialize database with required tables."""
        try:
            if self.use_postgres:
                self._init_postgres()
            else:
                self._init_sqlite()
            logging.info(f"Database initialized successfully ({'PostgreSQL' if self.use_postgres else 'SQLite'})")
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            
    def _init_postgres(self):
        """Initialize PostgreSQL database."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_runs (
                        id SERIAL PRIMARY KEY,
                        timestamp VARCHAR(50) NOT NULL,
                        success BOOLEAN NOT NULL,
                        message TEXT,
                        files_collected INTEGER,
                        topics_discovered INTEGER,
                        topics_data JSONB,
                        log_content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Add index for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_analysis_runs_success 
                    ON analysis_runs(success, created_at)
                ''')
                
                conn.commit()
                
    def _init_sqlite(self):
        """Initialize SQLite database."""
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    message TEXT,
                    files_collected INTEGER,
                    topics_discovered INTEGER,
                    topics_data TEXT,
                    log_content TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            
    def store_analysis_run(self, timestamp: str, success: bool, message: str, 
                          files_collected: int, topics_discovered: int, 
                          topics_data: Optional[List[Dict]], log_content: str = "") -> int:
        """Store an analysis run in the database."""
        try:
            if self.use_postgres:
                return self._store_postgres(timestamp, success, message, files_collected, 
                                          topics_discovered, topics_data, log_content)
            else:
                return self._store_sqlite(timestamp, success, message, files_collected, 
                                        topics_discovered, topics_data, log_content)
        except Exception as e:
            logging.error(f"Error storing analysis run: {e}")
            return -1
            
    def _store_postgres(self, timestamp: str, success: bool, message: str, 
                       files_collected: int, topics_discovered: int, 
                       topics_data: Optional[List[Dict]], log_content: str) -> int:
        """Store analysis run in PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO analysis_runs 
                    (timestamp, success, message, files_collected, topics_discovered, topics_data, log_content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (timestamp, success, message, files_collected, topics_discovered, 
                      json.dumps(topics_data) if topics_data else None, log_content))
                
                run_id = cursor.fetchone()[0]
                conn.commit()
                
                logging.info(f"Stored analysis run with ID: {run_id}")
                return run_id
                
    def _store_sqlite(self, timestamp: str, success: bool, message: str, 
                     files_collected: int, topics_discovered: int, 
                     topics_data: Optional[List[Dict]], log_content: str) -> int:
        """Store analysis run in SQLite."""
        topics_json = json.dumps(topics_data) if topics_data else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_runs 
                (timestamp, success, message, files_collected, topics_discovered, topics_data, log_content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, success, message, files_collected, topics_discovered, topics_json, log_content))
            
            run_id = cursor.lastrowid
            conn.commit()
            
            logging.info(f"Stored analysis run with ID: {run_id}")
            return run_id
            
    def get_analysis_history(self, limit: int = 100) -> List[Dict]:
        """Retrieve analysis history from database."""
        try:
            if self.use_postgres:
                return self._get_history_postgres(limit)
            else:
                return self._get_history_sqlite(limit)
        except Exception as e:
            logging.error(f"Error retrieving analysis history: {e}")
            return []
            
    def _get_history_postgres(self, limit: int) -> List[Dict]:
        """Get analysis history from PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute('''
                    SELECT id, timestamp, success, message, files_collected, 
                           topics_discovered, topics_data, created_at
                    FROM analysis_runs 
                    WHERE success = true 
                    ORDER BY created_at DESC 
                    LIMIT %s
                ''', (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
    def _get_history_sqlite(self, limit: int) -> List[Dict]:
        """Get analysis history from SQLite."""
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, success, message, files_collected, 
                       topics_discovered, topics_data, created_at
                FROM analysis_runs 
                WHERE success = 1 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            history = []
            
            for row in rows:
                run_data = dict(row)
                if run_data['topics_data']:
                    try:
                        run_data['topics_data'] = json.loads(run_data['topics_data'])
                    except json.JSONDecodeError:
                        logging.warning(f"Invalid JSON in topics_data for run {row['id']}")
                
                history.append(run_data)
                
            return history
            
    def get_analysis_run(self, run_id: int) -> Optional[Dict]:
        """Get a specific analysis run with full details."""
        try:
            if self.use_postgres:
                return self._get_run_postgres(run_id)
            else:
                return self._get_run_sqlite(run_id)
        except Exception as e:
            logging.error(f"Error retrieving analysis run {run_id}: {e}")
            return None
            
    def _get_run_postgres(self, run_id: int) -> Optional[Dict]:
        """Get analysis run from PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute('SELECT * FROM analysis_runs WHERE id = %s', (run_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
                
    def _get_run_sqlite(self, run_id: int) -> Optional[Dict]:
        """Get analysis run from SQLite."""
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM analysis_runs WHERE id = ?', (run_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            run_data = dict(row)
            if run_data['topics_data']:
                try:
                    run_data['topics_data'] = json.loads(run_data['topics_data'])
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON in topics_data for run {run_id}")
                    
            return run_data
            
    def migrate_json_data(self, analysis_history_path: str):
        """Migrate data from JSON files to database (for upgrading existing deployments)."""
        try:
            if not os.path.exists(analysis_history_path):
                return
                
            with open(analysis_history_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for run in data:
                    # Check if already exists by timestamp
                    existing = self.get_by_timestamp(run.get('timestamp', ''))
                    if not existing:
                        self.store_analysis_run(
                            timestamp=run.get('timestamp', ''),
                            success=run.get('success', False),
                            message=run.get('message', ''),
                            files_collected=run.get('files_collected', 0),
                            topics_discovered=run.get('topics_discovered', 0),
                            topics_data=run.get('topics_data'),
                            log_content=""
                        )
                        
            logging.info(f"Migrated data from {analysis_history_path}")
            
        except Exception as e:
            logging.error(f"Error migrating JSON data: {e}")
            
    def get_by_timestamp(self, timestamp: str) -> Optional[Dict]:
        """Check if run with timestamp already exists."""
        try:
            if self.use_postgres:
                with self.get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute('SELECT id FROM analysis_runs WHERE timestamp = %s', (timestamp,))
                        row = cursor.fetchone()
                        return dict(row) if row else None
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT id FROM analysis_runs WHERE timestamp = ?', (timestamp,))
                    row = cursor.fetchone()
                    return {'id': row[0]} if row else None
        except Exception as e:
            logging.error(f"Error checking timestamp: {e}")
            return None
            
    def export_data(self) -> Dict:
        """Export all data for download."""
        history = self.get_analysis_history(limit=1000)
        return {
            'export_timestamp': datetime.now().isoformat(),
            'total_runs': len(history),
            'analysis_history': history
        }
        
    def get_stats(self) -> Dict:
        """Get database statistics."""
        try:
            if self.use_postgres:
                return self._get_stats_postgres()
            else:
                return self._get_stats_sqlite()
        except Exception as e:
            logging.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
            
    def _get_stats_postgres(self) -> Dict:
        """Get statistics from PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT COUNT(*) FROM analysis_runs')
                total_runs = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM analysis_runs WHERE success = true')
                successful_runs = cursor.fetchone()[0]
                
                cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM analysis_runs')
                date_range = cursor.fetchone()
                
                return {
                    'total_runs': total_runs,
                    'successful_runs': successful_runs,
                    'success_rate': (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                    'database_type': 'PostgreSQL',
                    'date_range': {
                        'first_run': date_range[0].isoformat() if date_range[0] else None,
                        'last_run': date_range[1].isoformat() if date_range[1] else None
                    }
                }
                
    def _get_stats_sqlite(self) -> Dict:
        """Get statistics from SQLite."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM analysis_runs')
            total_runs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM analysis_runs WHERE success = 1')
            successful_runs = cursor.fetchone()[0]
            
            cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM analysis_runs')
            date_range = cursor.fetchone()
            
            return {
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate': (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                'database_type': 'SQLite',
                'date_range': {
                    'first_run': date_range[0],
                    'last_run': date_range[1]
                }
            }