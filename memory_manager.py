import gc
import psutil
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Any

class MemoryManager:
    """
    Advanced memory management utility for Python applications.
    Provides memory monitoring, garbage collection, and resource cleanup.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, process_name: str = "analyzeclaude"):
        """
        Initialize the memory manager.
        
        Args:
            logger: Optional logger instance for output
            process_name: Name for logging context
        """
        self.logger = logger or logging.getLogger(__name__)
        self.process_name = process_name
        self.start_time = time.time()
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_stats()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory stats in MB
        """
        try:
            memory_info = self.process.memory_info()
            return {
                'rss': round(memory_info.rss / 1024 / 1024, 2),  # Resident Set Size
                'vms': round(memory_info.vms / 1024 / 1024, 2),  # Virtual Memory Size
                'percent': round(self.process.memory_percent(), 2),
                'available': round(psutil.virtual_memory().available / 1024 / 1024, 2),
                'total': round(psutil.virtual_memory().total / 1024 / 1024, 2)
            }
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return {'rss': 0, 'vms': 0, 'percent': 0, 'available': 0, 'total': 0}
    
    def log_memory_usage(self, context: str = '') -> Dict[str, float]:
        """
        Log current memory usage with optional context.
        
        Args:
            context: Optional context string for logging
            
        Returns:
            Current memory statistics
        """
        stats = self.get_memory_stats()
        runtime = round((time.time() - self.start_time), 1)
        
        log_msg = f"📊 Memory{f' ({context})' if context else ''}: "
        log_msg += f"RSS {stats['rss']}MB | "
        log_msg += f"VMS {stats['vms']}MB | "
        log_msg += f"{stats['percent']}% of system | "
        log_msg += f"Available {stats['available']}MB | "
        log_msg += f"Runtime {runtime}s"
        
        self.logger.info(log_msg)
        return stats
    
    def force_garbage_collection(self, context: str = '') -> Dict[str, Any]:
        """
        Force garbage collection and measure impact.
        
        Args:
            context: Optional context for logging
            
        Returns:
            GC results including objects collected
        """
        before_stats = self.get_memory_stats()
        
        # Force full garbage collection
        collected = gc.collect()
        
        after_stats = self.get_memory_stats()
        memory_freed = round(before_stats['rss'] - after_stats['rss'], 2)
        
        log_msg = f"🗑️  GC{f' ({context})' if context else ''}: "
        log_msg += f"Collected {collected} objects"
        
        if memory_freed > 0:
            log_msg += f", freed {memory_freed}MB RSS"
        elif memory_freed < 0:
            log_msg += f", RSS increased by {abs(memory_freed)}MB"
        else:
            log_msg += ", no significant memory change"
            
        self.logger.info(log_msg)
        
        return {
            'objects_collected': collected,
            'memory_freed_mb': memory_freed,
            'before_stats': before_stats,
            'after_stats': after_stats
        }
    
    def monitor_memory_threshold(self, threshold_mb: int = 1000, context: str = '') -> bool:
        """
        Monitor memory usage against a threshold and force GC if exceeded.
        
        Args:
            threshold_mb: Memory threshold in MB
            context: Optional context for logging
            
        Returns:
            True if threshold was exceeded
        """
        stats = self.get_memory_stats()
        
        if stats['rss'] > threshold_mb:
            self.logger.warning(
                f"⚠️  Memory threshold exceeded{f' ({context})' if context else ''}: "
                f"{stats['rss']}MB > {threshold_mb}MB"
            )
            self.force_garbage_collection(f"threshold-{threshold_mb}MB")
            return True
        
        return False
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory usage report.
        
        Returns:
            Detailed memory report
        """
        current_stats = self.get_memory_stats()
        runtime_seconds = round(time.time() - self.start_time, 1)
        runtime_minutes = round(runtime_seconds / 60, 1)
        
        memory_growth = {
            'rss': round(current_stats['rss'] - self.initial_memory['rss'], 2),
            'vms': round(current_stats['vms'] - self.initial_memory['vms'], 2)
        }
        
        # Get garbage collection stats
        gc_stats = gc.get_stats()
        
        return {
            'process_name': self.process_name,
            'current': current_stats,
            'initial': self.initial_memory,
            'runtime': {
                'seconds': runtime_seconds,
                'minutes': runtime_minutes
            },
            'memory_growth': memory_growth,
            'gc_stats': {
                'generations': len(gc_stats),
                'total_collections': sum(stat['collections'] for stat in gc_stats),
                'total_collected': sum(stat['collected'] for stat in gc_stats),
                'total_uncollectable': sum(stat['uncollectable'] for stat in gc_stats)
            },
            'system': {
                'cpu_percent': round(self.process.cpu_percent(), 2),
                'num_threads': self.process.num_threads(),
                'open_files': len(self.process.open_files()) if hasattr(self.process, 'open_files') else 'N/A'
            }
        }
    
    def cleanup_resources(self, context: str = 'cleanup') -> None:
        """
        Perform comprehensive resource cleanup.
        
        Args:
            context: Context for logging
        """
        self.logger.info(f"🧹 Performing resource cleanup ({context})")
        
        # Force garbage collection
        self.force_garbage_collection(context)
        
        # Clear any matplotlib figures (if using plotting)
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
            self.logger.info("🧹 Closed all matplotlib figures")
        except ImportError:
            pass
        
        # Clear NLTK cache if using NLTK
        try:
            import nltk
            if hasattr(nltk.data, 'clear_cache'):
                nltk.data.clear_cache()
                self.logger.info("🧹 Cleared NLTK cache")
        except (ImportError, AttributeError):
            pass
    
    def log_final_report(self) -> Dict[str, Any]:
        """
        Log final memory usage report.
        
        Returns:
            Final memory report
        """
        self.logger.info("📊 === FINAL MEMORY REPORT ===")
        
        report = self.get_memory_report()
        
        self.logger.info(
            f"Final Memory Usage: {report['current']['rss']}MB RSS, "
            f"{report['current']['vms']}MB VMS ({report['current']['percent']}% of system)"
        )
        self.logger.info(
            f"Memory Growth: +{report['memory_growth']['rss']}MB RSS, "
            f"+{report['memory_growth']['vms']}MB VMS"
        )
        self.logger.info(f"Total Runtime: {report['runtime']['minutes']} minutes")
        self.logger.info(
            f"GC Stats: {report['gc_stats']['total_collections']} collections, "
            f"{report['gc_stats']['total_collected']} objects collected"
        )
        
        # Final cleanup
        self.cleanup_resources('final')
        
        return report

# Global memory manager instance
_global_memory_manager = None

def get_memory_manager(logger: Optional[logging.Logger] = None) -> MemoryManager:
    """
    Get or create the global memory manager instance.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        MemoryManager instance
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(logger)
    return _global_memory_manager

def log_memory(context: str = '', logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """
    Convenience function to log memory usage.
    
    Args:
        context: Context string for logging
        logger: Optional logger instance
        
    Returns:
        Memory statistics
    """
    return get_memory_manager(logger).log_memory_usage(context)

def force_gc(context: str = '', logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Convenience function to force garbage collection.
    
    Args:
        context: Context string for logging
        logger: Optional logger instance
        
    Returns:
        GC results
    """
    return get_memory_manager(logger).force_garbage_collection(context)

def monitor_threshold(threshold_mb: int = 1000, context: str = '', 
                     logger: Optional[logging.Logger] = None) -> bool:
    """
    Convenience function to monitor memory threshold.
    
    Args:
        threshold_mb: Memory threshold in MB
        context: Context string for logging
        logger: Optional logger instance
        
    Returns:
        True if threshold was exceeded
    """
    return get_memory_manager(logger).monitor_memory_threshold(threshold_mb, context)