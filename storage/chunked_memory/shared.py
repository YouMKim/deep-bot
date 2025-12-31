"""
Shared singleton instance of ChunkedMemoryService.

This module provides a single shared instance of ChunkedMemoryService
to prevent memory bloat from multiple cogs each creating their own instance.

The shared instance is lazily initialized on first access.
"""
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from storage.chunked_memory import ChunkedMemoryService

logger = logging.getLogger(__name__)

# Module-level singleton
_shared_instance: Optional['ChunkedMemoryService'] = None
_initialization_in_progress = False
_access_count = 0  # Track how many times the singleton is accessed


def get_shared_chunked_memory_service() -> 'ChunkedMemoryService':
    """
    Get the shared ChunkedMemoryService instance.
    
    Creates the instance on first call, returns cached instance thereafter.
    This prevents multiple cogs from each loading their own embedding models
    and BM25 caches, significantly reducing memory usage.
    
    Returns:
        Shared ChunkedMemoryService instance
    """
    global _shared_instance, _initialization_in_progress, _access_count
    
    _access_count += 1
    
    if _shared_instance is not None:
        # #region agent log - stdout for Railway
        import psutil
        process = psutil.Process()
        rss_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"[MEMORY DEBUG] Reusing shared ChunkedMemoryService (access #{_access_count}, RSS: {rss_mb:.1f}MB)")
        # #endregion
        return _shared_instance
    
    # Prevent recursive initialization
    if _initialization_in_progress:
        raise RuntimeError("Recursive ChunkedMemoryService initialization detected")
    
    _initialization_in_progress = True
    
    try:
        import psutil
        process = psutil.Process()
        rss_before = process.memory_info().rss / 1024 / 1024
        
        from storage.chunked_memory import ChunkedMemoryService
        from config import Config
        
        logger.info(f"[MEMORY DEBUG] Creating shared ChunkedMemoryService singleton (RSS before: {rss_before:.1f}MB)")
        _shared_instance = ChunkedMemoryService(config=Config)
        
        rss_after = process.memory_info().rss / 1024 / 1024
        logger.info(f"[MEMORY DEBUG] Shared ChunkedMemoryService created (RSS after: {rss_after:.1f}MB, delta: +{rss_after - rss_before:.1f}MB)")
        
        return _shared_instance
    finally:
        _initialization_in_progress = False


def clear_shared_instance():
    """
    Clear the shared instance (useful for testing or cleanup).
    """
    global _shared_instance, _access_count
    _shared_instance = None
    _access_count = 0
    logger.info("[MEMORY DEBUG] Shared ChunkedMemoryService instance cleared")

