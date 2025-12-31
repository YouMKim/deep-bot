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


def get_shared_chunked_memory_service() -> 'ChunkedMemoryService':
    """
    Get the shared ChunkedMemoryService instance.
    
    Creates the instance on first call, returns cached instance thereafter.
    This prevents multiple cogs from each loading their own embedding models
    and BM25 caches, significantly reducing memory usage.
    
    Returns:
        Shared ChunkedMemoryService instance
    """
    global _shared_instance, _initialization_in_progress
    
    if _shared_instance is not None:
        return _shared_instance
    
    # Prevent recursive initialization
    if _initialization_in_progress:
        raise RuntimeError("Recursive ChunkedMemoryService initialization detected")
    
    _initialization_in_progress = True
    
    try:
        from storage.chunked_memory import ChunkedMemoryService
        from config import Config
        
        logger.info("Creating shared ChunkedMemoryService instance (singleton)")
        _shared_instance = ChunkedMemoryService(config=Config)
        logger.info("Shared ChunkedMemoryService instance created successfully")
        
        return _shared_instance
    finally:
        _initialization_in_progress = False


def clear_shared_instance():
    """
    Clear the shared instance (useful for testing or cleanup).
    """
    global _shared_instance
    _shared_instance = None
    logger.info("Shared ChunkedMemoryService instance cleared")

