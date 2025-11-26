"""
Shared utility functions for storage layer.

Contains common error handling and helper functions used across storage modules.
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def is_chromadb_compatibility_error(error: Exception) -> bool:
    """
    Check if an exception is a ChromaDB compatibility/frozenset error.
    
    ChromaDB can throw various errors related to metadata compatibility issues,
    particularly frozenset-related errors when metadata is corrupted.
    
    Args:
        error: The exception to check
        
    Returns:
        True if this is a ChromaDB compatibility error, False otherwise
    """
    error_str = str(error).lower()
    
    # Check for various frozenset-related errors:
    # 1. KeyError with "frozenset" in message
    # 2. KeyError with empty message (frozenset() representation)
    # 3. TypeError from frozenset metadata issues
    # 4. General chromadb errors
    is_frozenset_error = (
        "frozenset" in error_str or
        (isinstance(error, KeyError) and (not str(error) or str(error) == "frozenset()")) or
        (isinstance(error, TypeError) and "frozenset" in error_str) or
        "chromadb" in error_str
    )
    
    return is_frozenset_error


def handle_chromadb_init_error(
    error: Exception,
    context: str = "initialization"
) -> Tuple[bool, Optional[RuntimeError]]:
    """
    Handle ChromaDB initialization errors consistently.
    
    If the error is a ChromaDB compatibility issue, logs an appropriate message
    and returns a RuntimeError to raise. Otherwise, returns None.
    
    Args:
        error: The exception that occurred
        context: Description of where the error occurred (for logging)
        
    Returns:
        Tuple of (is_chromadb_error, optional_runtime_error)
        - If is_chromadb_error is True, caller should raise the runtime_error
        - If is_chromadb_error is False, caller should re-raise the original error
    """
    if is_chromadb_compatibility_error(error):
        logger.error(
            f"ChromaDB compatibility issue detected during {context} "
            f"({type(error).__name__}: {error}). "
            "Set RESET_CHROMADB=true in environment variables to fix this."
        )
        
        runtime_error = RuntimeError(
            "ChromaDB initialization failed due to metadata compatibility issue. "
            "Set RESET_CHROMADB=true in your environment variables and redeploy. "
            "This will clear the ChromaDB database and allow it to be recreated."
        )
        runtime_error.__cause__ = error
        
        return True, runtime_error
    
    return False, None


def log_chromadb_warning(error: Exception, context: str = "operation") -> None:
    """
    Log a warning about a non-fatal ChromaDB compatibility issue.
    
    Use this for cases where the operation can continue despite the error.
    
    Args:
        error: The exception that occurred
        context: Description of where the error occurred
    """
    if is_chromadb_compatibility_error(error):
        logger.warning(
            f"ChromaDB compatibility issue detected during {context} "
            f"({type(error).__name__}: {error}). "
            "This may indicate corrupted metadata. Operations will continue."
        )

