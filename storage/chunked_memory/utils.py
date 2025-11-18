"""
Utility functions for chunked memory operations.

Shared helpers for collection naming, strategy resolution, etc.
"""
from typing import Optional
from chunking.constants import ChunkStrategy


def get_collection_name(strategy: str) -> str:
    """
    Get collection name for a chunking strategy.

    Args:
        strategy: Strategy name (e.g., "single", "tokens")

    Returns:
        Collection name (e.g., "discord_chunks_single")
    """
    return f"discord_chunks_{strategy}"


def resolve_strategy(
    strategy: Optional[ChunkStrategy],
    active_strategy: str = "single"
) -> str:
    """
    Resolve strategy to its string value, using active strategy as default.

    Args:
        strategy: Optional strategy override
        active_strategy: Active strategy name (used if strategy is None)

    Returns:
        Strategy string value
    """
    return (strategy or ChunkStrategy(active_strategy)).value


def calculate_fetch_k(
    top_k: int,
    needs_filtering: bool = False,
    needs_reranking: bool = False,
    fetch_multiplier: int = 3
) -> int:
    """
    Calculate how many results to fetch from vector store.

    Fetches more than requested to account for:
    - Filtering (blacklist, author whitelist) - may remove many results
    - Reranking (cross-encoder needs candidates) - needs more options
    - Hybrid search fusion (deduplication) - may merge results

    Args:
        top_k: Number of results requested
        needs_filtering: Whether author filtering will be applied
        needs_reranking: Whether reranking will be applied
        fetch_multiplier: Multiplier for fetch size (default: 3)

    Returns:
        Number of results to fetch
    """
    if needs_filtering or needs_reranking:
        return top_k * fetch_multiplier
    return top_k

