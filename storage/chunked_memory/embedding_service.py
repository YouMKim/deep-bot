"""
Embedding service for chunked memory.

Handles document embedding with fallback and batching support.
"""
import logging
import asyncio
from collections import OrderedDict
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
from embedding.base import EmbeddingBase

if TYPE_CHECKING:
    from config import Config


class EmbeddingService:
    """Service for embedding documents with error recovery, batching, and caching."""

    # Class-level cache settings
    QUERY_CACHE_SIZE = 500  # Max number of query embeddings to cache

    def __init__(
        self,
        embedder: EmbeddingBase,
        config: Optional['Config'] = None
    ):
        """
        Initialize EmbeddingService.

        Args:
            embedder: Embedder instance for encoding documents
            config: Configuration instance (defaults to Config class)
        """
        from config import Config as ConfigClass
        
        self.embedder = embedder
        self.config = config or ConfigClass
        self.logger = logging.getLogger(__name__)
        
        # Embedding failure tracking
        self._embedding_failure_count = 0
        self._embedding_success_count = 0
        
        # LRU cache for query embeddings (queries are repeated frequently)
        self._query_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def encode_query_cached(self, query: str) -> List[float]:
        """
        Encode a query with LRU caching.
        
        Query embeddings are frequently repeated (same questions, multi-query
        variations, etc.), so caching them provides significant speedup.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector for the query
        """
        # Check cache first
        if query in self._query_cache:
            # Move to end (most recently used) for LRU
            self._query_cache.move_to_end(query)
            self._cache_hits += 1
            self.logger.debug(f"Query embedding cache hit (hits: {self._cache_hits})")
            return self._query_cache[query]
        
        # Cache miss - compute embedding
        self._cache_misses += 1
        embedding = self.embedder.encode(query)
        
        # Add to cache
        self._query_cache[query] = embedding
        
        # Evict oldest if over limit
        if len(self._query_cache) > self.QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        
        return embedding
    
    async def encode_query_cached_async(self, query: str) -> List[float]:
        """
        Async version of encode_query_cached.
        
        Runs the embedding in an executor to avoid blocking the event loop.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector for the query
        """
        # Check cache first (no need for executor)
        if query in self._query_cache:
            self._query_cache.move_to_end(query)
            self._cache_hits += 1
            return self._query_cache[query]
        
        # Cache miss - compute in executor
        self._cache_misses += 1
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.embedder.encode, query)
        
        # Add to cache
        self._query_cache[query] = embedding
        
        # Evict oldest if over limit
        if len(self._query_cache) > self.QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        
        return embedding
    
    def clear_query_cache(self) -> None:
        """Clear the query embedding cache."""
        self._query_cache.clear()
        self.logger.info("Query embedding cache cleared")

    def embed_with_fallback(
        self,
        documents: List[str]
    ) -> List[List[float]]:
        """
        Embed documents with fallback to individual encoding on batch failure.

        Strategy:
        1. Try batch encoding (fast)
        2. If batch fails, try one-by-one (slower but more resilient)
        3. For individual failures, use zero vector (allows partial success)

        Args:
            documents: List of document strings to embed

        Returns:
            List of embedding vectors (same length as documents)
        """
        try:
            # Try batch encoding first (optimal path)
            embeddings = self.embedder.encode_batch(documents)

            # Validate dimensions
            if embeddings and len(embeddings[0]) != self.embedder.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedder.dimension}, "
                    f"got {len(embeddings[0])}"
                )

            self.logger.debug(f"Successfully batch-encoded {len(documents)} documents")
            self._embedding_success_count += len(documents)
            return embeddings

        except Exception as batch_error:
            self.logger.warning(
                f"Batch embedding failed ({batch_error}), "
                f"falling back to individual encoding for {len(documents)} documents"
            )

            # Fallback: Encode one by one
            embeddings = []
            failed_count = 0

            for i, doc in enumerate(documents):
                try:
                    embedding = self.embedder.encode(doc)

                    # Validate dimension
                    if len(embedding) != self.embedder.dimension:
                        raise ValueError(f"Dimension mismatch: {len(embedding)} != {self.embedder.dimension}")

                    embeddings.append(embedding)
                    self._embedding_success_count += 1

                except Exception as doc_error:
                    failed_count += 1
                    self.logger.error(
                        f"Failed to embed document {i} (preview: {doc[:100]}...): {doc_error}"
                    )

                    # Use zero vector as placeholder
                    # This allows partial batch success
                    zero_vector = [0.0] * self.embedder.dimension
                    embeddings.append(zero_vector)
                    self._embedding_failure_count += 1

                    self.logger.info(f"Using zero vector for failed document {i}")

            if failed_count > 0:
                self.logger.warning(
                    f"Embedding completed with {failed_count}/{len(documents)} failures "
                    f"(using zero vectors for failed documents)"
                )
            else:
                self.logger.info(
                    f"Successfully encoded all {len(documents)} documents individually "
                    f"after batch failure"
                )

            return embeddings

    async def embed_in_batches(
        self,
        documents: List[str],
        batch_size: Optional[int] = None,
        delay: Optional[float] = None
    ) -> List[List[float]]:
        """
        Embed documents in batches to avoid memory/rate limit issues.

        Benefits:
        - Prevents memory exhaustion
        - Avoids API rate limits
        - Allows progress tracking
        - More resilient to partial failures

        Args:
            documents: List of document strings to embed
            batch_size: Documents per batch (default from config)
            delay: Seconds to wait between batches (default from config)

        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or self.config.EMBEDDING_BATCH_SIZE
        delay = delay or self.config.EMBEDDING_BATCH_DELAY

        total_docs = len(documents)
        all_embeddings = []

        self.logger.info(
            f"Embedding {total_docs} documents in batches of {batch_size}"
        )

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size

            self.logger.info(
                f"Embedding batch {batch_num}/{total_batches} "
                f"({len(batch)} documents)"
            )

            try:
                # Use fallback method for resilience
                batch_embeddings = self.embed_with_fallback(batch)
                all_embeddings.extend(batch_embeddings)

                # Report progress
                progress_pct = ((i + len(batch)) / total_docs) * 100
                self.logger.debug(f"Progress: {progress_pct:.1f}%")

                # Rate limiting: wait between batches (except last batch)
                if i + batch_size < total_docs and delay > 0:
                    await asyncio.sleep(delay)

            except Exception as e:
                self.logger.error(
                    f"Failed to embed batch {batch_num}/{total_batches}: {e}"
                )
                # Fail fast for now (could implement retry logic later)
                raise

        self.logger.info(
            f"Successfully embedded {len(all_embeddings)}/{total_docs} documents"
        )

        return all_embeddings

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding success/failure and cache statistics."""
        total = self._embedding_success_count + self._embedding_failure_count
        cache_total = self._cache_hits + self._cache_misses
        return {
            'total_embedded': total,
            'successful': self._embedding_success_count,
            'failed': self._embedding_failure_count,
            'success_rate': self._embedding_success_count / total if total > 0 else 0.0,
            # Cache statistics
            'cache_size': len(self._query_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / cache_total if cache_total > 0 else 0.0
        }

