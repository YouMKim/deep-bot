"""
BM25 search service for chunked memory.

Handles BM25 keyword-based search with caching.
"""
import logging
import re
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from rank_bm25 import BM25Okapi
from storage.vectors.base import VectorStorage
from chunking.constants import ChunkStrategy
from .author_filter import AuthorFilter
from .utils import get_collection_name, resolve_strategy

if TYPE_CHECKING:
    from config import Config


class BM25Service:
    """Service for BM25 keyword-based search with caching."""

    def __init__(
        self,
        vector_store: VectorStorage,
        author_filter: AuthorFilter,
        config: Optional['Config'] = None
    ):
        """
        Initialize BM25Service.

        Args:
            vector_store: Vector storage instance for document retrieval
            author_filter: AuthorFilter instance for filtering results
            config: Configuration instance (defaults to Config class)
        """
        from config import Config as ConfigClass
        
        self.vector_store = vector_store
        self.author_filter = author_filter
        self.config = config or ConfigClass
        self.logger = logging.getLogger(__name__)
        
        # BM25 cache: {collection_name: {'bm25': BM25Okapi, 'tokenized_corpus': List, 'documents': List, 'version': int}}
        self._bm25_cache: Dict[str, Dict[str, Any]] = {}

    def search(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
        active_strategy: str = "single",
        top_k: int = 10,
        exclude_blacklisted: bool = True,
        filter_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Keyword-based search using BM25 algorithm.

        Args:
            query: Search query string
            strategy: Optional strategy override
            active_strategy: Active strategy name (used if strategy is None)
            top_k: Number of results to return
            exclude_blacklisted: Filter blacklisted authors
            filter_authors: Filter to specific authors

        Returns:
            List of search results with BM25 scores
        """
        strategy_value = resolve_strategy(strategy, active_strategy)
        collection_name = get_collection_name(strategy_value)

        # Check if cache is valid
        current_count = self.vector_store.get_collection_count(collection_name)
        cache_entry = self._bm25_cache.get(collection_name)
        
        if cache_entry and cache_entry.get('version') == current_count and current_count > 0:
            # Cache is valid, use it
            self.logger.debug(f"Using cached BM25 index for {collection_name}")
            bm25 = cache_entry['bm25']
            all_docs = cache_entry['documents']
        else:
            # Cache miss or invalid, rebuild
            self.logger.info(f"Building BM25 index for {collection_name} (count: {current_count})")
            
            try:
                all_docs = self.vector_store.get_all_documents(collection_name)
            except Exception as e:
                self.logger.error(f"Failed to fetch documents for BM25: {e}")
                return []

            if not all_docs:
                self.logger.warning(f"No documents found in {collection_name}")
                return []

            # Tokenize documents
            tokenized_corpus = [self.tokenize(doc['document']) for doc in all_docs]

            # Build BM25 index
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Cache the index
            self._bm25_cache[collection_name] = {
                'bm25': bm25,
                'tokenized_corpus': tokenized_corpus,
                'documents': all_docs,
                'version': current_count
            }
            self.logger.info(f"Cached BM25 index for {collection_name}")

        # Tokenize query
        tokenized_query = self.tokenize(query)

        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)

        # Sort by score and create results
        scored_docs = list(zip(all_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Format results (same as before)
        results = []
        for doc_data, score in scored_docs:
            document = doc_data['document']
            metadata = doc_data['metadata']
            author = metadata.get('author', '')

            # Apply filters using helper method
            if not self.author_filter.should_include(author, exclude_blacklisted, filter_authors):
                continue

            # Standardized result format
            normalized_similarity = float(score) / 100.0  # Normalize for consistency
            results.append({
                "content": document,
                "metadata": metadata,
                "bm25_score": float(score),  # Keep raw score for reference
                "similarity": normalized_similarity,  # Normalized score
                "search_type": "bm25",  # Standardized field
            })

            if len(results) >= top_k:
                break

        self.logger.info(f"BM25 search returned {len(results)} results")
        return results

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]

    def invalidate_cache(self, collection_name: str) -> None:
        """
        Invalidate BM25 cache for a collection.

        Args:
            collection_name: Name of collection to invalidate
        """
        if collection_name in self._bm25_cache:
            del self._bm25_cache[collection_name]
            self.logger.debug(f"Invalidated BM25 cache for {collection_name}")

