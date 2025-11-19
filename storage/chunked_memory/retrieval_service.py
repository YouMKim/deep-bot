"""
Retrieval service for chunked memory.

Handles vector search and hybrid search orchestration.
"""
import logging
from typing import Dict, List, Optional, TYPE_CHECKING
from storage.vectors.base import VectorStorage
from embedding.base import EmbeddingBase
from chunking.constants import ChunkStrategy
from .author_filter import AuthorFilter
from .bm25_service import BM25Service
from .utils import get_collection_name, resolve_strategy, calculate_fetch_k

if TYPE_CHECKING:
    from config import Config


class RetrievalService:
    """Service for vector and hybrid search operations."""

    def __init__(
        self,
        vector_store: VectorStorage,
        embedder: EmbeddingBase,
        author_filter: AuthorFilter,
        bm25_service: BM25Service,
        config: Optional['Config'] = None
    ):
        """
        Initialize RetrievalService.

        Args:
            vector_store: Vector storage instance
            embedder: Embedder instance for query encoding
            author_filter: AuthorFilter instance for filtering
            bm25_service: BM25Service instance for hybrid search
            config: Configuration instance (defaults to Config class)
        """
        from config import Config as ConfigClass
        
        self.vector_store = vector_store
        self.embedder = embedder
        self.author_filter = author_filter
        self.bm25_service = bm25_service
        self.config = config or ConfigClass
        self.logger = logging.getLogger(__name__)

    async def search(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
        active_strategy: str = "single",
        top_k: int = 10,
        exclude_blacklisted: bool = True,
        filter_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search for relevant chunks using vector similarity.

        Args:
            query: Text to embed and search for
            strategy: Optional strategy override; defaults to active strategy
            active_strategy: Active strategy name (used if strategy is None)
            top_k: Number of results to return
            exclude_blacklisted: If True, filter out chunks from blacklisted authors
            filter_authors: If provided, only return chunks from these authors

        Returns:
            List of search results with similarity scores
        """
        strategy_value = resolve_strategy(strategy, active_strategy)
        collection_name = get_collection_name(strategy_value)

        try:
            # Run embedding in executor to avoid blocking event loop
            import asyncio
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(None, self.embedder.encode, query)
        except Exception as exc:
            self.logger.error("Failed to generate query embedding: %s", exc)
            raise

        # Fetch more results to account for potential filtering
        needs_filtering = exclude_blacklisted or filter_authors
        fetch_k = calculate_fetch_k(top_k, needs_filtering=needs_filtering)

        try:
            results = self.vector_store.query(
                collection_name=collection_name,
                query_embeddings=[query_embedding],
                n_results=fetch_k,
            )
        except Exception as exc:
            self.logger.error(
                "Vector store query failed for strategy '%s': %s",
                strategy_value,
                exc,
            )
            raise

        formatted_results: List[Dict] = []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        distances = results.get("distances") or []

        self.logger.info(f"Vector search returned {len(documents[0]) if documents and documents[0] else 0} results")

        if documents and documents[0]:
            metadata_list = metadatas[0] if metadatas else []
            distance_list = distances[0] if distances else []

            for index, document in enumerate(documents[0]):
                metadata = metadata_list[index] if index < len(metadata_list) else {}
                
                author = (
                    metadata.get('author') or 
                    metadata.get('primary_author_name') or 
                    metadata.get('primary_author_id') or 
                    ''
                )
                
                # Get author_id separately for blacklist checking
                author_id = metadata.get('primary_author_id') or metadata.get('author_id') or None
                
                similarity = (
                    1 - distance_list[index]
                    if index < len(distance_list)
                    else None
                )
                
                # Log each chunk with similarity score
                content_preview = document[:100] + "..." if len(document) > 100 else document
                self.logger.info(
                    f"Chunk {index + 1}: similarity={similarity:.3f}, "
                    f"author={author}, author_id={author_id}, content='{content_preview}'"
                )
                
                # Use helper method for author filtering (pass author_id for blacklist check)
                if not self.author_filter.should_include(author, exclude_blacklisted, filter_authors, author_id=author_id):
                    self.logger.info(f"  [FILTERED] Author: {author} (ID: {author_id})")
                    continue
                
                formatted_results.append(
                    {
                        "content": document,
                        "metadata": metadata,
                        "similarity": similarity,
                        "search_type": "vector",  # Standardized field
                    }
                )
                
                self.logger.info(f"  [INCLUDED] (total so far: {len(formatted_results)}/{top_k})")
                
                # Stop once we have enough results
                if len(formatted_results) >= top_k:
                    break

        self.logger.info(f"Final results: {len(formatted_results)} chunks returned (top_k={top_k})")
        return formatted_results

    async def search_hybrid(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
        active_strategy: str = "single",
        top_k: int = 10,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        exclude_blacklisted: bool = True,
        filter_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and vector similarity.

        Args:
            query: Search query
            strategy: Optional strategy override
            active_strategy: Active strategy name (used if strategy is None)
            top_k: Number of results
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            exclude_blacklisted: Filter blacklisted authors
            filter_authors: Filter to specific authors

        Returns:
            Fused results from both search methods
        """
        from rag.hybrid_search import HybridSearchService

        # Retrieve more candidates from each method (for fusion deduplication)
        fetch_k = calculate_fetch_k(top_k, needs_filtering=False, needs_reranking=False)

        # BM25 search
        bm25_results = self.bm25_service.search(
            query=query,
            strategy=strategy,
            active_strategy=active_strategy,
            top_k=fetch_k,
            exclude_blacklisted=exclude_blacklisted,
            filter_authors=filter_authors
        )

        # Vector search
        vector_results = await self.search(
            query=query,
            strategy=strategy,
            active_strategy=active_strategy,
            top_k=fetch_k,
            exclude_blacklisted=exclude_blacklisted,
            filter_authors=filter_authors
        )

        # Fuse results
        hybrid_service = HybridSearchService()
        fused_results = hybrid_service.hybrid_search(
            query=query,
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            top_k=top_k
        )

        return fused_results

