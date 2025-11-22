"""
Chunked Memory Service - Facade for chunked memory operations.

This module provides a unified interface for:
- Embedding documents
- Storing chunks
- Searching chunks (vector, BM25, hybrid)
- Ingesting channels
- Author filtering

The service is split into focused sub-services for better maintainability.
"""
import logging
from typing import Dict, List, Optional, Sequence, Callable, Any, TYPE_CHECKING
from chunking.base import Chunk
from chunking.constants import ChunkStrategy
from embedding.base import EmbeddingBase
from storage.vectors.base import VectorStorage
from storage.messages.messages import MessageStorage
from chunking.service import ChunkingService

from .author_filter import AuthorFilter
from .embedding_service import EmbeddingService
from .bm25_service import BM25Service
from .retrieval_service import RetrievalService
from .ingestion_service import IngestionService
from .utils import get_collection_name, resolve_strategy, calculate_fetch_k

if TYPE_CHECKING:
    from config import Config


class ChunkedMemoryService:
    """
    Facade for chunked memory operations.
    
    Maintains backward-compatible API while delegating to focused sub-services.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStorage] = None,
        embedder: Optional[EmbeddingBase] = None,
        message_storage: Optional[MessageStorage] = None,
        chunking_service: Optional[ChunkingService] = None,
        config: Optional['Config'] = None,
        default_strategy: ChunkStrategy = ChunkStrategy.SINGLE,
    ):
        """
        Initialize ChunkedMemoryService.

        Args:
            vector_store: Vector storage instance (defaults to factory)
            embedder: Embedder instance (defaults to factory)
            message_storage: Message storage instance (defaults to MessageStorage())
            chunking_service: Chunking service instance (defaults to ChunkingService())
            config: Configuration instance (defaults to Config class)
            default_strategy: Default chunking strategy
        """
        from storage.vectors.factory import VectorStoreFactory
        from embedding.factory import EmbeddingFactory
        from config import Config as ConfigClass
        
        # Set config first so it's available for use below
        self.config = config or ConfigClass
        self.logger = logging.getLogger(__name__)
        
        self.vector_store = vector_store or VectorStoreFactory.create()
        # Use config to determine embedding provider and model
        if embedder is None:
            model_name = self.config.EMBEDDING_MODEL if self.config.EMBEDDING_MODEL else ""
            try:
                embedder = EmbeddingFactory.create_embedder(
                    provider=self.config.EMBEDDING_PROVIDER,
                    model_name=model_name
                )
            except ImportError as e:
                if "tokenizers" in str(e).lower() or "sentence-transformers" in str(e).lower():
                    self.logger.error(
                        f"Failed to initialize embedder: {e}. "
                        "Please install tokenizers with: pip install tokenizers "
                        "Or set EMBEDDING_PROVIDER=openai to use OpenAI embeddings instead."
                    )
                    raise RuntimeError(
                        f"Embedding provider '{self.config.EMBEDDING_PROVIDER}' requires tokenizers. "
                        "Install it with: pip install tokenizers "
                        "Or set EMBEDDING_PROVIDER=openai in your environment variables."
                    ) from e
                raise
        self.embedder = embedder
        self.message_storage = message_storage or MessageStorage()
        self.chunking_service = chunking_service or ChunkingService()
        self.active_strategy = default_strategy.value
        
        # Workaround for ChromaDB KeyError: frozenset() issue
        # Try to list collections early to catch any compatibility issues
        try:
            _ = self.vector_store.list_collections()
        except (KeyError, AttributeError, TypeError) as e:
            error_str = str(e).lower()
            # Check for various frozenset-related errors:
            # 1. KeyError with "frozenset" in message
            # 2. KeyError with empty message (frozenset() representation)
            # 3. TypeError from frozenset metadata issues
            is_frozenset_error = (
                "frozenset" in error_str or
                (isinstance(e, KeyError) and (not str(e) or str(e) == "frozenset()")) or
                (isinstance(e, TypeError) and "frozenset" in error_str)
            )
            if is_frozenset_error:
                self.logger.warning(
                    f"ChromaDB compatibility issue detected ({type(e).__name__}: {e}). "
                    "This may be due to corrupted metadata. Collections will be recreated as needed."
                )
            else:
                raise

        # Initialize sub-services
        self.author_filter = AuthorFilter(config=self.config)
        self.embedding_service = EmbeddingService(
            embedder=self.embedder,
            config=self.config
        )
        self.bm25_service = BM25Service(
            vector_store=self.vector_store,
            author_filter=self.author_filter,
            config=self.config
        )
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            embedder=self.embedder,
            author_filter=self.author_filter,
            bm25_service=self.bm25_service,
            config=self.config
        )
        self.ingestion_service = IngestionService(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            message_storage=self.message_storage,
            chunking_service=self.chunking_service,
            bm25_service=self.bm25_service,
            config=self.config
        )

    def set_active_strategy(self, strategy: ChunkStrategy) -> None:
        """Set the active chunking strategy."""
        if strategy.value not in ChunkStrategy.values():
            raise ValueError(f"Unsupported strategy: {strategy}")
        self.active_strategy = strategy.value

    # Delegate embedding methods
    def _embed_with_fallback(self, documents: List[str]) -> List[List[float]]:
        """Embed documents with fallback (backward compatibility)."""
        return self.embedding_service.embed_with_fallback(documents)

    async def _embed_in_batches(
        self,
        documents: List[str],
        batch_size: Optional[int] = None,
        delay: Optional[float] = None
    ) -> List[List[float]]:
        """Embed documents in batches (backward compatibility)."""
        return await self.embedding_service.embed_in_batches(documents, batch_size, delay)

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics (backward compatibility)."""
        return self.embedding_service.get_stats()

    # Delegate storage methods
    async def store_all_strategies(self, chunks_by_strategy: Dict[str, Sequence[Chunk]]) -> None:
        """Store chunks for all strategies (backward compatibility)."""
        await self.ingestion_service.store_all_strategies(chunks_by_strategy)

    # Delegate author filtering (backward compatibility)
    def _should_include_author(
        self,
        author: str,
        exclude_blacklisted: bool,
        filter_authors: Optional[List[str]],
        author_id: Optional[str] = None
    ) -> bool:
        """Check if author should be included (backward compatibility)."""
        return self.author_filter.should_include(author, exclude_blacklisted, filter_authors, author_id=author_id)

    # Delegate search methods
    async def search(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
        top_k: int = 10,
        exclude_blacklisted: bool = True,
        filter_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Search for relevant chunks using vector similarity (backward compatibility)."""
        return await self.retrieval_service.search(
            query=query,
            strategy=strategy,
            active_strategy=self.active_strategy,
            top_k=top_k,
            exclude_blacklisted=exclude_blacklisted,
            filter_authors=filter_authors
        )

    def search_bm25(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
        top_k: int = 10,
        exclude_blacklisted: bool = True,
        filter_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Keyword-based search using BM25 (backward compatibility)."""
        return self.bm25_service.search(
            query=query,
            strategy=strategy,
            active_strategy=self.active_strategy,
            top_k=top_k,
            exclude_blacklisted=exclude_blacklisted,
            filter_authors=filter_authors
        )

    async def search_hybrid(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
        top_k: int = 10,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        exclude_blacklisted: bool = True,
        filter_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Hybrid search combining BM25 and vector similarity (backward compatibility)."""
        return await self.retrieval_service.search_hybrid(
            query=query,
            strategy=strategy,
            active_strategy=self.active_strategy,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            exclude_blacklisted=exclude_blacklisted,
            filter_authors=filter_authors
        )

    # Delegate stats and config
    def get_strategy_stats(self) -> Dict[str, int]:
        """Return the number of stored chunks for each strategy (backward compatibility)."""
        stats: Dict[str, int] = {}
        for strategy in ChunkStrategy:
            collection_name = get_collection_name(strategy.value)
            try:
                stats[strategy.value] = self.vector_store.get_collection_count(collection_name)
            except Exception:
                stats[strategy.value] = 0
        return stats

    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback for reporting progress during ingestion (backward compatibility)."""
        self.ingestion_service.set_progress_callback(callback)

    # Delegate ingestion methods
    async def ingest_channel(
        self,
        channel_id: str,
        batch_size: int = 1000,
        strategies: Optional[List[ChunkStrategy]] = None
    ) -> Dict[str, Any]:
        """Process all unprocessed messages from a channel (backward compatibility)."""
        return await self.ingestion_service.ingest_channel(
            channel_id=channel_id,
            batch_size=batch_size,
            strategies=strategies
        )

    def _validate_chunk(self, chunk: Chunk) -> bool:
        """Validate a chunk before storing (backward compatibility)."""
        return self.ingestion_service.validate_chunk(chunk)

    async def _report_progress(self, progress: Dict[str, Any]) -> None:
        """Report progress via callback (backward compatibility)."""
        await self.ingestion_service.report_progress(progress)

