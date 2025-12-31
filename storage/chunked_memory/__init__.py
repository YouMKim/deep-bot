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
from .bot_knowledge_service import BotKnowledgeService
from .utils import get_collection_name, resolve_strategy, calculate_fetch_k

# #region agent log
import json as _dbg_json
import traceback as _dbg_traceback
def _dbg_log(location, message, data=None, hypothesis_id=None):
    try:
        log_entry = {"location": location, "message": message, "data": data or {}, "timestamp": __import__('time').time(), "hypothesisId": hypothesis_id, "sessionId": "debug-session"}
        with open("/Users/youmyeongkim/projects/deep-bot/.cursor/debug.log", "a") as f:
            f.write(_dbg_json.dumps(log_entry) + "\n")
    except: pass
# #endregion

if TYPE_CHECKING:
    from config import Config


class ChunkedMemoryService:
    # #region agent log
    _instance_count = 0  # Track how many instances are created
    # #endregion
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
        # #region agent log
        import psutil
        ChunkedMemoryService._instance_count += 1
        instance_num = ChunkedMemoryService._instance_count
        process = psutil.Process()
        # Get caller info
        stack = _dbg_traceback.extract_stack()
        caller = stack[-3] if len(stack) >= 3 else None  # -1 is this, -2 is __init__, -3 is caller
        caller_info = f"{caller.filename}:{caller.lineno}" if caller else "unknown"
        _dbg_log("chunked_memory/__init__.py:init:start", f"Creating ChunkedMemoryService instance #{instance_num}", {"rss_mb_before": process.memory_info().rss / 1024 / 1024, "instance_num": instance_num, "caller": caller_info, "embedder_provided": embedder is not None}, "A")
        # #endregion
        
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
            from storage.utils import is_chromadb_compatibility_error, log_chromadb_warning
            if is_chromadb_compatibility_error(e):
                log_chromadb_warning(e, "ChunkedMemoryService initialization")
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
            config=self.config,
            embedding_service=self.embedding_service
        )
        self.ingestion_service = IngestionService(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            message_storage=self.message_storage,
            chunking_service=self.chunking_service,
            bm25_service=self.bm25_service,
            config=self.config
        )
        self.bot_knowledge_service = BotKnowledgeService(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            config=self.config
        )
        
        # Initialize bot knowledge automatically on startup
        # Checks if already indexed, and if not, indexes it in the background
        self._bot_knowledge_init_pending = True
        self._bot_knowledge_init_attempted = False
        
        # Try to initialize immediately if event loop is available
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Event loop is running, initialize in background (non-blocking)
                # This checks if already indexed and only loads if needed
                asyncio.create_task(self._initialize_bot_knowledge())
                self._bot_knowledge_init_pending = False
                self._bot_knowledge_init_attempted = True
        except (RuntimeError, AttributeError):
            # No event loop yet, will initialize lazily on first search
            # This ensures bot knowledge is available even if startup initialization fails
            pass
        
        # #region agent log
        _dbg_log("chunked_memory/__init__.py:init:complete", f"ChunkedMemoryService instance #{instance_num} initialized", {"rss_mb_after": process.memory_info().rss / 1024 / 1024, "instance_num": instance_num, "total_instances": ChunkedMemoryService._instance_count}, "A")
        # #endregion

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
        # Ensure bot knowledge is initialized (lazy initialization)
        self._ensure_bot_knowledge_initialized()
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
        # Ensure bot knowledge is initialized (lazy initialization)
        self._ensure_bot_knowledge_initialized()
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
    
    async def _initialize_bot_knowledge(self) -> None:
        """Initialize bot knowledge in background."""
        try:
            self.logger.info("Initializing bot knowledge...")
            success = await self.bot_knowledge_service.ingest_bot_knowledge(force=False)
            if success:
                self.logger.info("Bot knowledge initialized successfully")
            else:
                self.logger.info("Bot knowledge already indexed or initialization skipped")
        except Exception as e:
            self.logger.warning(f"Failed to initialize bot knowledge: {e}", exc_info=True)
    
    def _ensure_bot_knowledge_initialized(self):
        """Ensure bot knowledge is initialized (lazy initialization on first search)."""
        if (hasattr(self, '_bot_knowledge_init_pending') and 
            self._bot_knowledge_init_pending and 
            not getattr(self, '_bot_knowledge_init_attempted', False)):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Initialize in background
                    asyncio.create_task(self._initialize_bot_knowledge())
                    self._bot_knowledge_init_pending = False
                    self._bot_knowledge_init_attempted = True
            except (RuntimeError, AttributeError):
                # Event loop not available, will try again next time
                pass
    
    async def reindex_bot_knowledge(self, force: bool = True) -> bool:
        """
        Re-index bot knowledge documentation.
        
        Args:
            force: If True, re-index even if already indexed
            
        Returns:
            True if successful, False otherwise
        """
        return await self.bot_knowledge_service.ingest_bot_knowledge(force=force)

