"""
Ingestion service for chunked memory.

Handles channel ingestion pipeline, chunk storage, and validation.
"""
import logging
import asyncio
from typing import Dict, List, Optional, Sequence, Callable, Any, TYPE_CHECKING
from datetime import datetime
from chunking.base import Chunk
from chunking.constants import ChunkStrategy
from chunking.service import ChunkingService
from storage.vectors.base import VectorStorage
from storage.messages.messages import MessageStorage
from .embedding_service import EmbeddingService
from .bm25_service import BM25Service

if TYPE_CHECKING:
    from config import Config


class IngestionService:
    """Service for ingesting and storing chunks."""

    def __init__(
        self,
        vector_store: VectorStorage,
        embedding_service: EmbeddingService,
        message_storage: MessageStorage,
        chunking_service: ChunkingService,
        bm25_service: BM25Service,
        config: Optional['Config'] = None
    ):
        """
        Initialize IngestionService.

        Args:
            vector_store: Vector storage instance
            embedding_service: EmbeddingService instance
            message_storage: MessageStorage instance
            chunking_service: ChunkingService instance
            bm25_service: BM25Service instance (for cache invalidation)
            config: Configuration instance (defaults to Config class)
        """
        from config import Config as ConfigClass
        
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.message_storage = message_storage
        self.chunking_service = chunking_service
        self.bm25_service = bm25_service
        self.config = config or ConfigClass
        self.logger = logging.getLogger(__name__)
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    async def store_all_strategies(self, chunks_by_strategy: Dict[str, Sequence[Chunk]]) -> None:
        """
        Persist chunks for every strategy into the vector store.

        Now async to support batched embedding with delays.

        Args:
            chunks_by_strategy: mapping of strategy name to a sequence of chunks.
        """
        if not chunks_by_strategy:
            self.logger.warning("No strategies provided for storage")
            return

        for strategy_name, chunks in chunks_by_strategy.items():
            if not chunks:
                self.logger.info("No chunks found for strategy '%s'; skipping", strategy_name)
                continue

            collection_name = f"discord_chunks_{strategy_name}"
            self.vector_store.create_collection(collection_name)

            # Invalidate BM25 cache for this collection
            self.bm25_service.invalidate_cache(collection_name)

            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [
                f"{strategy_name}_{idx}_{chunk.metadata.get('first_message_id', idx)}"
                for idx, chunk in enumerate(chunks)
            ]

            self.logger.info(
                "Generating embeddings for %s %s chunk(s)",
                len(chunks),
                strategy_name,
            )
            try:
                # Use batched embedding to avoid memory/rate limit issues
                embeddings = await self.embedding_service.embed_in_batches(documents)

                self.vector_store.add_documents(
                    collection_name=collection_name,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                self.logger.info(
                    "Stored %s chunk(s) for strategy '%s'", len(chunks), strategy_name
                )
            except Exception as exc:
                self.logger.error("Error storing '%s' chunks: %s", strategy_name, exc)
                raise

    async def ingest_channel(
        self,
        channel_id: str,
        batch_size: int = 1000,
        strategies: Optional[List[ChunkStrategy]] = None
    ) -> Dict[str, Any]:
        """
        Process all unprocessed messages from a channel through all strategies.
        
        This implements the Stage 2 pipeline:
        1. For each strategy:
           a. Get chunking checkpoint (last processed message ID)
           b. Load batch of messages after checkpoint
           c. Generate chunks
           d. Embed and store chunks
           e. Update checkpoint
           f. Repeat until no more messages
        
        Args:
            channel_id: Channel ID to process
            batch_size: Number of messages to process per batch (default 1000)
            strategies: List of strategies to use (default: from config, typically single+tokens)
            
        Returns:
            Statistics dictionary with per-strategy results
        """
        start_time = datetime.now()
        
        # Use configured default strategies if none specified
        if strategies is None:
            default_strategies = self.config.CHUNKING_DEFAULT_STRATEGIES.split(',')
            strategies = [ChunkStrategy(s.strip()) for s in default_strategies]
            self.logger.info(f"Using default strategies: {[s.value for s in strategies]}")
        
        self.logger.info(
            f"Starting channel ingestion for {channel_id} with {len(strategies)} strategies"
        )
        
        overall_stats = {
            'channel_id': channel_id,
            'start_time': start_time.isoformat(),
            'strategies_processed': 0,
            'total_messages_processed': 0,
            'total_chunks_created': 0,
            'total_errors': 0,
            'strategy_details': {}
        }
        
        # Process each strategy independently
        for strategy in strategies:
            strategy_name = strategy.value
            self.logger.info(f"Processing strategy: {strategy_name}")
            
            strategy_stats = {
                'messages_processed': 0,
                'chunks_created': 0,
                'batches_processed': 0,
                'errors': 0,
                'last_message_id': None,
                'last_timestamp': None
            }
            
            try:
                # Get checkpoint for this strategy
                checkpoint = self.message_storage.get_chunking_checkpoint(
                    channel_id, strategy_name
                )
                
                # Determine starting point
                if checkpoint and checkpoint.get('last_message_id'):
                    self.logger.info(
                        f"Resuming {strategy_name} from checkpoint: "
                        f"{checkpoint['last_message_id']}"
                    )
                    last_processed_id = checkpoint['last_message_id']
                else:
                    self.logger.info(f"Starting {strategy_name} from beginning")
                    last_processed_id = None
                
                # Process messages in batches until no more remain
                while True:
                    # Load next batch of messages
                    if last_processed_id:
                        # Continue from last processed message
                        messages = self.message_storage.get_messages_after(
                            channel_id, last_processed_id, batch_size
                        )
                    else:
                        # Start from oldest messages
                        messages = self.message_storage.get_oldest_messages(
                            channel_id, batch_size
                        )
                    
                    if not messages:
                        self.logger.info(
                            f"No more messages to process for {strategy_name}"
                        )
                        break
                    
                    self.logger.info(
                        f"Processing batch of {len(messages)} messages for {strategy_name}"
                    )
                    
                    # Report progress
                    await self.report_progress({
                        'channel_id': channel_id,
                        'strategy': strategy_name,
                        'batch_messages': len(messages),
                        'total_processed': strategy_stats['messages_processed'],
                        'chunks_created': strategy_stats['chunks_created']
                    })
                    
                    try:
                        # Generate chunks for this batch using the specific strategy
                        chunks_by_strategy = self.chunking_service.chunk_messages(
                            messages, strategies=[strategy_name]
                        )
                        
                        chunks = chunks_by_strategy.get(strategy_name, [])
                        
                        if not chunks:
                            self.logger.warning(
                                f"No chunks generated for {strategy_name} "
                                f"from {len(messages)} messages"
                            )
                            # Still update checkpoint to avoid reprocessing
                            last_message = messages[-1]
                            last_processed_id = last_message['message_id']
                            strategy_stats['messages_processed'] += len(messages)
                            continue
                        
                        # Validate chunks before storing
                        valid_chunks = []
                        for chunk in chunks:
                            if self.validate_chunk(chunk):
                                valid_chunks.append(chunk)
                            else:
                                self.logger.warning(
                                    f"Skipping invalid chunk for {strategy_name}"
                                )
                        
                        if not valid_chunks:
                            self.logger.warning(
                                f"No valid chunks after validation for {strategy_name}"
                            )
                            last_message = messages[-1]
                            last_processed_id = last_message['message_id']
                            strategy_stats['messages_processed'] += len(messages)
                            continue
                        
                        # Store chunks in vector DB
                        await self.store_all_strategies({strategy_name: valid_chunks})
                        
                        # Update statistics
                        strategy_stats['messages_processed'] += len(messages)
                        strategy_stats['chunks_created'] += len(valid_chunks)
                        strategy_stats['batches_processed'] += 1
                        
                        # Get last message for checkpoint
                        last_message = messages[-1]
                        last_processed_id = last_message['message_id']
                        last_timestamp = last_message['timestamp']
                        
                        strategy_stats['last_message_id'] = last_processed_id
                        strategy_stats['last_timestamp'] = last_timestamp
                        
                        # Update checkpoint
                        last_chunk = valid_chunks[-1]
                        last_chunk_id = last_chunk.metadata.get('first_message_id', last_processed_id)
                        
                        self.message_storage.update_chunking_checkpoint(
                            channel_id=channel_id,
                            strategy=strategy_name,
                            last_chunk_id=str(last_chunk_id),
                            last_message_id=str(last_processed_id),
                            last_timestamp=last_timestamp
                        )
                        
                        self.logger.info(
                            f"Batch complete for {strategy_name}: "
                            f"{len(valid_chunks)} chunks created"
                        )
                        
                        if len(messages) < batch_size:
                            self.logger.info(
                                f"Reached end of messages for {strategy_name} "
                                f"(got {len(messages)} < {batch_size})"
                            )
                            break
                        
                    except Exception as e:
                        self.logger.error(
                            f"Error processing batch for {strategy_name}: {e}",
                            exc_info=True
                        )
                        strategy_stats['errors'] += 1
                        if messages:
                            last_message = messages[-1]
                            last_processed_id = last_message['message_id']
                            strategy_stats['messages_processed'] += len(messages)
                
                overall_stats['strategies_processed'] += 1
                overall_stats['strategy_details'][strategy_name] = strategy_stats
                
                self.logger.info(
                    f"Strategy {strategy_name} complete: "
                    f"{strategy_stats['messages_processed']} messages, "
                    f"{strategy_stats['chunks_created']} chunks, "
                    f"{strategy_stats['errors']} errors"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Fatal error processing strategy {strategy_name}: {e}",
                    exc_info=True
                )
                strategy_stats['errors'] += 1
                overall_stats['strategy_details'][strategy_name] = strategy_stats
                overall_stats['total_errors'] += 1
                continue
        
        for strategy_stats in overall_stats['strategy_details'].values():
            overall_stats['total_messages_processed'] += strategy_stats['messages_processed']
            overall_stats['total_chunks_created'] += strategy_stats['chunks_created']
            overall_stats['total_errors'] += strategy_stats['errors']
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        overall_stats['end_time'] = end_time.isoformat()
        overall_stats['duration_seconds'] = duration
        
        self.logger.info(
            f"Channel ingestion complete for {channel_id}: "
            f"{overall_stats['strategies_processed']} strategies, "
            f"{overall_stats['total_messages_processed']} messages, "
            f"{overall_stats['total_chunks_created']} chunks, "
            f"{overall_stats['total_errors']} errors, "
            f"{duration:.1f} seconds"
        )
        
        return overall_stats

    def validate_chunk(self, chunk: Chunk) -> bool:
        """
        Validate a chunk before storing.

        Args:
            chunk: Chunk to validate

        Returns:
            True if valid, False otherwise
        """
        if not chunk.content or not chunk.content.strip():
            self.logger.warning("Chunk has empty content")
            return False
        
        required_fields = ['chunk_strategy', 'channel_id', 'first_message_id']
        for field in required_fields:
            if field not in chunk.metadata:
                self.logger.warning(f"Chunk missing required metadata field: {field}")
                return False
        
        if not chunk.message_ids:
            self.logger.warning("Chunk has no message IDs")
            return False
        
        return True

    async def report_progress(self, progress: Dict[str, Any]) -> None:
        """
        Report progress via callback if set.

        Args:
            progress: Progress dictionary
        """
        if not self.progress_callback:
            return
        
        try:
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(progress)
            else:
                self.progress_callback(progress)
        except Exception as e:
            self.logger.error(f"Error in progress callback: {e}")

    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Set callback for reporting progress during ingestion.

        Args:
            callback: Progress callback function
        """
        self.progress_callback = callback

