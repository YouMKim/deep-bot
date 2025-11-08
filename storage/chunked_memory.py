from storage.vectors.factory import VectorStoreFactory
from storage.vectors.base import VectorStorage
from embedding.factory import EmbeddingFactory
from embedding.base import EmbeddingBase
from chunking.base import Chunk
from chunking.constants import ChunkStrategy
from typing import Dict, List, Optional, Sequence, Callable, Any
from storage.messages.messages import MessageStorage
from chunking.service import ChunkingService
import logging
import asyncio
from datetime import datetime

class ChunkedMemoryService:

    def __init__(
        self,
        vector_store: Optional[VectorStorage] = None,
        embedder: Optional[EmbeddingBase] = None,
        message_storage: Optional[MessageStorage] = None,
        chunking_service: Optional[ChunkingService] = None,
        default_strategy: ChunkStrategy = ChunkStrategy.TEMPORAL,
    ):
        self.vector_store = vector_store or VectorStoreFactory.create()
        self.embedder = embedder or EmbeddingFactory.create_embedder()
        self.message_storage = message_storage or MessageStorage()
        self.chunking_service = chunking_service or ChunkingService()
        self.active_strategy = default_strategy.value
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        self.logger = logging.getLogger(__name__)

    def set_active_strategy(self, strategy: ChunkStrategy) -> None:
        if strategy.value not in ChunkStrategy.values():
            raise ValueError(f"Unsupported strategy: {strategy}")
        self.active_strategy = strategy.value

    def store_all_strategies(self, chunks_by_strategy: Dict[str, Sequence[Chunk]]) -> None:
        """
        Persist chunks for every strategy into the vector store.

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
                embeddings = self.embedder.encode_batch(documents)
                if embeddings and len(embeddings[0]) != self.embedder.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.embedder.dimension}, "
                        f"got {len(embeddings[0])}"
                    )

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

    def search(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search for relevant chunks using the specified (or active) strategy.

        Args:
            query: text to embed and search for
            strategy: optional strategy override; defaults to active strategy
            top_k: number of results to return
        """
        strategy_value = (strategy or ChunkStrategy(self.active_strategy)).value
        collection_name = f"discord_chunks_{strategy_value}"

        try:
            query_embedding = self.embedder.encode(query)
        except Exception as exc:
            self.logger.error("Failed to generate query embedding: %s", exc)
            raise

        try:
            results = self.vector_store.query(
                collection_name=collection_name,
                query_embeddings=[query_embedding],
                n_results=top_k,
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

        if documents and documents[0]:
            metadata_list = metadatas[0] if metadatas else []
            distance_list = distances[0] if distances else []

            for index, document in enumerate(documents[0]):
                similarity = (
                    1 - distance_list[index]
                    if index < len(distance_list)
                    else None
                )
                formatted_results.append(
                    {
                        "content": document,
                        "metadata": metadata_list[index] if index < len(metadata_list) else {},
                        "similarity": similarity,
                    }
                )

        return formatted_results

    def get_strategy_stats(self) -> Dict[str, int]:
        """Return the number of stored chunks for each strategy."""
        stats: Dict[str, int] = {}
        for strategy in ChunkStrategy:
            collection_name = f"discord_chunks_{strategy.value}"
            try:
                stats[strategy.value] = self.vector_store.get_collection_count(collection_name)
            except Exception:
                stats[strategy.value] = 0
        return stats

    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback for reporting progress during ingestion."""
        self.progress_callback = callback

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
            strategies: List of strategies to use (default: all strategies)
            
        Returns:
            Statistics dictionary with per-strategy results
        """
        start_time = datetime.now()
        
        # Use all strategies if none specified
        if strategies is None:
            strategies = list(ChunkStrategy)
        
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
                        messages = self.message_storage.get_messages_after(
                            channel_id, last_processed_id, batch_size
                        )
                    else:
                        messages = self.message_storage.get_recent_messages(
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
                    await self._report_progress({
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
                            if self._validate_chunk(chunk):
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
                        self.store_all_strategies({strategy_name: valid_chunks})
                        
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

    def _validate_chunk(self, chunk: Chunk) -> bool:
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

    async def _report_progress(self, progress: Dict[str, Any]) -> None:
        """Report progress via callback if set."""
        if not self.progress_callback:
            return
        
        try:
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(progress)
            else:
                self.progress_callback(progress)
        except Exception as e:
            self.logger.error(f"Error in progress callback: {e}")