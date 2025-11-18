from storage.vectors.factory import VectorStoreFactory
from storage.vectors.base import VectorStorage
from embedding.factory import EmbeddingFactory
from embedding.base import EmbeddingBase
from chunking.base import Chunk
from chunking.constants import ChunkStrategy
from typing import Dict, List, Optional, Sequence, Callable, Any
from storage.messages.messages import MessageStorage
from chunking.service import ChunkingService
from rank_bm25 import BM25Okapi
import re 
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
        default_strategy: ChunkStrategy = ChunkStrategy.SINGLE,
    ):
        self.vector_store = vector_store or VectorStoreFactory.create()
        self.embedder = embedder or EmbeddingFactory.create_embedder()
        self.message_storage = message_storage or MessageStorage()
        self.chunking_service = chunking_service or ChunkingService()
        self.active_strategy = default_strategy.value
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ChunkedMemoryService initialized with active strategy: {self.active_strategy}")

        # BM25 cache: {collection_name: {'bm25': BM25Okapi, 'tokenized_corpus': List, 'documents': List, 'version': int}}
        self._bm25_cache: Dict[str, Dict[str, Any]] = {}
    
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

            if collection_name in self._bm25_cache:
                self.logger.debug(f"Invalidating BM25 cache for {collection_name}")
                del self._bm25_cache[collection_name]

            documents = [chunk.content for chunk in chunks]

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
        top_k: int = 10,
        exclude_blacklisted: bool = True,
        filter_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search for relevant chunks using the specified (or active) strategy.

        Args:
            query: text to embed and search for
            strategy: optional strategy override; defaults to active strategy
            top_k: number of results to return
            exclude_blacklisted: if True, filter out chunks from blacklisted authors
            filter_authors: if provided, only return chunks from these authors
        """
        strategy_value = (strategy or ChunkStrategy(self.active_strategy)).value
        collection_name = f"discord_chunks_{strategy_value}"

        try:
            query_embedding = self.embedder.encode(query)
        except Exception as exc:
            self.logger.error("Failed to generate query embedding: %s", exc)
            raise

        # Fetch more results to account for potential filtering
        # If any filtering is enabled, request extra results
        needs_filtering = exclude_blacklisted or filter_authors
        fetch_k = top_k * 3 if needs_filtering else top_k

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
                author = metadata.get('author', '')
                
                similarity = (
                    1 - distance_list[index]
                    if index < len(distance_list)
                    else None
                )
                
                # Log each chunk with similarity score
                content_preview = document[:100] + "..." if len(document) > 100 else document
                self.logger.info(
                    f"Chunk {index + 1}: similarity={similarity:.3f}, "
                    f"author={author}, content='{content_preview}'"
                )
                
                # Check if author is blacklisted
                if exclude_blacklisted:
                    from config import Config
                    if author in Config.BLACKLIST_IDS or str(author) in [str(bid) for bid in Config.BLACKLIST_IDS]:
                        self.logger.info(f"  [FILTERED] Blacklisted author: {author}")
                        continue
                
                # Check if we're filtering to specific authors
                if filter_authors:
                    # Normalize author for comparison (username and display name)
                    author_lower = author.lower()
                    # Check if author matches any of the filter authors (case-insensitive)
                    if not any(fa.lower() in author_lower or author_lower in fa.lower() for fa in filter_authors):
                        self.logger.info(f"  [FILTERED] Author {author} not in filter list {filter_authors}")
                        continue
                
                formatted_results.append(
                    {
                        "content": document,
                        "metadata": metadata,
                        "similarity": similarity,
                    }
                )
                
                self.logger.info(f"  [INCLUDED] (total so far: {len(formatted_results)}/{top_k})")
                
                # Stop once we have enough results
                if len(formatted_results) >= top_k:
                    break

        self.logger.info(f"Final results: {len(formatted_results)} chunks returned (top_k={top_k})")
        return formatted_results

    def search_bm25(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
        top_k: int = 10,
        exclude_blacklisted: bool = True,
        filter_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Keyword-based search using BM25 algorithm.
        """

        strategy_value = (strategy or ChunkStrategy(self.active_strategy)).value
        collection_name = f"discord_chunks_{strategy_value}"

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
            tokenized_corpus = [self._tokenize(doc['document']) for doc in all_docs]

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
        tokenized_query = self._tokenize(query)

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

            # Apply filters
            if exclude_blacklisted:
                from config import Config
                if author in Config.BLACKLIST_IDS or str(author) in [str(bid) for bid in Config.BLACKLIST_IDS]:
                    continue

            if filter_authors:
                author_lower = author.lower()
                if not any(fa.lower() in author_lower or author_lower in fa.lower() for fa in filter_authors):
                    continue

            results.append({
                "content": document,
                "metadata": metadata,
                "bm25_score": float(score),
                "similarity": float(score) / 100.0  # Normalize for consistency
            })

            if len(results) >= top_k:
                break

        self.logger.info(f"BM25 search returned {len(results)} results")
        return results

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) > 2] 

    def search_hybrid(
        self,
        query: str,
        strategy: Optional[ChunkStrategy] = None,
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
            strategy: Chunking strategy
            top_k: Number of results
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            exclude_blacklisted: Filter blacklisted authors
            filter_authors: Filter to specific authors

        Returns:
            Fused results from both search methods
        """
        from rag.hybrid_search import HybridSearchService

        # Retrieve more candidates from each method
        fetch_k = top_k * 3

        # BM25 search
        bm25_results = self.search_bm25(
            query=query,
            strategy=strategy,
            top_k=fetch_k,
            exclude_blacklisted=exclude_blacklisted,
            filter_authors=filter_authors
        )

        # Vector search
        vector_results = self.search(
            query=query,
            strategy=strategy,
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
            strategies: List of strategies to use (default: from config, typically single+tokens)
            
        Returns:
            Statistics dictionary with per-strategy results
        """
        start_time = datetime.now()
        
        # Use configured default strategies if none specified
        if strategies is None:
            from config import Config
            default_strategies = Config.CHUNKING_DEFAULT_STRATEGIES.split(',')
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