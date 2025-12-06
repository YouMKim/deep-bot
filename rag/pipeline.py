import logging
from typing import List, Dict, Optional, TYPE_CHECKING
from storage.chunked_memory import ChunkedMemoryService
from ai.service import AIService
from storage.messages.messages import MessageStorage
from .models import RAGConfig, RAGResult
from chunking.constants import ChunkStrategy
from rag.query_enhancement import QueryEnhancementService
from rag.hybrid_search import reciprocal_rank_fusion
from .validation import QueryValidator

if TYPE_CHECKING:
    from config import Config

class RAGPipeline:
    
    def __init__(
        self,
        chunked_memory_service: Optional[ChunkedMemoryService] = None,
        ai_service: Optional[AIService] = None,
        message_storage: Optional[MessageStorage] = None,
        config: Optional['Config'] = None,
    ):
        from config import Config as ConfigClass  # Import here to avoid circular
        
        self.config = config or ConfigClass
        self.logger = logging.getLogger(__name__)
        # Initialize ChunkedMemoryService with error handling for ChromaDB issues
        try:
            self.chunked_memory = chunked_memory_service or ChunkedMemoryService(config=self.config)
        except (KeyError, AttributeError, TypeError, RuntimeError) as e:
            from storage.utils import handle_chromadb_init_error
            is_chromadb_error, runtime_error = handle_chromadb_init_error(e, "RAGPipeline initialization")
            if is_chromadb_error:
                raise runtime_error
            raise
        self.ai_service = ai_service or AIService()
        self.query_enhancer = QueryEnhancementService(ai_service=self.ai_service)
        self.message_storage = message_storage or MessageStorage()
        self.reranker = None 
        self._reranker_prewarmed = False
    
    async def prewarm_models(self) -> None:
        """
        Pre-warm models to avoid cold start latency on first query.
        
        This loads the reranker model in the background so the first
        query doesn't incur the model loading penalty.
        
        Should be called during bot startup (e.g., in setup_hook).
        """
        import asyncio
        
        if self._reranker_prewarmed:
            self.logger.debug("Reranker already prewarmed, skipping")
            return
        
        try:
            self.logger.info("Pre-warming reranker model...")
            
            # Load reranker in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def load_reranker():
                from rag.reranking import ReRankingService
                return ReRankingService()
            
            self.reranker = await loop.run_in_executor(None, load_reranker)
            self._reranker_prewarmed = True
            self.logger.info("Reranker model pre-warmed successfully")
            
        except ImportError as e:
            self.logger.warning(
                f"Could not prewarm reranker: {e}. "
                "Reranking will be disabled or load on first use."
            )
        except Exception as e:
            self.logger.warning(f"Failed to prewarm reranker: {e}") 

    async def answer_question(
        self,
        question: str,
        config: Optional[RAGConfig] = None,
        user_id: Optional[str] = None,
        user_display_name: Optional[str] = None,
    ) -> RAGResult:
        """
        Execute the complete RAG pipeline.
        
        Pipeline stages:
        0. Validate and sanitize query
        1. Retrieve relevant chunks
        2. Filter by similarity
        3. Build context with metadata
        4. Generate answer with LLM
        5. Return structured result

        """
        config = config or RAGConfig()
        
        # Stage 0: Validate input
        try:
            question = await QueryValidator.validate(
                question,
                user_id=user_id,
                user_display_name=user_display_name,
                social_credit_manager=getattr(self.ai_service, 'social_credit_manager', None)
            )
        except ValueError as e:
            self.logger.warning(f"Query validation failed: {e}")
            return RAGResult(
                answer=f"Invalid query: {str(e)}",
                sources=[],
                config_used=config,
                model="none",
            )
        
        self.logger.info(f"Starting RAG pipeline for question: {question[:50]}...")
        
        try:
            # Stage 1: Retrieve
            chunks = await self._retrieve_chunks(question, config)
            
            # Stage 2: Filter
            filtered_chunks = self._filter_by_similarity(chunks, config.similarity_threshold)
            
            if not filtered_chunks:
                return RAGResult(
                    answer="I couldn't find any relevant information to answer that question.",
                    sources=[],
                    config_used=config,
                    model="none",
                )
            
            # Stage 3: Build Context
            context = self._build_context_with_metadata(
                filtered_chunks,
                max_tokens=config.max_context_tokens
            )
            
            # Stage 4: Generate Answer
            prompt = self._create_rag_prompt(question, context)
            generation_result = await self.ai_service.generate(
                prompt=prompt,
                max_tokens=config.max_output_tokens, 
                temperature=config.temperature,
                user_id=user_id,
                user_display_name=user_display_name,
            )
            
            # Stage 5: Return Result
            return RAGResult(
                answer=generation_result['content'],
                sources=filtered_chunks,
                config_used=config,
                tokens_used=generation_result['tokens_total'],
                cost=generation_result['cost'],
                model=generation_result['model'],
            )
            
        except Exception as e:
            self.logger.error(f"RAG pipeline failed: {e}", exc_info=True)
            return RAGResult(
                answer=f"Sorry, I encountered an error: {str(e)}",
                sources=[],
                config_used=config,
                model="error",
            )

    async def _retrieve_chunks(
        self,
        query: str,
        config: RAGConfig,
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using vector similarity search.
        
        How it works:
        1. Check for multi-query (early return if enabled)
        2. Embed the query (convert text → vector)
        3. Search vector store (find similar vectors)
        4. Re-rank if enabled (optional refinement)
        5. Return top_k most similar chunks
        """
        
        # Handle HyDE: Generate hypothetical answer and use it for search
        search_query = query
        if config.use_hyde:
            self.logger.info("Generating HyDE document for query enhancement")
            hyde_doc = await self.query_enhancer.generate_hyde_document(query)
            search_query = hyde_doc  # Use HyDE document instead of original query
        
        # Handle multi-query first (it does its own retrieval)
        if config.use_multi_query:
            return await self._retrieve_multi_query(search_query, config)
        
        filter_info = f" (filtering to authors: {config.filter_authors})" if config.filter_authors else ""
        search_method = "hybrid" if config.use_hybrid_search else "vector"
        hyde_info = " (HyDE)" if config.use_hyde else ""
        self.logger.info(
            f"Retrieving top {config.top_k} chunks with strategy: {config.strategy} "
            f"using {search_method} search{hyde_info}{filter_info}"
        )

        try:
            strategy = ChunkStrategy(config.strategy)
        except ValueError:
            self.logger.warning(f"Invalid strategy '{config.strategy}', using default")
            strategy = ChunkStrategy.TOKENS

        # Retrieve more candidates if reranking is enabled
        # (reranker needs more options to choose from)
        from storage.chunked_memory.utils import calculate_fetch_k, get_collection_name
        fetch_k = calculate_fetch_k(config.top_k, needs_reranking=config.use_reranking)
        
        # Also search bot knowledge collection
        bot_docs_results = []
        try:
            bot_docs_collection = get_collection_name('bot_docs')
            collections = self.chunked_memory.vector_store.list_collections()
            if bot_docs_collection in collections:
                # Search bot docs collection directly
                import asyncio
                loop = asyncio.get_event_loop()
                query_embedding = await loop.run_in_executor(
                    None,
                    self.chunked_memory.embedder.encode,
                    search_query
                )
                
                bot_docs_raw = self.chunked_memory.vector_store.query(
                    collection_name=bot_docs_collection,
                    query_embeddings=[query_embedding],
                    n_results=min(config.top_k // 2, 5)  # Get fewer from bot docs
                )
                
                # Format results similar to retrieval_service
                if bot_docs_raw and bot_docs_raw.get('documents') and bot_docs_raw['documents'][0]:
                    documents = bot_docs_raw['documents'][0]
                    metadatas = bot_docs_raw.get('metadatas', [[]])[0] if bot_docs_raw.get('metadatas') else []
                    distances = bot_docs_raw.get('distances', [[]])[0] if bot_docs_raw.get('distances') else []
                    
                    for idx, doc in enumerate(documents):
                        metadata = metadatas[idx] if idx < len(metadatas) else {}
                        distance = distances[idx] if idx < len(distances) else 1.0
                        similarity = 1 - distance if distance <= 1 else 0
                        
                        bot_docs_results.append({
                            'content': doc,
                            'metadata': {
                                **metadata,
                                'source': 'bot_documentation',
                                'collection': 'bot_docs',
                                'channel_id': 'system',
                                'author': 'system'
                            },
                            'similarity': similarity,
                            'distance': distance
                        })
        except Exception as e:
            self.logger.debug(f"Could not search bot knowledge: {e}")

        # Use hybrid search if enabled
        if config.use_hybrid_search:
            chunks = await self.chunked_memory.search_hybrid(
                query=search_query,  # Use HyDE-enhanced query if enabled
                strategy=strategy,
                top_k=fetch_k,  # Fetch more if reranking
                bm25_weight=config.bm25_weight,
                vector_weight=config.vector_weight,
                exclude_blacklisted=True,  # Explicitly exclude blacklisted authors
                filter_authors=config.filter_authors,
            )
        else:
            # Standard vector search
            chunks = await self.chunked_memory.search(
                query=search_query,  # Use HyDE-enhanced query if enabled
                strategy=strategy,
                top_k=fetch_k,  # Fetch more if reranking
                exclude_blacklisted=True,  # Explicitly exclude blacklisted authors
                filter_authors=config.filter_authors,
            )

        # Combine bot docs with regular chunks
        if bot_docs_results:
            chunks = bot_docs_results + chunks
            self.logger.info(f"Retrieved {len(chunks)} chunks ({len(bot_docs_results)} from bot docs, {len(chunks) - len(bot_docs_results)} from messages)")
        else:
            self.logger.info(f"Retrieved {len(chunks)} chunks")
        
        # Re-rank if enabled (AFTER retrieval)
        # Run reranking in executor to avoid blocking event loop
        if config.use_reranking and chunks:
            if self.reranker is None:
                # Lazy import to avoid errors if sentence-transformers not installed
                try:
                    from rag.reranking import ReRankingService
                    self.reranker = ReRankingService()
                except ImportError as e:
                    self.logger.error(
                        f"Failed to import ReRankingService: {e}. "
                        "Reranking requires sentence-transformers. "
                        "Install it with: pip install sentence-transformers tokenizers"
                    )
                    # Disable reranking for this request
                    return chunks

            import asyncio
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                self.reranker.rerank,
                query,
                chunks,
                config.top_k  # Return top_k after reranking
            )
            self.logger.info(f"Re-ranked to {len(chunks)} chunks")

        return chunks

    def _filter_by_similarity(
        self,
        chunks: List[Dict],
        threshold: float,
    ) -> List[Dict]:
        """
        Filter chunks below similarity threshold.

        Supports different scoring methods:
        - ce_score: Cross-encoder reranking scores (highest priority)
        - rrf_score: Reciprocal Rank Fusion scores (hybrid search)
        - similarity: Standard vector/BM25 similarity scores

        Similarity score ranges (vector/BM25):
        - 1.0 = identical match
        - 0.7-0.9 = very relevant
        - 0.5-0.7 = somewhat relevant
        - <0.5 = probably not relevant

        RRF score ranges (hybrid search):
        - ~0.016+ = very relevant (top results from both methods)
        - ~0.008-0.016 = relevant
        - <0.008 = less relevant
        """
        # Detect which scoring method was used
        score_field = 'similarity'  # Default
        if chunks and 'ce_score' in chunks[0]:
            score_field = 'ce_score'
            score_type = "cross-encoder"
        elif chunks and 'rrf_score' in chunks[0]:
            score_field = 'rrf_score'
            score_type = "RRF"
        else:
            score_type = "similarity"

        self.logger.info(f"Applying {score_type} threshold: {threshold} (using field: {score_field})")

        filtered = []
        for i, chunk in enumerate(chunks):
            score = chunk.get(score_field, 0)
            content_preview = chunk.get('content', '')[:80] + "..."

            if score >= threshold:
                self.logger.info(f"  [KEPT] Chunk {i+1}: {score:.3f} >= {threshold} - '{content_preview}'")
                filtered.append(chunk)
            else:
                self.logger.info(f"  [SKIP] Chunk {i+1}: {score:.3f} < {threshold} - '{content_preview}'")

        self.logger.info(
            f"{score_type.capitalize()} filtering complete: {len(filtered)}/{len(chunks)} chunks passed "
            f"(removed {len(chunks) - len(filtered)})"
        )

        return filtered

    def _build_context_with_metadata(
        self,
        chunks: List[Dict],
        max_tokens: int,
    ) -> str:
        """
        Build context string with timestamp and author metadata.
        Format: [timestamp] author: content
        """
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  
        
        for chunk in chunks:
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            
            # Extract metadata - try multiple field names for compatibility
            author = (
                metadata.get('author') or 
                metadata.get('primary_author_name') or 
                metadata.get('primary_author_id') or 
                'Unknown'
            )
            timestamp = (
                metadata.get('first_timestamp') or 
                metadata.get('timestamp') or 
                metadata.get('last_timestamp') or 
                'Unknown time'
            )
            
            # Format with metadata
            formatted = f"[{timestamp}] {author}: {content}"
            
            # Check token budget
            chunk_chars = len(formatted)
            if total_chars + chunk_chars > max_chars:
                self.logger.info(
                    f"Reached token limit, including {len(context_parts)}/{len(chunks)} chunks"
                )
                break
            
            context_parts.append(formatted)
            total_chars += chunk_chars
        
        context = "\n\n".join(context_parts)
        
        self.logger.info(
            f"Built context with {len(context_parts)} chunks, "
            f"~{total_chars // 4} tokens"
        )
        
        return context

    def _create_rag_prompt(self, question: str, context: str, max_tokens: int = 800) -> str:
        """
        Create prompt for LLM with context and question.
        """
        system_message = """You are a helpful Discord bot that answers questions based on Discord conversation history. You're chatting with friends, not writing a report.

Your goal is to have a natural conversation that answers their question. Think of it like you're catching up with someone and sharing what you remember from past conversations.

Write like you're actually talking:
- Use natural transitions and flow between ideas
- Connect related thoughts together smoothly
- Reference people by name when it makes sense (e.g., "Oh yeah, Thomas mentioned that...")
- Show how ideas evolved over time if relevant
- If you're not sure about something, say so casually
- Keep it conversational - imagine you're explaining this to a friend in voice chat

Only use information from the provided context. Don't make things up. If there's not enough info, just say so naturally like you would in a real conversation.

Write in flowing paragraphs - no bullet points or lists. Just talk naturally about what you found.
        """

        # Use list join instead of f-string concatenation for better memory efficiency
        parts = [
            system_message.strip(),
            "",
            "Context from Discord conversations:",
            "",
            context,
            "",
            "---",
            "",
            f"Question: {question}",
            "",
            "Answer (have a natural conversation about this):"
        ]
        
        return "\n".join(parts)

    async def _retrieve_multi_query(
        self,
        query: str,
        config: RAGConfig
    ) -> List[Dict]:
        """
        Retrieve using multiple query variations and fuse results.
        
        Runs all query variations in PARALLEL for better performance.
        """
        import asyncio
        
        # Generate query variations
        queries = await self.query_enhancer.generate_multi_queries(
            query,
            num_queries=config.num_query_variations
        )

        self.logger.info(f"Retrieving with {len(queries)} query variations (parallel)")

        # Get strategy with fallback (same as in _retrieve_chunks)
        try:
            strategy = ChunkStrategy(config.strategy)
        except ValueError:
            self.logger.warning(f"Invalid strategy '{config.strategy}', using default")
            strategy = ChunkStrategy.TOKENS

        # Retrieve with each query
        # Use configurable multiplier (default 1.5, reduced from 2.0)
        multi_query_multiplier = self.config.RAG_MULTI_QUERY_MULTIPLIER
        fetch_k_multi = int(config.top_k * multi_query_multiplier)
        
        # Create search tasks for all queries in PARALLEL
        async def search_single_query(q: str):
            if config.use_hybrid_search:
                return await self.chunked_memory.search_hybrid(
                    query=q,
                    strategy=strategy,
                    top_k=fetch_k_multi,
                    exclude_blacklisted=True,
                    filter_authors=config.filter_authors
                )
            else:
                return await self.chunked_memory.search(
                    query=q,
                    strategy=strategy,
                    top_k=fetch_k_multi,
                    exclude_blacklisted=True,
                    filter_authors=config.filter_authors
                )

        # Run all searches in parallel
        tasks = [search_single_query(q) for q in queries]
        all_results = await asyncio.gather(*tasks)

        # Fuse all results using RRF
        fused_results = reciprocal_rank_fusion(
            ranked_lists=list(all_results),
            top_k=config.top_k
        )

        self.logger.info(
            f"Multi-query fusion: {len(queries)} queries → "
            f"{sum(len(r) for r in all_results)} total results → "
            f"{len(fused_results)} fused results"
        )

        return fused_results