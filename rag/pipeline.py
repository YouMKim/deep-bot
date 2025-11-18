import logging
from typing import List, Dict, Optional, TYPE_CHECKING
from storage.chunked_memory import ChunkedMemoryService
from ai.service import AIService
from storage.messages.messages import MessageStorage
from .models import RAGConfig, RAGResult
from chunking.constants import ChunkStrategy
from rag.query_enhancement import QueryEnhancementService
from rag.hybrid_search import reciprocal_rank_fusion
from rag.reranking import ReRankingService
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
        self.chunked_memory = chunked_memory_service or ChunkedMemoryService(config=self.config)
        self.ai_service = ai_service or AIService()
        self.query_enhancer = QueryEnhancementService(ai_service=self.ai_service)
        self.message_storage = message_storage or MessageStorage()
        self.logger = logging.getLogger(__name__)
        self.reranker = None 

    async def answer_question(
        self,
        question: str,
        config: Optional[RAGConfig] = None,
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
            question = QueryValidator.validate(question)
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
                max_tokens=1000, 
                temperature=config.temperature,
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
        
        # Handle multi-query first (it does its own retrieval)
        if config.use_multi_query:
            return await self._retrieve_multi_query(query, config)
        
        filter_info = f" (filtering to authors: {config.filter_authors})" if config.filter_authors else ""
        search_method = "hybrid" if config.use_hybrid_search else "vector"
        self.logger.info(
            f"Retrieving top {config.top_k} chunks with strategy: {config.strategy} "
            f"using {search_method} search{filter_info}"
        )

        try:
            strategy = ChunkStrategy(config.strategy)
        except ValueError:
            self.logger.warning(f"Invalid strategy '{config.strategy}', using default")
            strategy = ChunkStrategy.TOKENS

        # Retrieve more candidates if reranking is enabled
        # (reranker needs more options to choose from)
        from storage.chunked_memory.utils import calculate_fetch_k
        fetch_k = calculate_fetch_k(config.top_k, needs_reranking=config.use_reranking)

        # Use hybrid search if enabled
        if config.use_hybrid_search:
            chunks = self.chunked_memory.search_hybrid(
                query=query,
                strategy=strategy,
                top_k=fetch_k,  # Fetch more if reranking
                bm25_weight=config.bm25_weight,
                vector_weight=config.vector_weight,
                filter_authors=config.filter_authors,
            )
        else:
            # Standard vector search
            chunks = self.chunked_memory.search(
                query=query,
                strategy=strategy,
                top_k=fetch_k,  # Fetch more if reranking
                filter_authors=config.filter_authors,
            )

        self.logger.info(f"Retrieved {len(chunks)} chunks")
        
        # Re-rank if enabled (AFTER retrieval)
        if config.use_reranking and chunks:
            if self.reranker is None:
                self.reranker = ReRankingService()

            chunks = self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=config.top_k  # Return top_k after reranking
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

    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create prompt for LLM with context and question.
        """
        system_message = """You are a helpful Discord bot that answers questions based on Discord conversation history.

        Instructions:
        - Answer ONLY using information from the provided context
        - Write in a natural, conversational style like a Discord message
        - Write as flowing paragraphs, NOT bullet points or lists
        - If the context doesn't contain enough information, say so naturally
        - When referencing information, mention the person's name and date naturally in the text (e.g., "Thomas mentioned on 2024-06-11 that...")
        - Do NOT use bullet points, numbered lists, or structured formats
        - Do NOT make up information not in the context
        - Write like you're explaining something to a friend in Discord chat
        """

        user_message = f"""Context from Discord conversations:

        {context}

        ---

        Question: {question}

        Answer (write naturally as a Discord message, no bullet points):"""

        # Combine into single prompt (simple format for AIService)
        full_prompt = f"{system_message}\n\n{user_message}"
        
        return full_prompt

    async def _retrieve_multi_query(
        self,
        query: str,
        config: RAGConfig
    ) -> List[Dict]:
        """
        Retrieve using multiple query variations and fuse results.
        """
        # Generate query variations
        queries = await self.query_enhancer.generate_multi_queries(
            query,
            num_queries=config.num_query_variations
        )

        self.logger.info(f"Retrieving with {len(queries)} query variations")

        # Get strategy with fallback (same as in _retrieve_chunks)
        try:
            strategy = ChunkStrategy(config.strategy)
        except ValueError:
            self.logger.warning(f"Invalid strategy '{config.strategy}', using default")
            strategy = ChunkStrategy.TOKENS

        # Retrieve with each query
        all_results = []
        for q in queries:
            if config.use_hybrid_search:
                results = self.chunked_memory.search_hybrid(
                    query=q,
                    strategy=strategy,
                    top_k=config.top_k * 2,  # Get more candidates
                    filter_authors=config.filter_authors
                )
            else:
                results = self.chunked_memory.search(
                    query=q,
                    strategy=strategy,
                    top_k=config.top_k * 2,
                    filter_authors=config.filter_authors
                )

            all_results.append(results)

        # Fuse all results using RRF
        fused_results = reciprocal_rank_fusion(
            ranked_lists=all_results,
            top_k=config.top_k
        )

        self.logger.info(
            f"Multi-query fusion: {len(queries)} queries → "
            f"{sum(len(r) for r in all_results)} total results → "
            f"{len(fused_results)} fused results"
        )

        return fused_results