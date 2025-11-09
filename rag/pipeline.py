import logging
from typing import List, Dict, Optional
from storage.chunked_memory import ChunkedMemoryService
from ai.service import AIService
from storage.messages.messages import MessageStorage
from .models import RAGConfig, RAGResult
from chunking.constants import ChunkStrategy

class RAGPipeline:
    
    def __init__(
        self,
        chunked_memory_service: Optional[ChunkedMemoryService] = None,
        ai_service: Optional[AIService] = None,
        message_storage: Optional[MessageStorage] = None,
    ):
        self.chunked_memory = chunked_memory_service or ChunkedMemoryService()
        self.ai_service = ai_service or AIService()
        self.message_storage = message_storage or MessageStorage()
        self.logger = logging.getLogger(__name__)

    async def answer_question(
        self,
        question: str,
        config: Optional[RAGConfig] = None,
    ) -> RAGResult:
        """
        Execute the complete RAG pipeline.
        
        Pipeline stages:
        1. Retrieve relevant chunks
        2. Filter by similarity
        3. Build context with metadata
        4. Generate answer with LLM
        5. Return structured result

        """
        config = config or RAGConfig()
        
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
        1. Embed the query (convert text → vector)
        2. Search vector store (find similar vectors)
        3. Return top_k most similar chunks
        """
        filter_info = f" (filtering to authors: {config.filter_authors})" if config.filter_authors else ""
        self.logger.info(f"Retrieving top {config.top_k} chunks with strategy: {config.strategy}{filter_info}")
        
        try:
            strategy = ChunkStrategy(config.strategy)
        except ValueError:
            self.logger.warning(f"Invalid strategy '{config.strategy}', using default")
            strategy = ChunkStrategy.TOKENS
        
        chunks = self.chunked_memory.search(
            query=query,
            strategy=strategy,
            top_k=config.top_k,
            filter_authors=config.filter_authors,
        )
        
        self.logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks

    def _filter_by_similarity(
        self,
        chunks: List[Dict],
        threshold: float,
    ) -> List[Dict]:
        """
        Filter chunks below similarity threshold.
        
        Similarity scores:
        - 1.0 = identical match
        - 0.7-0.9 = very relevant
        - 0.5-0.7 = somewhat relevant
        - <0.5 = probably not relevant
        """
        self.logger.info(f"Applying similarity threshold: {threshold}")
        
        filtered = []
        for i, chunk in enumerate(chunks):
            similarity = chunk.get('similarity', 0)
            content_preview = chunk.get('content', '')[:80] + "..."
            
            if similarity >= threshold:
                self.logger.info(f"  [KEPT] Chunk {i+1}: {similarity:.3f} >= {threshold} - '{content_preview}'")
                filtered.append(chunk)
            else:
                self.logger.info(f"  [SKIP] Chunk {i+1}: {similarity:.3f} < {threshold} - '{content_preview}'")
        
        self.logger.info(
            f"Similarity filtering complete: {len(filtered)}/{len(chunks)} chunks passed "
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
            
            # Extract metadata
            timestamp = metadata.get('timestamp', 'Unknown time')
            author = metadata.get('author', 'Unknown')
            
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
        system_message = """You are a helpful discord bot that answers questions based on Discord conversation history.

        Instructions:
        - Answer ONLY using information from the provided context
        - If the context doesn't contain enough information, say so
        - Be concise and accurate
        - Reference specific messages when relevant
        - Do NOT make up information not in the context
        """

        user_message = f"""Context from Discord conversations:

        {context}

        ---

        Question: {question}

        Answer:"""

        # Combine into single prompt (simple format for AIService)
        full_prompt = f"{system_message}\n\n{user_message}"
        
        return full_prompt