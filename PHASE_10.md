# Phase 10: RAG Query Pipeline

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

RAG Query Pipeline - Complete End-to-End Implementation

### Learning Objectives
- Understand the complete RAG pipeline (Retrieve → Generate)
- Learn context formatting for LLMs
- Practice source citation and attribution
- Implement query optimization techniques
- Handle edge cases (no results, context overflow)

### Design Principles
- **User-Centric**: Focus on answer quality and usefulness
- **Transparency**: Always show sources
- **Fault Tolerance**: Graceful degradation when retrieval fails
- **Performance**: Fast retrieval + generation

### The RAG Pipeline

```
User Query
    ↓
1. Retrieve relevant chunks (vector similarity search)
    ↓
2. Filter by similarity threshold
    ↓
3. Build context from chunks
    ↓
4. Generate answer with LLM
    ↓
5. Return answer + sources
```

---

## Implementation Steps

### Step 10.1: Create RAG Query Service

Create `rag/pipeline.py`:

```python
"""
Complete RAG query pipeline for Discord message retrieval.

Learning: This is the culmination of all previous phases.
Brings together: chunking, embedding, retrieval, and generation.
"""

from storage.chunked_memory import ChunkedMemoryService
from ai.service import AIService
from typing import List, Dict, Optional
from config import Config
import logging

class RAGQueryService:
    """
    End-to-end RAG pipeline: Query → Retrieve → Generate

    Learning: This is what users actually interact with.
    Everything else (chunking, embedding, storage) is infrastructure
    to make this work well.
    """

    def __init__(
        self,
        chunked_memory: ChunkedMemoryService,
        ai_service: AIService
    ):
        self.memory = chunked_memory
        self.ai = ai_service
        self.logger = logging.getLogger(__name__)

    async def query(
        self,
        user_question: str,
        strategy: str = None,
        top_k: int = None,
        min_similarity: float = None,
        include_sources: bool = None
    ) -> Dict[str, any]:
        """
        Execute RAG pipeline for a user question.

        Args:
            user_question: The question to answer
            strategy: Chunking strategy to use (default from config)
            top_k: Number of chunks to retrieve (default from config)
            min_similarity: Minimum similarity threshold (default from config)
            include_sources: Include source citations (default from config)

        Returns:
            {
                "answer": "The team meeting is at 3pm...",
                "sources": [...],
                "chunks_retrieved": 5,
                "chunks_used": 3,
                "strategy": "token_aware"
            }

        Learning: Returning metadata helps debug and improve the system.
        """
        # Use config defaults if not specified
        strategy = strategy or Config.RAG_DEFAULT_STRATEGY
        top_k = top_k or Config.RAG_TOP_K
        min_similarity = min_similarity or Config.RAG_MIN_SIMILARITY
        include_sources = include_sources if include_sources is not None else Config.RAG_INCLUDE_SOURCES

        self.logger.info(
            f"RAG query: '{user_question[:50]}...' "
            f"(strategy={strategy}, top_k={top_k})"
        )

        # Step 1: Retrieve relevant chunks
        chunks = self.memory.search(
            query=user_question,
            strategy=strategy,
            top_k=top_k
        )

        chunks_retrieved = len(chunks)

        # Step 2: Filter by similarity threshold
        filtered_chunks = [
            chunk for chunk in chunks
            if chunk.get('similarity', 0) >= min_similarity
        ]

        chunks_used = len(filtered_chunks)

        # Handle no results
        if not filtered_chunks:
            return {
                "answer": self._no_results_message(user_question),
                "sources": [],
                "chunks_retrieved": chunks_retrieved,
                "chunks_used": 0,
                "strategy": strategy,
                "fallback": True
            }

        # Step 3: Build context from chunks
        context = self._build_context(filtered_chunks)

        # Check context length
        # TODO: Add token counting here to prevent overflow
        # For now, truncate if needed
        if len(context) > Config.RAG_MAX_CONTEXT_TOKENS * 4:  # Rough estimate
            self.logger.warning(
                f"Context too long ({len(context)} chars), "
                f"may need to reduce top_k or improve filtering"
            )
            # Truncate context
            context = context[:Config.RAG_MAX_CONTEXT_TOKENS * 4]

        # Step 4: Generate answer with LLM
        answer = await self._generate_answer(user_question, context)

        # Step 5: Format sources
        sources = self._format_sources(filtered_chunks) if include_sources else []

        return {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": chunks_retrieved,
            "chunks_used": chunks_used,
            "strategy": strategy,
            "fallback": False
        }

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from chunks.

        Learning: Context formatting affects LLM quality!

        Good context:
        - Clear structure (numbered chunks)
        - Metadata (date, author, channel)
        - Actual message content

        Bad context:
        - Unstructured text dump
        - Missing metadata
        - Redundant information
        """
        if not chunks:
            return ""

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')
            similarity = chunk.get('similarity', 0)

            # Format chunk with metadata
            chunk_header = f"[Chunk {i}]"

            # Add metadata if available
            meta_parts = []
            if metadata.get('first_timestamp'):
                meta_parts.append(f"Date: {metadata['first_timestamp'][:10]}")
            if metadata.get('message_count'):
                meta_parts.append(f"{metadata['message_count']} messages")
            if metadata.get('channel_id'):
                meta_parts.append(f"Channel: {metadata['channel_id']}")

            if meta_parts:
                chunk_header += f" ({', '.join(meta_parts)}, relevance: {similarity:.2f})"

            # Combine header and content
            chunk_text = f"{chunk_header}\n{content}"
            context_parts.append(chunk_text)

        return "\n\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM.

        Learning: Prompt engineering matters!

        Good prompts:
        - Clear instructions
        - Context provided
        - Question stated clearly
        - Constraints (e.g., "answer based only on context")

        Bad prompts:
        - Vague instructions
        - Missing context
        - No constraints (LLM hallucinates)
        """
        prompt = f"""You are a helpful assistant answering questions based on Discord chat history.

Answer the following question based ONLY on the provided context from Discord messages.

Context from Discord messages:
{context}

Question: {question}

Instructions:
1. Answer the question based on the context above
2. If the context doesn't contain enough information, say so
3. Be concise but complete
4. Quote relevant parts of messages when useful
5. If the question cannot be answered from the context, say "I don't have enough information in the chat history to answer that question."

Answer:"""

        try:
            # Use your AIService to generate
            response = await self.ai.generate(
                prompt=prompt,
                max_tokens=500,  # Adjust based on your needs
                temperature=0.3  # Lower = more focused, higher = more creative
            )

            return response.strip()

        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return f"I encountered an error generating an answer: {str(e)}"

    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Format source citations.

        Learning: Good citations enable users to:
        - Verify the answer
        - Get more context
        - Trust the system

        Returns list of source dicts with metadata.
        """
        sources = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})

            source = {
                "rank": i,
                "similarity": chunk.get('similarity', 0),
                "message_count": metadata.get('message_count', 0),
                "channel_id": metadata.get('channel_id', 'unknown'),
                "date_range": self._format_date_range(metadata),
                "message_ids": metadata.get('message_ids', [])[:3],  # First 3 IDs
                "preview": chunk.get('content', '')[:100] + "..."  # First 100 chars
            }

            sources.append(source)

        return sources

    def _format_date_range(self, metadata: Dict) -> str:
        """Format date range from metadata"""
        first = metadata.get('first_timestamp', '')
        last = metadata.get('last_timestamp', '')

        if not first:
            return "unknown"

        first_date = first[:10] if first else ""
        last_date = last[:10] if last else ""

        if first_date == last_date:
            return first_date
        else:
            return f"{first_date} to {last_date}"

    def _no_results_message(self, question: str) -> str:
        """
        Message when no relevant chunks found.

        Learning: Graceful degradation improves UX.
        """
        return (
            f"I couldn't find any relevant information in the Discord chat "
            f"history to answer: \"{question}\"\n\n"
            f"This could mean:\n"
            f"- The topic wasn't discussed\n"
            f"- The messages aren't stored yet (try !chunk_channel)\n"
            f"- The similarity threshold is too high (try lowering it)\n"
            f"- The chunking strategy doesn't work well for this query"
        )
```

### Step 10.2: Add RAG Query Bot Command

Add to `bot/cogs/admin.py` or create new `bot/cogs/rag.py`:

```python
import discord
from discord.ext import commands
from rag.pipeline import RAGQueryService
from storage.chunked_memory import ChunkedMemoryService
from storage.vectors.factory import VectorStoreFactory
from embedding.factory import EmbeddingServiceFactory
from ai.service import AIService
from config import Config
import logging

class RAGCommands(commands.Cog):
    """RAG query commands for Discord bot"""

    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger(__name__)

    @commands.command(name='ask')
    async def ask(
        self,
        ctx,
        *,
        question: str
    ):
        """
        Ask a question about Discord chat history using RAG.

        Usage:
            !ask What time is the team meeting?
            !ask Who suggested using Docker?
            !ask What were the main concerns about the API?

        Learning: This is the user-facing RAG command.
        Everything else was infrastructure to make this work.
        """
        try:
            # Show typing indicator
            async with ctx.typing():
                # Initialize services
                vector_store = VectorStoreFactory.create()
                embedding_provider = EmbeddingServiceFactory.create()
                chunked_memory = ChunkedMemoryService(vector_store, embedding_provider)
                ai_service = AIService()  # Your existing AI service
                rag_service = RAGQueryService(chunked_memory, ai_service)

                # Execute RAG query
                result = await rag_service.query(question)

                # Create embed for response
                embed = discord.Embed(
                    title="💬 Answer",
                    description=result["answer"],
                    color=discord.Color.blue()
                )

                # Add metadata
                embed.add_field(
                    name="📊 Retrieval Stats",
                    value=(
                        f"Chunks retrieved: {result['chunks_retrieved']}\n"
                        f"Chunks used: {result['chunks_used']}\n"
                        f"Strategy: {result['strategy']}"
                    ),
                    inline=False
                )

                # Add sources if available
                if result.get('sources'):
                    sources_text = ""
                    for source in result['sources'][:3]:  # Show top 3
                        sources_text += (
                            f"**{source['rank']}.** {source['date_range']} "
                            f"({source['message_count']} msgs, "
                            f"similarity: {source['similarity']:.2f})\n"
                        )

                    embed.add_field(
                        name="📚 Sources",
                        value=sources_text or "No sources",
                        inline=False
                    )

                # Set footer
                embed.set_footer(text=f"Question: {question[:100]}")

                await ctx.send(embed=embed)

        except Exception as e:
            self.logger.error(f"Error in ask command: {e}", exc_info=True)
            await ctx.send(f"❌ Error: {e}")

    @commands.command(name='ask_detailed')
    @commands.is_owner()
    async def ask_detailed(
        self,
        ctx,
        strategy: str,
        top_k: int,
        *,
        question: str
    ):
        """
        Ask with custom parameters (for testing/tuning).

        Usage: !ask_detailed token_aware 10 What time is the meeting?

        Learning: Useful for experimenting with different strategies and top_k values.
        """
        if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
            await ctx.send("🚫 **Access Denied!**")
            return

        try:
            async with ctx.typing():
                # Initialize services
                vector_store = VectorStoreFactory.create()
                embedding_provider = EmbeddingServiceFactory.create()
                chunked_memory = ChunkedMemoryService(vector_store, embedding_provider)
                ai_service = AIService()
                rag_service = RAGQueryService(chunked_memory, ai_service)

                # Execute with custom parameters
                result = await rag_service.query(
                    question,
                    strategy=strategy,
                    top_k=top_k
                )

                # Send detailed response
                embed = discord.Embed(
                    title=f"💬 Answer (strategy={strategy}, top_k={top_k})",
                    description=result["answer"],
                    color=discord.Color.green() if not result.get('fallback') else discord.Color.orange()
                )

                embed.add_field(
                    name="📊 Stats",
                    value=(
                        f"Retrieved: {result['chunks_retrieved']}\n"
                        f"Used: {result['chunks_used']}\n"
                        f"Fallback: {result.get('fallback', False)}"
                    )
                )

                # Show all sources
                if result.get('sources'):
                    for i, source in enumerate(result['sources'][:5], 1):
                        embed.add_field(
                            name=f"Source {i}",
                            value=(
                                f"Date: {source['date_range']}\n"
                                f"Similarity: {source['similarity']:.3f}\n"
                                f"Messages: {source['message_count']}\n"
                                f"Preview: {source['preview'][:50]}..."
                            ),
                            inline=True
                        )

                await ctx.send(embed=embed)

        except Exception as e:
            self.logger.error(f"Error in ask_detailed: {e}", exc_info=True)
            await ctx.send(f"❌ Error: {e}")

async def setup(bot):
    await bot.add_cog(RAGCommands(bot))
```

### Step 10.3: Advanced: Query Optimization

Create `rag/query_optimizer.py` (optional, advanced):

```python
"""
Query optimization techniques for better RAG retrieval.

Learning: These techniques improve retrieval quality.
"""

from typing import List
import logging

class QueryOptimizer:
    """
    Optimize queries for better retrieval.

    Techniques:
    1. Query expansion: Generate related queries
    2. Keyword extraction: Extract key terms
    3. Query rewriting: Rephrase for better matching
    """

    def __init__(self, ai_service):
        self.ai = ai_service
        self.logger = logging.getLogger(__name__)

    async def expand_query(self, query: str) -> List[str]:
        """
        Generate multiple query variations.

        Learning: Single query might miss relevant docs.
        Multiple queries increase recall.

        Example:
            Input: "What time is the meeting?"
            Output: [
                "What time is the meeting?",
                "When is our meeting scheduled?",
                "Meeting time and date"
            ]
        """
        prompt = f"""Generate 2-3 variations of this query that would help find the same information:

Original query: {query}

Variations (one per line):"""

        try:
            response = await self.ai.generate(prompt, max_tokens=100, temperature=0.7)
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return [query] + variations[:2]  # Original + 2 variations
        except Exception as e:
            self.logger.error(f"Error expanding query: {e}")
            return [query]  # Fallback to original only

    async def extract_keywords(self, query: str) -> List[str]:
        """
        Extract key terms from query.

        Learning: Sometimes keyword search works better than semantic.
        Can be combined with vector search (hybrid search).
        """
        # Simple implementation: extract words
        words = query.lower().split()
        # Filter stop words (you, is, the, etc.)
        stop_words = {'what', 'when', 'where', 'who', 'how', 'is', 'the', 'a', 'an', 'to', 'for'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
```

---

## Usage Examples

### Basic Query

```python
# In Discord:
!ask What time is the team meeting?

# Response:
# Answer: Based on the chat history, the team meeting is scheduled for 3pm on Friday.
# Sources:
# 1. 2024-01-15 (2 msgs, similarity: 0.89)
# 2. 2024-01-14 (1 msg, similarity: 0.76)
```

### Programmatic Usage

```python
from rag.pipeline import RAGQueryService

# Initialize
rag = RAGQueryService(chunked_memory, ai_service)

# Query
result = await rag.query("What did Alice say about the project deadline?")

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
print(f"Chunks used: {result['chunks_used']}/{result['chunks_retrieved']}")
```

---

## Common Pitfalls - Phase 10

1. **No chunks stored**: Users must run `!chunk_channel` first
2. **Context too long**: Monitor context length, truncate if needed
3. **LLM hallucination**: Use good prompts with constraints
4. **Missing API key**: AI service needs API key configured
5. **Similarity threshold too high**: No results if threshold > max similarity
6. **Poor prompt**: Generic prompts = generic answers

## Debugging Tips - Phase 10

- **Test retrieval separately**: Use `!chunk_stats` to verify chunks exist
- **Check similarity scores**: Log similarities to understand threshold
- **Inspect context**: Log the full context sent to LLM
- **Test prompts**: Try different prompts in playground first
- **Monitor token usage**: Track costs if using paid API

## Performance Considerations - Phase 10

- **Caching**: Cache embeddings for common queries
- **Async operations**: Retrieval and generation can be parallelized
- **Batch queries**: If answering multiple questions, batch embed them
- **Token limits**: Monitor context + answer tokens

---

## Next Steps

After implementing RAG query:

1. **Test with real questions**: Ask questions you know the answers to
2. **Tune parameters**: Experiment with top_k, similarity threshold
3. **Add feedback loop**: Track which answers are helpful
4. **Monitor quality**: Use evaluation metrics from Phase 6.5
5. **Iterate**: Add features like multi-query, reranking, hybrid search

**Congratulations!** You now have a complete RAG system. 🎉

