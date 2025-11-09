# RAG System Implementation Guide

**Focus:** Immediate improvements to fix "!ask coming up short" issue
**Timeline:** This week
**Goal:** Significantly improve retrieval quality

---

## 🎯 Priority 1: Hybrid Search (BM25 + Vector)

### Why This First?
- **Biggest impact** on retrieval quality (30-50% improvement)
- Catches exact keyword matches that semantic search misses
- Relatively easy to implement
- No additional dependencies needed (just `rank_bm25`)

### Implementation Steps

#### Step 1: Install BM25 Library
```bash
pip install rank-bm25
```

Add to `requirements.txt`:
```
rank-bm25>=0.2.2
```

#### Step 2: Create Hybrid Search Service
**File:** `rag/hybrid_search.py`

```python
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np
import logging

class HybridSearchService:
    """
    Combines BM25 (keyword) and vector (semantic) search using Reciprocal Rank Fusion.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def hybrid_search(
        self,
        query: str,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Merge BM25 and vector search results using Reciprocal Rank Fusion.

        Args:
            query: Search query
            bm25_results: Results from BM25 search (with scores)
            vector_results: Results from vector search (with similarity)
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
            top_k: Number of results to return

        Returns:
            Merged and re-ranked results
        """
        # RRF constant (standard value)
        k = 60

        # Create lookup dictionaries
        all_docs = {}
        rrf_scores = {}

        # Add BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.get('metadata', {}).get('first_message_id', f"bm25_{rank}")
            all_docs[doc_id] = result
            rrf_scores[doc_id] = bm25_weight / (k + rank)

        # Add vector results (merge scores if already seen)
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.get('metadata', {}).get('first_message_id', f"vector_{rank}")
            if doc_id not in all_docs:
                all_docs[doc_id] = result
                rrf_scores[doc_id] = 0

            rrf_scores[doc_id] += vector_weight / (k + rank)

        # Sort by RRF score
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k with updated scores
        results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = all_docs[doc_id]
            doc['rrf_score'] = rrf_score
            doc['fusion_rank'] = len(results) + 1
            results.append(doc)

        self.logger.info(
            f"Hybrid search: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"→ {len(results)} fused results"
        )

        return results


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    top_k: int = 10,
    k_constant: int = 60
) -> List[Dict]:
    """
    Generic RRF implementation for merging multiple ranked lists.

    Args:
        ranked_lists: List of ranked result lists (each list already sorted by relevance)
        top_k: Number of final results to return
        k_constant: RRF constant (default: 60)

    Returns:
        Merged and re-ranked results
    """
    all_docs = {}
    rrf_scores = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            # Use first_message_id as unique document identifier
            doc_id = doc.get('metadata', {}).get('first_message_id', f"doc_{id(doc)}")

            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                rrf_scores[doc_id] = 0

            # RRF formula: 1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k_constant + rank)

    # Sort by RRF score
    sorted_docs = sorted(
        [(doc_id, score) for doc_id, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-k
    results = []
    for doc_id, score in sorted_docs[:top_k]:
        doc = all_docs[doc_id]
        doc['rrf_score'] = score
        results.append(doc)

    return results
```

#### Step 3: Add BM25 Search to ChunkedMemoryService
**File:** `storage/chunked_memory.py`

Add BM25 search method:

```python
from rank_bm25 import BM25Okapi
import re

class ChunkedMemoryService:
    # ... existing code ...

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

        Args:
            query: Search query
            strategy: Chunking strategy to search
            top_k: Number of results
            exclude_blacklisted: Filter blacklisted authors
            filter_authors: Filter to specific authors

        Returns:
            List of BM25-ranked chunks with scores
        """
        strategy_value = (strategy or ChunkStrategy(self.active_strategy)).value
        collection_name = f"discord_chunks_{strategy_value}"

        try:
            # Get all documents from collection
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

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)

        # Sort by score and create results
        scored_docs = list(zip(all_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Format results
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
        """
        Simple tokenization for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Filter short tokens

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
```

#### Step 4: Add `get_all_documents` to Vector Store
**File:** `storage/vectors/providers/chroma.py`

```python
def get_all_documents(
    self,
    collection_name: str
) -> List[Dict]:
    """
    Get all documents from a collection (for BM25 indexing).

    Args:
        collection_name: Name of collection

    Returns:
        List of documents with metadata
    """
    try:
        collection = self.client.get_collection(name=collection_name)
        results = collection.get()

        documents = []
        for i, doc in enumerate(results['documents']):
            documents.append({
                'document': doc,
                'metadata': results['metadatas'][i] if i < len(results['metadatas']) else {},
                'id': results['ids'][i] if i < len(results['ids']) else str(i)
            })

        return documents

    except Exception as e:
        self.logger.error(f"Failed to get all documents from {collection_name}: {e}")
        return []
```

#### Step 5: Update RAG Pipeline to Use Hybrid Search
**File:** `rag/pipeline.py`

```python
# Update _retrieve_chunks method

async def _retrieve_chunks(
    self,
    query: str,
    config: RAGConfig,
) -> List[Dict]:
    """
    Retrieve relevant chunks using vector similarity search or hybrid search.
    """
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

    # Use hybrid search if enabled
    if config.use_hybrid_search:
        chunks = self.chunked_memory.search_hybrid(
            query=query,
            strategy=strategy,
            top_k=config.top_k,
            bm25_weight=config.bm25_weight,
            vector_weight=config.vector_weight,
            filter_authors=config.filter_authors,
        )
    else:
        # Standard vector search
        chunks = self.chunked_memory.search(
            query=query,
            strategy=strategy,
            top_k=config.top_k,
            filter_authors=config.filter_authors,
        )

    self.logger.info(f"Retrieved {len(chunks)} chunks")
    return chunks
```

#### Step 6: Update RAGConfig
**File:** `rag/models.py`

```python
@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    # Retrieval settings
    top_k: int = 10
    similarity_threshold: float = 0.35
    strategy: str = "tokens"

    # Hybrid search settings
    use_hybrid_search: bool = False
    bm25_weight: float = 0.5
    vector_weight: float = 0.5

    # Context settings
    max_context_tokens: int = 4000

    # Generation settings
    temperature: float = 0.7

    # Filtering
    filter_authors: Optional[List[str]] = None
    show_sources: bool = False
```

#### Step 7: Update RAG Command to Support Hybrid Search
**File:** `bot/cogs/rag.py`

Add admin command to toggle hybrid search:

```python
@commands.command(name='ask_hybrid')
async def ask_hybrid(self, ctx, *, question: str):
    """
    Ask a question using hybrid search (BM25 + vector).

    Usage: !ask_hybrid What was decided about the database?
    """
    async with ctx.typing():
        self.logger.info(f"User {ctx.author} asked (hybrid): {question}")

        mentioned_users = []
        if ctx.message.mentions:
            mentioned_users = [user.display_name for user in ctx.message.mentions]

        config = RAGConfig(
            top_k=Config.RAG_DEFAULT_TOP_K,
            similarity_threshold=Config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
            max_context_tokens=Config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
            temperature=Config.RAG_DEFAULT_TEMPERATURE,
            strategy=Config.RAG_DEFAULT_STRATEGY,
            use_hybrid_search=True,  # Enable hybrid search
            bm25_weight=0.5,
            vector_weight=0.5,
            filter_authors=mentioned_users if mentioned_users else None,
        )

        result = await self.pipeline.answer_question(question, config)

        title = "💡 Answer (Hybrid Search)"
        if mentioned_users:
            authors_str = ", ".join(mentioned_users)
            title = f"💡 Answer (Hybrid - {authors_str})"

        embed = discord.Embed(
            title=title,
            description=result.answer,
            color=discord.Color.green()  # Different color for hybrid
        )

        sources_count = len(result.sources) if result.sources else 0
        embed.set_footer(
            text=f"Model: {result.model} | Cost: ${result.cost:.4f} | {sources_count} sources"
        )

        message = await ctx.send(embed=embed)
        await message.add_reaction("📚")

        self.bot._rag_cache = getattr(self.bot, '_rag_cache', {})
        self.bot._rag_cache[message.id] = result
```

---

## 🎯 Priority 2: Multi-Query Retrieval

### Why This?
- Handles vague/ambiguous queries
- 20-40% improvement in precision
- Easy to implement with existing LLM

### Implementation

**File:** `rag/query_enhancement.py`

```python
from typing import List
from ai.service import AIService
import logging

class QueryEnhancementService:
    """
    Enhances queries for better retrieval.

    Techniques:
    - Multi-query generation
    - Query expansion
    - HyDE (Hypothetical Document Embeddings)
    """

    def __init__(self, ai_service: Optional[AIService] = None):
        self.ai_service = ai_service or AIService()
        self.logger = logging.getLogger(__name__)

    async def generate_multi_queries(
        self,
        query: str,
        num_queries: int = 3
    ) -> List[str]:
        """
        Generate multiple variations of a query for improved retrieval.

        Args:
            query: Original user query
            num_queries: Number of query variations to generate

        Returns:
            List of query variations (including original)
        """
        prompt = f"""You are a helpful assistant that generates alternative phrasings of questions to improve search results.

Given the original question, generate {num_queries} alternative ways to ask the same question.
Each alternative should capture the same intent but use different words or phrasing.

Original Question: {query}

Generate {num_queries} alternative questions (one per line, no numbering):"""

        result = await self.ai_service.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )

        variations = [line.strip() for line in result['content'].strip().split('\n') if line.strip()]
        variations = [v.lstrip('0123456789.- ') for v in variations]  # Remove numbering
        variations = variations[:num_queries]

        # Include original query
        all_queries = [query] + variations

        self.logger.info(
            f"Generated {len(variations)} query variations for: {query[:50]}..."
        )
        for i, var in enumerate(variations, 1):
            self.logger.info(f"  Variation {i}: {var}")

        return all_queries

    async def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical answer to the query (HyDE technique).

        Instead of embedding the query, we:
        1. Generate a hypothetical answer
        2. Embed the answer
        3. Search for documents similar to the answer

        This works because answers are more similar to documents than queries.

        Args:
            query: User question

        Returns:
            Hypothetical answer
        """
        prompt = f"""You are a Discord conversation participant. Generate a hypothetical answer to this question as if you were responding in a Discord channel.

Make the answer:
- Conversational and natural (like Discord chat)
- Specific and detailed
- 2-3 sentences

Question: {query}

Hypothetical Answer:"""

        result = await self.ai_service.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.8
        )

        hyde_doc = result['content'].strip()

        self.logger.info(
            f"Generated HyDE document for query: {query[:50]}...\n"
            f"HyDE: {hyde_doc[:100]}..."
        )

        return hyde_doc
```

**Update RAG Pipeline:**

```python
# In rag/pipeline.py

from rag.query_enhancement import QueryEnhancementService
from rag.hybrid_search import reciprocal_rank_fusion

class RAGPipeline:
    def __init__(self, ...):
        # ... existing code ...
        self.query_enhancer = QueryEnhancementService(ai_service=self.ai_service)

    async def _retrieve_chunks(self, query: str, config: RAGConfig) -> List[Dict]:
        """Enhanced retrieval with multi-query support."""

        # Multi-query retrieval
        if config.use_multi_query:
            return await self._retrieve_multi_query(query, config)

        # Standard retrieval (existing code)
        # ...

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

        # Retrieve with each query
        all_results = []
        for q in queries:
            if config.use_hybrid_search:
                results = self.chunked_memory.search_hybrid(
                    query=q,
                    strategy=ChunkStrategy(config.strategy),
                    top_k=config.top_k * 2,  # Get more candidates
                    filter_authors=config.filter_authors
                )
            else:
                results = self.chunked_memory.search(
                    query=q,
                    strategy=ChunkStrategy(config.strategy),
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
```

**Update RAGConfig:**

```python
@dataclass
class RAGConfig:
    # ... existing fields ...

    # Multi-query settings
    use_multi_query: bool = False
    num_query_variations: int = 3

    # HyDE settings
    use_hyde: bool = False
```

---

## 🎯 Priority 3: Re-Ranking with Cross-Encoder

### Why This?
- 15-30% improvement in top-k precision
- Re-ranks initial results for better quality
- Minimal latency impact

### Implementation

**Install:**
```bash
pip install sentence-transformers
```

**File:** `rag/reranking.py`

```python
from typing import List, Dict
from sentence_transformers import CrossEncoder
import logging

class ReRankingService:
    """
    Re-ranks retrieval results using a cross-encoder model.

    Cross-encoders are more accurate than bi-encoders (used for initial retrieval)
    but slower, so we use them only for re-ranking top candidates.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Re-rank chunks using cross-encoder.

        Args:
            query: Search query
            chunks: Initial retrieval results
            top_k: Number of results to return

        Returns:
            Re-ranked chunks with cross-encoder scores
        """
        if not chunks:
            return []

        self.logger.info(f"Re-ranking {len(chunks)} chunks with cross-encoder")

        # Create query-document pairs
        pairs = [(query, chunk['content']) for chunk in chunks]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Add scores to chunks and sort
        for chunk, score in zip(chunks, scores):
            chunk['ce_score'] = float(score)
            chunk['original_rank'] = chunks.index(chunk) + 1

        # Sort by cross-encoder score
        reranked = sorted(chunks, key=lambda x: x['ce_score'], reverse=True)

        # Take top-k
        reranked = reranked[:top_k]

        # Log improvements
        for i, chunk in enumerate(reranked[:5], 1):
            original_rank = chunk.get('original_rank', '?')
            ce_score = chunk.get('ce_score', 0)
            original_score = chunk.get('similarity', chunk.get('rrf_score', 0))

            self.logger.info(
                f"  Rank {i} (was {original_rank}): "
                f"CE={ce_score:.3f}, Original={original_score:.3f}"
            )

        return reranked
```

**Update RAG Pipeline:**

```python
from rag.reranking import ReRankingService

class RAGPipeline:
    def __init__(self, ...):
        # ... existing code ...
        self.reranker = None  # Lazy load

    async def _retrieve_chunks(self, query: str, config: RAGConfig) -> List[Dict]:
        """Enhanced retrieval with re-ranking."""

        # ... existing retrieval code ...

        # Re-rank if enabled
        if config.use_reranking and chunks:
            if self.reranker is None:
                self.reranker = ReRankingService()

            chunks = self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=config.top_k
            )

        return chunks
```

**Update RAGConfig:**

```python
@dataclass
class RAGConfig:
    # ... existing fields ...

    # Re-ranking settings
    use_reranking: bool = False
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

---

## 🎯 Priority 4: Chatbot Command

**File:** `bot/cogs/chat.py`

```python
import discord
from discord.ext import commands
import logging
from rag.pipeline import RAGPipeline
from rag.models import RAGConfig
from config import Config
from typing import Dict
from datetime import datetime, timedelta

class ConversationHistory:
    """Manages conversation history for a user."""

    def __init__(self, max_turns: int = 10, timeout_minutes: int = 30):
        self.max_turns = max_turns
        self.timeout_minutes = timeout_minutes
        self.messages = []
        self.last_activity = datetime.now()

    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        self.last_activity = datetime.now()
        self._trim()

    def get_history(self, format="text") -> str:
        """Get conversation history as text."""
        if format == "text":
            lines = []
            for msg in self.messages:
                role = "You" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            return "\n".join(lines)
        return self.messages

    def is_expired(self) -> bool:
        """Check if conversation has timed out."""
        elapsed = datetime.now() - self.last_activity
        return elapsed > timedelta(minutes=self.timeout_minutes)

    def clear(self):
        """Clear conversation history."""
        self.messages = []
        self.last_activity = datetime.now()

    def _trim(self):
        """Keep only last max_turns messages."""
        if len(self.messages) > self.max_turns:
            self.messages = self.messages[-self.max_turns:]


class ChatBot(commands.Cog):
    """Conversational RAG chatbot."""

    def __init__(self, bot):
        self.bot = bot
        self.pipeline = RAGPipeline()
        self.conversations: Dict[int, ConversationHistory] = {}
        self.logger = logging.getLogger(__name__)

    @commands.command(name='chat')
    async def chat(self, ctx, *, message: str):
        """
        Multi-turn conversational RAG chatbot.

        Maintains conversation context for follow-up questions.

        Usage:
            !chat What was decided about the database?
            !chat What were the alternatives?
            !chat Why did we choose that one?
        """
        async with ctx.typing():
            user_id = ctx.author.id

            # Get or create conversation
            if user_id not in self.conversations:
                self.conversations[user_id] = ConversationHistory()
            elif self.conversations[user_id].is_expired():
                self.logger.info(f"Conversation expired for user {user_id}, starting new session")
                self.conversations[user_id].clear()

            conversation = self.conversations[user_id]

            # Add user message
            conversation.add_message("user", message)

            # Contextualize query if there's history
            if len(conversation.messages) > 1:
                contextualized_query = await self._contextualize_query(
                    message,
                    conversation.get_history()
                )
                self.logger.info(
                    f"Contextualized query: '{message}' → '{contextualized_query}'"
                )
            else:
                contextualized_query = message

            # Retrieve and answer
            config = RAGConfig(
                top_k=15,
                use_hybrid_search=True,
                use_multi_query=True,
                use_reranking=True,
            )

            result = await self.pipeline.answer_question(contextualized_query, config)

            # Add assistant response
            conversation.add_message("assistant", result.answer)

            # Format response
            embed = discord.Embed(
                title="💬 Chat Response",
                description=result.answer,
                color=discord.Color.purple()
            )

            # Show conversation info
            turn_num = len([m for m in conversation.messages if m['role'] == 'user'])
            embed.set_footer(
                text=f"Turn {turn_num} | {len(result.sources)} sources | ${result.cost:.4f}"
            )

            message = await ctx.send(embed=embed)
            await message.add_reaction("📚")
            await message.add_reaction("🔄")  # Regenerate
            await message.add_reaction("🗑️")  # Clear history

            # Cache result
            self.bot._rag_cache = getattr(self.bot, '_rag_cache', {})
            self.bot._rag_cache[message.id] = result

    @commands.command(name='chat_clear')
    async def chat_clear(self, ctx):
        """Clear your conversation history."""
        user_id = ctx.author.id
        if user_id in self.conversations:
            self.conversations[user_id].clear()
            await ctx.send("✅ Conversation history cleared!")
        else:
            await ctx.send("ℹ️ No active conversation to clear.")

    @commands.command(name='chat_history')
    async def chat_history(self, ctx):
        """Show your conversation history."""
        user_id = ctx.author.id

        if user_id not in self.conversations or not self.conversations[user_id].messages:
            await ctx.send("ℹ️ No conversation history.")
            return

        conversation = self.conversations[user_id]
        history = conversation.get_history()

        embed = discord.Embed(
            title="💬 Conversation History",
            description=history[:4000],  # Discord embed limit
            color=discord.Color.blue()
        )

        turn_count = len([m for m in conversation.messages if m['role'] == 'user'])
        elapsed = datetime.now() - conversation.last_activity
        minutes_ago = int(elapsed.total_seconds() / 60)

        embed.set_footer(
            text=f"{turn_count} turns | Last activity: {minutes_ago}m ago"
        )

        await ctx.send(embed=embed)

    async def _contextualize_query(
        self,
        current_query: str,
        conversation_history: str
    ) -> str:
        """
        Rewrite query to include conversation context.

        Example:
        History:
        - You: "What database did we choose?"
        - Assistant: "PostgreSQL was chosen."
        - You: "Why?"

        Contextualized: "Why was PostgreSQL chosen as the database?"
        """
        prompt = f"""Given this conversation history, rewrite the user's latest question to be standalone and self-contained.

Conversation History:
{conversation_history}

Current Question: {current_query}

Rewrite the current question to include necessary context from the conversation history, making it a standalone question that can be understood without the history.

Standalone Question:"""

        result = await self.pipeline.ai_service.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.3
        )

        return result['content'].strip()

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        """Handle chat reactions (clear history, regenerate)."""
        if user.bot:
            return

        # Clear history
        if str(reaction.emoji) == "🗑️":
            user_id = user.id
            if user_id in self.conversations:
                self.conversations[user_id].clear()
                await reaction.message.channel.send(
                    f"✅ {user.mention} Conversation history cleared!"
                )


async def setup(bot):
    await bot.add_cog(ChatBot(bot))
```

---

## 📝 Testing Plan

### Test Queries
Create `tests/test_queries.txt`:

```
# Exact keyword matches (BM25 should excel)
What did Alice say about PostgreSQL?
Show me messages mentioning "database schema"

# Semantic queries (vector search should excel)
What were the main technical decisions?
How did the team feel about the new feature?

# Vague queries (multi-query should help)
What was decided?
Tell me about the backend

# Follow-up questions (chatbot context)
What database did we choose?
Why did we choose it?
What were the alternatives?
```

### Evaluation Script
**File:** `scripts/evaluate_retrieval.py`

```python
import asyncio
from rag.pipeline import RAGPipeline
from rag.models import RAGConfig

async def evaluate():
    pipeline = RAGPipeline()

    with open('tests/test_queries.txt') as f:
        queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    results = []

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        # Test different configurations
        configs = {
            "baseline": RAGConfig(),
            "hybrid": RAGConfig(use_hybrid_search=True),
            "multi-query": RAGConfig(use_multi_query=True),
            "full": RAGConfig(
                use_hybrid_search=True,
                use_multi_query=True,
                use_reranking=True
            )
        }

        for name, config in configs.items():
            result = await pipeline.answer_question(query, config)
            print(f"\n[{name.upper()}]")
            print(f"Sources: {len(result.sources)}")
            if result.sources:
                print(f"Top similarity: {result.sources[0].get('similarity', 0):.3f}")
            print(f"Answer: {result.answer[:200]}...")

            results.append({
                "query": query,
                "config": name,
                "num_sources": len(result.sources),
                "answer": result.answer
            })

    # Save results
    import json
    with open('tests/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation complete! Results saved to tests/evaluation_results.json")

if __name__ == "__main__":
    asyncio.run(evaluate())
```

---

## 📊 Configuration Updates

**File:** `config.py`

Add new configuration options:

```python
# RAG Advanced Features
RAG_USE_HYBRID_SEARCH: bool = os.getenv("RAG_USE_HYBRID_SEARCH", "True").lower() == "true"
RAG_USE_MULTI_QUERY: bool = os.getenv("RAG_USE_MULTI_QUERY", "False").lower() == "true"
RAG_USE_RERANKING: bool = os.getenv("RAG_USE_RERANKING", "False").lower() == "true"

# Hybrid search weights
RAG_BM25_WEIGHT: float = float(os.getenv("RAG_BM25_WEIGHT", "0.5"))
RAG_VECTOR_WEIGHT: float = float(os.getenv("RAG_VECTOR_WEIGHT", "0.5"))

# Multi-query settings
RAG_NUM_QUERY_VARIATIONS: int = int(os.getenv("RAG_NUM_QUERY_VARIATIONS", "3"))

# Chat settings
CHAT_MAX_HISTORY_TURNS: int = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
CHAT_SESSION_TIMEOUT_MINUTES: int = int(os.getenv("CHAT_SESSION_TIMEOUT_MINUTES", "30"))
```

---

## ✅ Checklist

### Phase 1: Setup
- [ ] Install `rank-bm25` package
- [ ] Create `rag/hybrid_search.py`
- [ ] Create `rag/query_enhancement.py`
- [ ] Create `rag/reranking.py`
- [ ] Update `requirements.txt`

### Phase 2: Hybrid Search
- [ ] Add `search_bm25()` to `ChunkedMemoryService`
- [ ] Add `search_hybrid()` to `ChunkedMemoryService`
- [ ] Add `get_all_documents()` to ChromaDB provider
- [ ] Update `RAGPipeline._retrieve_chunks()`
- [ ] Update `RAGConfig` with hybrid settings
- [ ] Add `!ask_hybrid` command

### Phase 3: Multi-Query
- [ ] Implement `QueryEnhancementService`
- [ ] Add `_retrieve_multi_query()` to pipeline
- [ ] Update `RAGConfig` with multi-query settings
- [ ] Test multi-query retrieval

### Phase 4: Re-Ranking
- [ ] Implement `ReRankingService`
- [ ] Integrate re-ranking into pipeline
- [ ] Test re-ranking improvements

### Phase 5: Chatbot
- [ ] Create `bot/cogs/chat.py`
- [ ] Implement `ConversationHistory` class
- [ ] Add `!chat` command
- [ ] Add `!chat_clear` and `!chat_history` commands
- [ ] Test conversation context

### Phase 6: Testing & Evaluation
- [ ] Create test queries
- [ ] Run evaluation script
- [ ] Compare baseline vs enhanced retrieval
- [ ] Measure improvements

---

## 🎯 Expected Results

### Baseline (Current)
- Top-k results: ~3-5 relevant chunks
- Many queries return 0 results
- Similarity scores: 0.35-0.50 average

### With Hybrid Search
- Top-k results: ~7-9 relevant chunks
- Fewer zero-result queries
- Better keyword + semantic matching

### With Multi-Query
- Handles vague queries much better
- More diverse results
- Higher recall

### With Re-Ranking
- Top 3 results highly relevant
- Better precision
- Cross-encoder scores >0.8 for good matches

### With Chatbot
- Handles follow-up questions
- Maintains context across turns
- Natural conversation flow

---

## 🚀 Next Steps After Implementation

1. **Collect metrics** - Track retrieval quality
2. **A/B test** - Compare configurations
3. **User feedback** - Add thumbs up/down
4. **Optimize** - Tune weights and thresholds
5. **Scale** - Consider caching, better embeddings

---

Good luck! 🎉
