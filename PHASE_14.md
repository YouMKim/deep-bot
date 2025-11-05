# Phase 14: Hybrid Search (Vector + Keyword)

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Hybrid Search - Combining Dense (Vector) and Sparse (Keyword) Retrieval

### Learning Objectives
- Understand dense vs sparse retrieval
- Learn BM25 algorithm for keyword search
- Practice combining multiple retrieval methods
- Design reciprocal rank fusion (RRF)
- Compare hybrid vs pure vector search

### Why Hybrid Search?

**Problem with Vector Search Alone:**
- Misses exact keyword matches (e.g., "API v2.0" vs "API version 2")
- Poor for rare terms (proper nouns, technical jargon)
- Semantic similarity can be too broad

**Problem with Keyword Search Alone:**
- Misses semantic similarities ("car" won't match "automobile")
- Sensitive to exact wording
- No understanding of context

**Solution: Hybrid Search**
- Combines strengths of both approaches
- Vector search finds semantic matches
- Keyword search finds exact/rare terms
- Results are fused using ranking algorithms

---

## Implementation Steps

### Step 14.1: Implement BM25 Keyword Search

Create `services/bm25_retriever.py`:

```python
"""
BM25 (Best Matching 25) - Industry-standard keyword search algorithm.

Learning: BM25 is a probabilistic ranking function used by search engines.
It considers:
- Term frequency (TF): How often term appears in document
- Inverse document frequency (IDF): How rare the term is across corpus
- Document length normalization: Avoid bias toward long documents
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import logging

class BM25Retriever:
    """
    Keyword-based retrieval using BM25.

    Learning: BM25 is what powers traditional search engines.
    It's fast, interpretable, and excellent for exact matches.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bm25 = None
        self.documents = []
        self.doc_ids = []

    def index_documents(self, documents: List[Dict]):
        """
        Index documents for BM25 search.

        Args:
            documents: List of {
                "id": "chunk_id",
                "content": "text content",
                "metadata": {...}
            }

        Learning: BM25 requires preprocessing:
        1. Tokenize documents (split into words)
        2. Build inverted index (term -> document mapping)
        3. Calculate IDF scores for each term
        """
        if not documents:
            self.logger.warning("No documents to index")
            return

        self.documents = documents
        self.doc_ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]

        # Tokenize documents (simple whitespace + lowercase)
        # Production: Use better tokenizer (nltk, spaCy, etc.)
        tokenized_docs = [
            doc.get("content", "").lower().split()
            for doc in documents
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        self.logger.info(f"Indexed {len(documents)} documents with BM25")

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score, document) tuples

        Learning: BM25 scoring:
        - Higher scores = better matches
        - Exact keyword matches rank high
        - Rare terms contribute more to score
        """
        if not self.bm25:
            self.logger.error("BM25 not initialized. Call index_documents() first.")
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k document indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Return results with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return docs with positive scores
                results.append((
                    self.doc_ids[idx],
                    float(scores[idx]),
                    self.documents[idx]
                ))

        return results

    def get_document_stats(self) -> Dict:
        """Get statistics about indexed corpus"""
        if not self.bm25:
            return {"indexed": False}

        return {
            "indexed": True,
            "num_documents": len(self.documents),
            "avg_doc_length": self.bm25.avgdl,
            "total_terms": len(self.bm25.idf)
        }
```

### Step 14.2: Implement Reciprocal Rank Fusion (RRF)

Create `services/hybrid_search_service.py`:

```python
"""
Hybrid Search - Combines vector and keyword search.

Learning: Combining rankings is tricky!
- Can't just add scores (different scales)
- Need ranking-based fusion
- Reciprocal Rank Fusion (RRF) is simple and effective
"""

from services.chunked_memory_service import ChunkedMemoryService
from services.bm25_retriever import BM25Retriever
from typing import List, Dict, Tuple
import logging

class HybridSearchService:
    """
    Combines vector search and BM25 keyword search.

    Learning: Hybrid search improves both precision and recall.
    Vector search: High recall (finds similar content)
    Keyword search: High precision (finds exact matches)
    """

    def __init__(
        self,
        chunked_memory: ChunkedMemoryService,
        bm25_retriever: BM25Retriever
    ):
        self.vector_search = chunked_memory
        self.keyword_search = bm25_retriever
        self.logger = logging.getLogger(__name__)

    def search(
        self,
        query: str,
        strategy: str = "token_aware",
        top_k: int = 10,
        alpha: float = 0.5,  # Weight: 0=keyword only, 1=vector only
        rrf_k: int = 60  # RRF constant (typical value: 60)
    ) -> List[Dict]:
        """
        Hybrid search using RRF fusion.

        Args:
            query: Search query
            strategy: Chunking strategy for vector search
            top_k: Final number of results
            alpha: Fusion weight (0-1)
            rrf_k: RRF constant (larger = less weight on rank)

        Returns:
            Fused and ranked results

        Learning: Reciprocal Rank Fusion (RRF) formula:
        RRF_score = Σ 1 / (k + rank)

        Why RRF?
        - Scale-independent (works with different scoring functions)
        - Simple and effective
        - No parameter tuning needed
        - Used by major search engines
        """
        # Step 1: Vector search
        vector_results = self.vector_search.search(
            query=query,
            strategy=strategy,
            top_k=top_k * 2  # Get more for fusion
        )

        # Step 2: Keyword search
        keyword_results = self.keyword_search.search(
            query=query,
            top_k=top_k * 2
        )

        # Step 3: Fuse using RRF
        fused_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            alpha=alpha,
            k=rrf_k
        )

        # Step 4: Return top-k
        return fused_results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        keyword_results: List[Tuple[str, float, Dict]],
        alpha: float,
        k: int
    ) -> List[Dict]:
        """
        Fuse results using Reciprocal Rank Fusion.

        Learning: RRF combines multiple rankings by:
        1. For each result, calculate 1/(k + rank)
        2. Sum scores across all ranking sources
        3. Sort by final score

        Why reciprocal rank?
        - Rank 1 gets 1/(k+1) = ~1/61 ≈ 0.016
        - Rank 10 gets 1/(k+10) = ~1/70 ≈ 0.014
        - Diminishing returns for lower ranks
        """
        scores = {}  # doc_id -> RRF score
        doc_map = {}  # doc_id -> full document

        # Add vector search results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.get("metadata", {}).get("first_message_id", f"vec_{rank}")
            rrf_score = alpha / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = result

        # Add keyword search results
        for rank, (doc_id, bm25_score, doc) in enumerate(keyword_results, start=1):
            rrf_score = (1 - alpha) / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_map:
                # Convert BM25 doc to vector search format
                doc_map[doc_id] = {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "similarity": 0,  # No vector similarity
                    "bm25_score": bm25_score
                }

        # Sort by RRF score
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build final results
        results = []
        for doc_id, rrf_score in sorted_docs:
            doc = doc_map[doc_id]
            doc["rrf_score"] = rrf_score
            doc["fusion_method"] = "RRF"
            results.append(doc)

        return results

    def compare_methods(
        self,
        query: str,
        strategy: str = "token_aware",
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Compare vector, keyword, and hybrid search side-by-side.

        Returns:
            {
                "vector": [results],
                "keyword": [results],
                "hybrid": [results]
            }

        Learning: This lets you see what each method retrieves!
        """
        # Vector only
        vector_results = self.vector_search.search(query, strategy, top_k)

        # Keyword only
        keyword_results = self.keyword_search.search(query, top_k)
        keyword_formatted = [
            {
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "bm25_score": score
            }
            for _, score, doc in keyword_results
        ]

        # Hybrid (50/50 fusion)
        hybrid_results = self.search(query, strategy, top_k, alpha=0.5)

        return {
            "vector": vector_results,
            "keyword": keyword_formatted,
            "hybrid": hybrid_results
        }
```

### Step 14.3: Add Hybrid Search Commands

Add to `cogs/chatbot.py`:

```python
@commands.command(name='hybrid_search')
async def hybrid_search(self, ctx, *, query: str):
    """
    Search using hybrid (vector + keyword) retrieval.

    Usage: !hybrid_search What is the best database?

    Learning: Compare with !ask to see the difference!
    """
    try:
        async with ctx.typing():
            # Initialize services
            from services.hybrid_search_service import HybridSearchService
            from services.bm25_retriever import BM25Retriever

            # Get documents for BM25
            # TODO: Load from your chunked memory
            # For now, simplified example

            bm25 = BM25Retriever()
            # bm25.index_documents(your_chunks)

            hybrid_service = HybridSearchService(
                self.conversational_rag.rag.memory,
                bm25
            )

            # Search
            results = hybrid_service.search(query, top_k=5, alpha=0.5)

            # Display results
            embed = discord.Embed(
                title="🔍 Hybrid Search Results",
                description=f"Query: {query}",
                color=discord.Color.purple()
            )

            for i, result in enumerate(results[:3], 1):
                embed.add_field(
                    name=f"Result {i} (RRF: {result.get('rrf_score', 0):.3f})",
                    value=result.get("content", "")[:200] + "...",
                    inline=False
                )

            await ctx.send(embed=embed)

    except Exception as e:
        self.logger.error(f"Error in hybrid_search: {e}", exc_info=True)
        await ctx.send(f"❌ Error: {e}")

@commands.command(name='compare_search')
async def compare_search(self, ctx, *, query: str):
    """
    Compare vector, keyword, and hybrid search side-by-side.

    Usage: !compare_search MongoDB performance

    Learning: See how different methods retrieve different results!
    """
    try:
        async with ctx.typing():
            from services.hybrid_search_service import HybridSearchService
            from services.bm25_retriever import BM25Retriever

            bm25 = BM25Retriever()
            # bm25.index_documents(your_chunks)

            hybrid_service = HybridSearchService(
                self.conversational_rag.rag.memory,
                bm25
            )

            # Compare all methods
            comparison = hybrid_service.compare_methods(query, top_k=3)

            # Create comparison embed
            embed = discord.Embed(
                title="⚖️ Search Method Comparison",
                description=f"Query: \"{query}\"",
                color=discord.Color.gold()
            )

            # Vector results
            vector_preview = "\n".join([
                f"{i}. {r.get('content', '')[:50]}... "
                f"(sim: {r.get('similarity', 0):.2f})"
                for i, r in enumerate(comparison["vector"][:3], 1)
            ])
            embed.add_field(
                name="🔮 Vector Search",
                value=vector_preview or "No results",
                inline=False
            )

            # Keyword results
            keyword_preview = "\n".join([
                f"{i}. {r.get('content', '')[:50]}... "
                f"(BM25: {r.get('bm25_score', 0):.2f})"
                for i, r in enumerate(comparison["keyword"][:3], 1)
            ])
            embed.add_field(
                name="🔑 Keyword Search (BM25)",
                value=keyword_preview or "No results",
                inline=False
            )

            # Hybrid results
            hybrid_preview = "\n".join([
                f"{i}. {r.get('content', '')[:50]}... "
                f"(RRF: {r.get('rrf_score', 0):.3f})"
                for i, r in enumerate(comparison["hybrid"][:3], 1)
            ])
            embed.add_field(
                name="⚡ Hybrid (RRF Fusion)",
                value=hybrid_preview or "No results",
                inline=False
            )

            await ctx.send(embed=embed)

    except Exception as e:
        self.logger.error(f"Error in compare_search: {e}", exc_info=True)
        await ctx.send(f"❌ Error: {e}")
```

---

## Usage Examples

### Hybrid Search

```
!hybrid_search What is the MongoDB performance like?

Bot: 🔍 Hybrid Search Results
     Query: What is the MongoDB performance like?

     Result 1 (RRF: 0.042)
     Bob: MongoDB performs really well for our use case, handling
     10K writes/sec...

     Result 2 (RRF: 0.035)
     Alice: We benchmarked MongoDB vs PostgreSQL. MongoDB: 5ms latency...

     Result 3 (RRF: 0.028)
     Charlie: Performance depends on schema design. MongoDB shines with...
```

### Compare Search Methods

```
!compare_search API v2.0

Bot: ⚖️ Search Method Comparison

     🔮 Vector Search:
     1. Discussion about API design patterns... (sim: 0.82)
     2. REST API best practices discussion... (sim: 0.78)
     3. Microservices API architecture... (sim: 0.75)

     🔑 Keyword Search (BM25):
     1. Bob: We released API v2.0 yesterday... (BM25: 12.4)
     2. API v2.0 has breaking changes in auth... (BM25: 11.8)
     3. Migrating from API v1.5 to v2.0... (BM25: 10.2)

     ⚡ Hybrid (RRF Fusion):
     1. Bob: We released API v2.0 yesterday... (RRF: 0.045)
     2. API v2.0 has breaking changes in auth... (RRF: 0.038)
     3. Discussion about API design patterns... (RRF: 0.032)
```

**Notice:**
- Vector search finds semantically related content
- Keyword search finds exact "API v2.0" mentions
- Hybrid combines both, ranking exact matches higher!

---

## When to Use Each Method

### Vector Search Best For:
- ✅ Conceptual questions ("How do we handle errors?")
- ✅ Semantic similarity ("car" matches "automobile")
- ✅ Natural language queries
- ❌ Poor for exact terms ("API v2.0", rare names)

### Keyword Search (BM25) Best For:
- ✅ Exact terms (product names, version numbers)
- ✅ Rare/technical jargon
- ✅ Proper nouns (people, places)
- ❌ Misses semantic similarity

### Hybrid Search Best For:
- ✅ **General purpose** (combines strengths)
- ✅ Unknown query types
- ✅ Production systems
- ⚠️ Slightly higher latency

---

## Performance Considerations

### BM25 Indexing
- **Time**: O(N * M) where N=docs, M=avg doc length
- **Space**: O(V * D) where V=vocabulary size, D=num docs
- **Fast**: 1000 docs ~ 10ms indexing

### Search Performance

| Method | Latency | When to Use |
|--------|---------|-------------|
| Vector | 50-200ms | Semantic queries |
| BM25 | 5-20ms | Exact matches |
| Hybrid | 70-250ms | Best overall quality |

### Optimization Tips
1. **Cache BM25 index** (only rebuild when docs change)
2. **Parallel retrieval** (run vector + BM25 concurrently)
3. **Early termination** (stop BM25 after top-k)
4. **Index pruning** (remove stop words)

---

## Advanced: Tuning Alpha Parameter

The `alpha` parameter controls the fusion weight:
- `alpha = 0.0`: 100% keyword (BM25)
- `alpha = 0.5`: 50/50 hybrid
- `alpha = 1.0`: 100% vector

**How to find optimal alpha:**

```python
# Test different alpha values
test_queries = [
    "What is MongoDB performance?",
    "API v2.0 breaking changes",
    "How to handle authentication?"
]

for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    results = hybrid_service.search(query, alpha=alpha)
    # Evaluate quality (manual or using metrics from Phase 6.5)
    print(f"Alpha {alpha}: {evaluate_results(results)}")
```

**General guidelines:**
- **Technical docs**: alpha = 0.3-0.4 (favor keywords)
- **Conversations**: alpha = 0.6-0.7 (favor vectors)
- **Mixed content**: alpha = 0.5 (balanced)

---

## Common Pitfalls - Phase 14

1. **Not normalizing scores**: BM25 and vector scores on different scales
2. **Forgetting to rebuild BM25 index**: After adding new documents
3. **Over-weighting one method**: Test different alpha values
4. **Poor tokenization**: Use proper tokenizer (nltk, spaCy) for production
5. **Missing stopword removal**: Can improve BM25 quality

## Debugging Tips - Phase 14

- **Check BM25 scores**: Are they reasonable? (typical range: 0-30)
- **Inspect top terms**: What terms have highest IDF?
- **Compare overlaps**: How many results appear in both methods?
- **Visualize rankings**: Plot rank vs score for each method

---

## Next Steps

This gives you **hybrid search**:
- ✅ Vector search for semantic similarity
- ✅ Keyword search for exact matches
- ✅ RRF fusion for best of both
- ✅ Comparison tools to see differences

Next: **Phase 15 - Reranking & Query Optimization** 🚀

Where we'll learn to:
- Use cross-encoders to rerank results
- Expand queries for better recall
- Optimize retrieval quality
