# Phase 15: Reranking & Query Optimization

## Overview

In **Phase 14**, we learned how to combine vector and keyword search using hybrid search. While this improves recall (finding more relevant documents), we can further improve **precision** (ranking the most relevant documents first) using reranking and query optimization techniques.

**Learning Objectives:**
- Understand the difference between bi-encoders (retrieval) and cross-encoders (reranking)
- Implement cross-encoder reranking for improved relevance
- Learn query expansion techniques to handle vocabulary mismatch
- Build a query optimization pipeline
- Compare reranked vs non-reranked results

**Prerequisites:** Phase 14 (Hybrid Search)

**Estimated Time:** 4-6 hours

---

## Table of Contents

1. [Understanding Reranking](#1-understanding-reranking)
2. [Cross-Encoder Reranking](#2-cross-encoder-reranking)
3. [Query Expansion](#3-query-expansion)
4. [Query Optimization Pipeline](#4-query-optimization-pipeline)
5. [Discord Commands](#5-discord-commands)
6. [Testing & Comparison](#6-testing--comparison)
7. [Performance Considerations](#7-performance-considerations)

---

## 1. Understanding Reranking

### Why Reranking?

**Two-Stage Retrieval Pipeline:**
```
Stage 1: Retrieval (Fast, Broad)
└─ Use bi-encoders to quickly get top-100 candidates
└─ Encode query once, compare with pre-computed doc embeddings
└─ Speed: ~100ms for millions of docs

Stage 2: Reranking (Slow, Precise)
└─ Use cross-encoders to rerank top-100 → top-10
└─ Compute joint query-document score
└─ Speed: ~500ms for 100 docs
```

### Bi-Encoder vs Cross-Encoder

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|------------|---------------|
| **Architecture** | Separate encoders for query & doc | Joint encoding of [query, doc] |
| **Speed** | Very fast (pre-computed embeddings) | Slow (compute on-the-fly) |
| **Accuracy** | Good | Excellent |
| **Use Case** | Retrieval (Stage 1) | Reranking (Stage 2) |
| **Scalability** | Millions of docs | Hundreds of docs |

**Analogy:**
- **Bi-encoder**: Fast filter - "Find all books about cooking" (broad search)
- **Cross-encoder**: Careful review - "Which of these 10 books best answers 'how to make sourdough bread?'" (precise ranking)

---

## 2. Cross-Encoder Reranking

### Implementation

Create `services/reranking_service.py`:

```python
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RerankingService:
    """Cross-encoder reranking for improved relevance."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder model.

        Popular models:
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good)
        - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better)
        - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual)
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded successfully")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            List of (document, rerank_score) tuples, sorted by score
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc.get("content", "")] for doc in documents]

        # Compute cross-encoder scores
        scores = self.model.predict(pairs)

        # Combine documents with scores
        doc_scores = list(zip(documents, scores))

        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return doc_scores[:top_k]

    def rerank_with_comparison(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> Dict:
        """
        Rerank and compare with original ranking.

        Returns:
            {
                "original_ranking": [...],
                "reranked": [...],
                "changes": [...]
            }
        """
        # Store original order
        original_ranking = [
            {
                "doc": doc,
                "original_score": doc.get("score", 0.0),
                "original_rank": idx + 1
            }
            for idx, doc in enumerate(documents[:top_k])
        ]

        # Rerank
        reranked = self.rerank(query, documents, top_k)

        # Track position changes
        changes = []
        for new_rank, (doc, new_score) in enumerate(reranked, 1):
            # Find original rank
            doc_id = doc.get("id")
            original_rank = next(
                (item["original_rank"] for item in original_ranking
                 if item["doc"].get("id") == doc_id),
                None
            )

            if original_rank:
                rank_change = original_rank - new_rank
                if rank_change != 0:
                    changes.append({
                        "doc_id": doc_id,
                        "original_rank": original_rank,
                        "new_rank": new_rank,
                        "change": rank_change,
                        "direction": "↑" if rank_change > 0 else "↓"
                    })

        return {
            "original_ranking": original_ranking,
            "reranked": reranked,
            "changes": changes
        }
```

---

## 3. Query Expansion

### Why Query Expansion?

**Problem: Vocabulary Mismatch**
- User query: "how to fix a broken pipe"
- Relevant doc contains: "repairing plumbing leaks"
- Semantic search might miss this due to different vocabulary

**Solution: Query Expansion**
Generate multiple query variations to capture different phrasings.

### Implementation

Add to `services/query_optimizer.py`:

```python
from typing import List, Set
import re
from openai import OpenAI
import config

class QueryOptimizer:
    """Query optimization and expansion techniques."""

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def expand_query_llm(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Use LLM to generate query variations.

        Example:
            Input: "how to make bread"
            Output: [
                "how to make bread",
                "bread baking instructions",
                "homemade bread recipe",
                "steps for baking bread"
            ]
        """
        prompt = f"""Given this search query, generate {num_variations} alternative phrasings that mean the same thing.

Original query: "{query}"

Generate variations that:
- Use different vocabulary (synonyms)
- Are more specific or more general
- Cover different ways people might phrase this question

Return only the variations, one per line, without numbering."""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )

        variations = response.choices[0].message.content.strip().split("\n")
        variations = [v.strip() for v in variations if v.strip()]

        # Always include original query
        return [query] + variations[:num_variations]

    def expand_query_keywords(self, query: str) -> List[str]:
        """
        Expand query by extracting and combining keywords.

        Example:
            Input: "best python web frameworks 2024"
            Output: [
                "best python web frameworks 2024",
                "python web frameworks",
                "python frameworks 2024",
                "web frameworks"
            ]
        """
        # Remove common stopwords
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}

        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords]

        variations = [query]  # Original query

        # Add full keyword combination
        if len(keywords) > 1:
            variations.append(" ".join(keywords))

        # Add partial combinations (for queries with 3+ keywords)
        if len(keywords) >= 3:
            # First half + last half
            mid = len(keywords) // 2
            variations.append(" ".join(keywords[:mid]))
            variations.append(" ".join(keywords[mid:]))

        return list(set(variations))  # Remove duplicates

    def optimize_query(self, query: str) -> str:
        """
        Clean and optimize a single query.

        - Remove extra whitespace
        - Fix common typos (basic)
        - Normalize punctuation
        """
        # Remove extra whitespace
        query = " ".join(query.split())

        # Remove trailing punctuation (except ?)
        if query.endswith((".", ",", ";", ":")):
            query = query[:-1]

        # Normalize quotes
        query = query.replace(""", '"').replace(""", '"')
        query = query.replace("'", "'").replace("'", "'")

        return query.strip()

    def merge_results(
        self,
        results_list: List[List[Dict]],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Merge results from multiple query variations.

        Uses RRF (Reciprocal Rank Fusion) to combine rankings.
        """
        from collections import defaultdict

        # RRF scores
        doc_scores = defaultdict(float)
        doc_map = {}
        k = 60  # RRF constant

        for results in results_list:
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get("id")
                doc_scores[doc_id] += 1 / (k + rank)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k with scores
        return [
            {**doc_map[doc_id], "fusion_score": score}
            for doc_id, score in sorted_docs[:top_k]
        ]
```

---

## 4. Query Optimization Pipeline

### Complete Pipeline

Create `services/advanced_rag_service.py`:

```python
from typing import List, Dict
from rag.query_optimizer import QueryOptimizer
from retrieval.reranking import RerankingService
from retrieval.hybrid import HybridSearchService
import logging

logger = logging.getLogger(__name__)

class AdvancedRAGService:
    """
    Advanced RAG pipeline with query optimization and reranking.

    Pipeline:
    1. Query Optimization (clean query)
    2. Query Expansion (generate variations)
    3. Hybrid Search (retrieve candidates for each variation)
    4. Result Fusion (merge results from all variations)
    5. Reranking (cross-encoder rerank top results)
    6. Final Selection (top-k most relevant)
    """

    def __init__(self):
        self.query_optimizer = QueryOptimizer()
        self.hybrid_search = HybridSearchService()
        self.reranker = RerankingService()

    async def search_with_optimization(
        self,
        query: str,
        expand_query: bool = True,
        rerank: bool = True,
        top_k: int = 10
    ) -> Dict:
        """
        Complete advanced RAG pipeline.

        Returns:
            {
                "original_query": str,
                "optimized_query": str,
                "query_variations": List[str],
                "results": List[Dict],
                "pipeline_stats": Dict
            }
        """
        stats = {
            "original_query": query,
            "steps": []
        }

        # Step 1: Optimize query
        optimized_query = self.query_optimizer.optimize_query(query)
        stats["optimized_query"] = optimized_query
        stats["steps"].append("Query optimization")

        # Step 2: Expand query (optional)
        if expand_query:
            query_variations = self.query_optimizer.expand_query_llm(
                optimized_query,
                num_variations=2
            )
            stats["query_variations"] = query_variations
            stats["steps"].append(f"Query expansion ({len(query_variations)} variations)")
        else:
            query_variations = [optimized_query]

        # Step 3: Hybrid search for each variation
        all_results = []
        for variation in query_variations:
            results = await self.hybrid_search.search(
                variation,
                top_k=50,  # Retrieve more candidates for reranking
                alpha=0.5
            )
            all_results.append(results)

        stats["steps"].append(f"Hybrid search ({len(all_results)} queries)")

        # Step 4: Merge results using RRF
        merged_results = self.query_optimizer.merge_results(
            all_results,
            top_k=50
        )
        stats["steps"].append(f"Result fusion ({len(merged_results)} candidates)")

        # Step 5: Rerank (optional)
        if rerank:
            reranked = self.reranker.rerank(
                optimized_query,
                merged_results,
                top_k=top_k
            )
            final_results = [
                {**doc, "rerank_score": score}
                for doc, score in reranked
            ]
            stats["steps"].append(f"Cross-encoder reranking (top-{top_k})")
        else:
            final_results = merged_results[:top_k]

        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "query_variations": query_variations if expand_query else None,
            "results": final_results,
            "pipeline_stats": stats
        }

    async def compare_pipelines(self, query: str, top_k: int = 5) -> Dict:
        """
        Compare different pipeline configurations.

        Compares:
        1. Basic search (no optimization, no reranking)
        2. With optimization only
        3. With reranking only
        4. Full pipeline (optimization + expansion + reranking)
        """
        # Basic search
        basic = await self.hybrid_search.search(query, top_k=top_k)

        # With optimization only
        optimized_query = self.query_optimizer.optimize_query(query)
        with_optimization = await self.hybrid_search.search(
            optimized_query,
            top_k=top_k
        )

        # With reranking only
        basic_50 = await self.hybrid_search.search(query, top_k=50)
        reranked = self.reranker.rerank(query, basic_50, top_k=top_k)
        with_reranking = [doc for doc, score in reranked]

        # Full pipeline
        full_pipeline = await self.search_with_optimization(
            query,
            expand_query=True,
            rerank=True,
            top_k=top_k
        )

        return {
            "query": query,
            "basic": basic,
            "with_optimization": with_optimization,
            "with_reranking": with_reranking,
            "full_pipeline": full_pipeline["results"],
            "pipeline_stats": full_pipeline["pipeline_stats"]
        }
```

---

## 5. Discord Commands

### Update `bot/cogs/rag_cog.py`

Add these commands:

```python
from rag.advanced import AdvancedRAGService

class RAGCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.advanced_rag = AdvancedRAGService()

    @commands.command(name="rerank_search")
    async def rerank_search(self, ctx, *, query: str):
        """
        Search with reranking.

        Usage: !rerank_search how to make pizza
        """
        await ctx.send(f"🔍 Searching with reranking: `{query}`")

        try:
            result = await self.advanced_rag.search_with_optimization(
                query,
                expand_query=False,  # No expansion for this command
                rerank=True,
                top_k=5
            )

            # Display results
            embed = discord.Embed(
                title=f"🎯 Reranked Results for: {query}",
                color=discord.Color.green()
            )

            for idx, doc in enumerate(result["results"], 1):
                embed.add_field(
                    name=f"{idx}. Score: {doc.get('rerank_score', 0):.4f}",
                    value=doc["content"][:100] + "...",
                    inline=False
                )

            # Pipeline stats
            steps = result["pipeline_stats"]["steps"]
            embed.set_footer(text=f"Pipeline: {' → '.join(steps)}")

            await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

    @commands.command(name="expand_search")
    async def expand_search(self, ctx, *, query: str):
        """
        Search with query expansion.

        Usage: !expand_search how to bake bread
        """
        await ctx.send(f"🔍 Searching with query expansion: `{query}`")

        try:
            result = await self.advanced_rag.search_with_optimization(
                query,
                expand_query=True,
                rerank=False,
                top_k=5
            )

            # Show query variations
            variations = result["query_variations"]
            embed = discord.Embed(
                title=f"📝 Query Expansion",
                description=f"**Original:** {query}\n\n**Variations:**",
                color=discord.Color.blue()
            )

            for idx, var in enumerate(variations[1:], 1):  # Skip original
                embed.description += f"\n{idx}. {var}"

            # Display results
            embed.add_field(
                name="🎯 Top Results",
                value=f"Found {len(result['results'])} results",
                inline=False
            )

            for idx, doc in enumerate(result["results"][:3], 1):
                embed.add_field(
                    name=f"{idx}. {doc.get('author', 'Unknown')}",
                    value=doc["content"][:100] + "...",
                    inline=False
                )

            await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

    @commands.command(name="advanced_search")
    async def advanced_search(self, ctx, *, query: str):
        """
        Full advanced RAG pipeline (optimization + expansion + reranking).

        Usage: !advanced_search what are the best python frameworks
        """
        await ctx.send(f"🚀 Running advanced RAG pipeline: `{query}`")

        try:
            result = await self.advanced_rag.search_with_optimization(
                query,
                expand_query=True,
                rerank=True,
                top_k=5
            )

            # Display results
            embed = discord.Embed(
                title=f"🎯 Advanced Search Results",
                description=f"**Original:** {result['original_query']}\n"
                           f"**Optimized:** {result['optimized_query']}",
                color=discord.Color.gold()
            )

            # Query variations
            if result["query_variations"]:
                variations_text = "\n".join(
                    f"{idx}. {var}"
                    for idx, var in enumerate(result["query_variations"][1:], 1)
                )
                embed.add_field(
                    name="📝 Query Variations",
                    value=variations_text or "None",
                    inline=False
                )

            # Top results
            for idx, doc in enumerate(result["results"], 1):
                score = doc.get("rerank_score", doc.get("fusion_score", 0))
                embed.add_field(
                    name=f"{idx}. Score: {score:.4f}",
                    value=doc["content"][:100] + "...",
                    inline=False
                )

            # Pipeline stats
            steps = result["pipeline_stats"]["steps"]
            embed.set_footer(text=f"Pipeline: {' → '.join(steps)}")

            await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

    @commands.command(name="compare_pipelines")
    async def compare_pipelines(self, ctx, *, query: str):
        """
        Compare different RAG pipeline configurations.

        Usage: !compare_pipelines how to learn python
        """
        await ctx.send(f"📊 Comparing RAG pipelines for: `{query}`")

        try:
            comparison = await self.advanced_rag.compare_pipelines(query, top_k=3)

            embed = discord.Embed(
                title=f"📊 Pipeline Comparison",
                description=f"Query: **{query}**",
                color=discord.Color.purple()
            )

            # Basic search
            basic_docs = [doc["content"][:50] for doc in comparison["basic"][:2]]
            embed.add_field(
                name="1️⃣ Basic Search",
                value="\n".join(f"• {doc}..." for doc in basic_docs),
                inline=False
            )

            # With optimization
            opt_docs = [doc["content"][:50] for doc in comparison["with_optimization"][:2]]
            embed.add_field(
                name="2️⃣ With Query Optimization",
                value="\n".join(f"• {doc}..." for doc in opt_docs),
                inline=False
            )

            # With reranking
            rerank_docs = [doc["content"][:50] for doc in comparison["with_reranking"][:2]]
            embed.add_field(
                name="3️⃣ With Reranking",
                value="\n".join(f"• {doc}..." for doc in rerank_docs),
                inline=False
            )

            # Full pipeline
            full_docs = [doc["content"][:50] for doc in comparison["full_pipeline"][:2]]
            embed.add_field(
                name="4️⃣ Full Pipeline (Optimization + Expansion + Reranking)",
                value="\n".join(f"• {doc}..." for doc in full_docs),
                inline=False
            )

            steps = comparison["pipeline_stats"]["steps"]
            embed.set_footer(text=f"Full pipeline: {' → '.join(steps)}")

            await ctx.send(embed=embed)

        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")
```

---

## 6. Testing & Comparison

### Test Queries

Try these queries to see the difference:

**1. Query Expansion Test:**
```
!expand_search how to fix bugs
```
Expected variations:
- "debugging techniques"
- "troubleshooting code errors"
- "solving programming issues"

**2. Reranking Test:**
```
!compare_pipelines what's the best way to learn Python
```
Compare how reranking changes the order of results.

**3. Full Pipeline Test:**
```
!advanced_search explain machine learning for beginners
```
See all stages: optimization → expansion → hybrid search → fusion → reranking

### Evaluation

Add to `services/retrieval_evaluator.py`:

```python
def evaluate_reranking(
    self,
    query: str,
    original_results: List[Dict],
    reranked_results: List[Dict],
    ground_truth_ids: List[str]
) -> Dict:
    """
    Compare original vs reranked results.

    Returns:
        {
            "original_mrr": float,
            "reranked_mrr": float,
            "improvement": float,
            "rank_changes": List[Dict]
        }
    """
    # Calculate MRR for original
    original_ids = [doc["id"] for doc in original_results]
    original_mrr = self.mean_reciprocal_rank([original_ids], [ground_truth_ids])

    # Calculate MRR for reranked
    reranked_ids = [doc["id"] for doc in reranked_results]
    reranked_mrr = self.mean_reciprocal_rank([reranked_ids], [ground_truth_ids])

    # Calculate improvement
    improvement = (reranked_mrr - original_mrr) / original_mrr * 100

    return {
        "original_mrr": original_mrr,
        "reranked_mrr": reranked_mrr,
        "improvement_pct": improvement,
        "improved": improvement > 0
    }
```

---

## 7. Performance Considerations

### Speed vs Accuracy Trade-offs

| Configuration | Speed | Accuracy | Best For |
|---------------|-------|----------|----------|
| **Basic search** | ⚡⚡⚡ Fast | ⭐⭐ Good | High-volume queries |
| **+ Optimization** | ⚡⚡⚡ Fast | ⭐⭐⭐ Better | Cleaning messy queries |
| **+ Expansion** | ⚡⚡ Medium | ⭐⭐⭐⭐ Great | Vocabulary mismatch |
| **+ Reranking** | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | High-precision needs |
| **Full pipeline** | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | Best possible results |

### Optimization Tips

**1. Batch Reranking:**
```python
# Instead of reranking one-by-one
for query in queries:
    rerank(query, docs)  # ❌ Slow

# Batch all pairs together
all_pairs = [[q, d] for q in queries for d in docs]
scores = model.predict(all_pairs)  # ✅ Fast
```

**2. Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_rerank(query: str, doc_ids: tuple):
    """Cache reranking results for repeated queries."""
    pass
```

**3. Selective Reranking:**
```python
# Only rerank if initial scores are close
if max_score - min_score < 0.1:
    # Scores are very close, use reranking
    reranked = reranker.rerank(query, docs)
else:
    # Clear winner, skip reranking
    return docs
```

### Cost Optimization

**Query Expansion with LLM:**
- Cost: ~$0.0001 per query (GPT-3.5-turbo)
- Alternative: Use keyword-based expansion (free but less effective)

**When to skip expansion:**
- Short, clear queries: "Python tutorial"
- Named entity queries: "OpenAI API documentation"
- Already specific queries: "how to install pandas version 1.5.3"

---

## Summary

### What We Learned

1. **Reranking**: Two-stage retrieval (fast bi-encoder → slow cross-encoder)
2. **Query Expansion**: Generate variations to handle vocabulary mismatch
3. **Query Optimization**: Clean and normalize queries
4. **Pipeline Design**: Chain multiple techniques for best results
5. **Trade-offs**: Speed vs accuracy considerations

### Key Takeaways

✅ **DO:**
- Use reranking for high-precision requirements
- Expand queries when dealing with diverse vocabulary
- Cache expensive operations
- Measure before and after (use evaluation metrics)

❌ **DON'T:**
- Rerank every query (too slow for high-volume)
- Expand already-clear queries (wastes money)
- Skip optimization in production (helps a lot)

### Next Steps

**Phase 16**: Advanced RAG Techniques (HyDE, Self-RAG, RAG Fusion)
**Phase 17**: RAG Strategy Comparison Dashboard

---

## Additional Resources

- [Cross-Encoders vs Bi-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/) (Cross-encoder training data)
- [Query Expansion Techniques](https://en.wikipedia.org/wiki/Query_expansion)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
