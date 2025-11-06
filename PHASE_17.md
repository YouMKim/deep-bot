# Phase 17: RAG Strategy Comparison Dashboard

## Overview

Congratulations! You've implemented **8+ different RAG strategies** across Phases 14-16:
- Vector Search
- Keyword Search (BM25)
- Hybrid Search
- Query Optimization & Expansion
- Cross-Encoder Reranking
- HyDE (Hypothetical Document Embeddings)
- Self-RAG (Self-Reflective Retrieval)
- RAG Fusion (Multi-Query Synthesis)

But which one should you use? **It depends!** Different queries benefit from different strategies. This phase builds a **comprehensive comparison dashboard** that lets you:
- Compare all strategies side-by-side
- Toggle between strategies interactively
- Visualize performance differences
- Get automatic recommendations
- Learn when to use each strategy

**Learning Objectives:**
- Build a unified RAG service that supports all strategies
- Create comparison tools for A/B testing
- Implement performance metrics tracking
- Develop a strategy recommendation system
- Visualize results in Discord with interactive embeds

**Prerequisites:** Phases 14-16 (All Advanced RAG Techniques)

**Estimated Time:** 4-6 hours

---

## Table of Contents

1. [Unified RAG Service](#1-unified-rag-service)
2. [Comparison Dashboard](#2-comparison-dashboard)
3. [Performance Metrics](#3-performance-metrics)
4. [Strategy Recommendation Engine](#4-strategy-recommendation-engine)
5. [Discord Commands](#5-discord-commands)
6. [Interactive Comparison Tools](#6-interactive-comparison-tools)
7. [Visualization & Analytics](#7-visualization--analytics)

---

## 1. Unified RAG Service

### Architecture

Create `rag/unified_service.py`:

```python
from typing import Dict, List, Optional, Literal
from retrieval.vector import VectorDBService
from retrieval.keyword import BM25Retriever
from retrieval.hybrid import HybridSearchService
from retrieval.advanced.query_optimizer import QueryOptimizer
from retrieval.reranking import RerankingService
from retrieval.advanced.hyde import HyDEService
from retrieval.advanced.self_rag import SelfRAGService
from retrieval.advanced.fusion import RAGFusionService
import time
import logging

logger = logging.getLogger(__name__)

RAGStrategy = Literal[
    "vector",
    "keyword",
    "hybrid",
    "hybrid_rerank",
    "query_expansion",
    "query_expansion_rerank",
    "hyde",
    "self_rag",
    "rag_fusion"
]

class UnifiedRAGService:
    """
    Unified RAG service supporting all strategies.

    Provides a single interface for all RAG techniques with:
    - Consistent API across all strategies
    - Performance tracking
    - Automatic logging
    - Easy comparison
    """

    def __init__(self):
        # Initialize all services
        self.vector_db = VectorDBService()
        self.bm25 = BM25Retriever()
        self.hybrid_search = HybridSearchService()
        self.query_optimizer = QueryOptimizer()
        self.reranker = RerankingService()
        self.hyde = HyDEService()
        self.self_rag = SelfRAGService()
        self.rag_fusion = RAGFusionService()

        logger.info("UnifiedRAGService initialized with all strategies")

    async def search(
        self,
        query: str,
        strategy: RAGStrategy = "hybrid_rerank",
        top_k: int = 10,
        **kwargs
    ) -> Dict:
        """
        Execute search with specified strategy.

        Args:
            query: User query
            strategy: RAG strategy to use
            top_k: Number of results to return
            **kwargs: Strategy-specific parameters

        Returns:
            {
                "strategy": str,
                "query": str,
                "results": List[Dict],
                "metadata": Dict,  # Strategy-specific info
                "performance": Dict  # Latency, cost, etc.
            }
        """
        start_time = time.time()

        # Route to appropriate strategy
        if strategy == "vector":
            result = await self._vector_search(query, top_k)

        elif strategy == "keyword":
            result = await self._keyword_search(query, top_k)

        elif strategy == "hybrid":
            result = await self._hybrid_search(query, top_k, kwargs)

        elif strategy == "hybrid_rerank":
            result = await self._hybrid_rerank_search(query, top_k, kwargs)

        elif strategy == "query_expansion":
            result = await self._query_expansion_search(query, top_k, kwargs)

        elif strategy == "query_expansion_rerank":
            result = await self._query_expansion_rerank_search(query, top_k, kwargs)

        elif strategy == "hyde":
            result = await self._hyde_search(query, top_k, kwargs)

        elif strategy == "self_rag":
            result = await self._self_rag_search(query, top_k, kwargs)

        elif strategy == "rag_fusion":
            result = await self._rag_fusion_search(query, top_k, kwargs)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add performance metrics
        latency_ms = (time.time() - start_time) * 1000

        result["performance"] = {
            "latency_ms": latency_ms,
            "timestamp": time.time()
        }

        return result

    # Strategy implementations
    async def _vector_search(self, query: str, top_k: int) -> Dict:
        """Basic vector search."""
        from embedding import EmbeddingService
        embedding_service = EmbeddingService()

        embedding = await embedding_service.embed_text(query)
        results = await self.vector_db.search(embedding=embedding, top_k=top_k)

        return {
            "strategy": "vector",
            "query": query,
            "results": results,
            "metadata": {
                "description": "Dense vector similarity search"
            }
        }

    async def _keyword_search(self, query: str, top_k: int) -> Dict:
        """BM25 keyword search."""
        results = self.bm25.search(query, top_k=top_k)

        return {
            "strategy": "keyword",
            "query": query,
            "results": results,
            "metadata": {
                "description": "BM25 sparse keyword retrieval"
            }
        }

    async def _hybrid_search(self, query: str, top_k: int, kwargs: Dict) -> Dict:
        """Hybrid vector + keyword search."""
        alpha = kwargs.get("alpha", 0.5)

        results = await self.hybrid_search.search(
            query,
            top_k=top_k,
            alpha=alpha
        )

        return {
            "strategy": "hybrid",
            "query": query,
            "results": results,
            "metadata": {
                "description": f"Hybrid search (alpha={alpha})",
                "alpha": alpha
            }
        }

    async def _hybrid_rerank_search(self, query: str, top_k: int, kwargs: Dict) -> Dict:
        """Hybrid search + cross-encoder reranking."""
        alpha = kwargs.get("alpha", 0.5)

        # Get more candidates for reranking
        candidates = await self.hybrid_search.search(
            query,
            top_k=top_k * 5,
            alpha=alpha
        )

        # Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        results = [doc for doc, score in reranked]

        return {
            "strategy": "hybrid_rerank",
            "query": query,
            "results": results,
            "metadata": {
                "description": f"Hybrid search + reranking",
                "alpha": alpha,
                "candidates_retrieved": len(candidates),
                "final_results": len(results)
            }
        }

    async def _query_expansion_search(self, query: str, top_k: int, kwargs: Dict) -> Dict:
        """Query expansion + search."""
        num_variations = kwargs.get("num_variations", 2)

        # Expand query
        variations = self.query_optimizer.expand_query_llm(query, num_variations)

        # Search with all variations
        from embedding import EmbeddingService
        embedding_service = EmbeddingService()

        all_results = []
        for var in variations:
            embedding = await embedding_service.embed_text(var)
            results = await self.vector_db.search(embedding=embedding, top_k=top_k)
            all_results.append(results)

        # Merge using RRF
        merged = self.query_optimizer.merge_results(all_results, top_k=top_k)

        return {
            "strategy": "query_expansion",
            "query": query,
            "results": merged,
            "metadata": {
                "description": f"Query expansion ({len(variations)} variations)",
                "query_variations": variations
            }
        }

    async def _query_expansion_rerank_search(self, query: str, top_k: int, kwargs: Dict) -> Dict:
        """Query expansion + reranking."""
        num_variations = kwargs.get("num_variations", 2)

        # Get expanded results
        expanded = await self._query_expansion_search(query, top_k * 3, kwargs)

        # Rerank
        reranked = self.reranker.rerank(query, expanded["results"], top_k=top_k)
        results = [doc for doc, score in reranked]

        return {
            "strategy": "query_expansion_rerank",
            "query": query,
            "results": results,
            "metadata": {
                "description": f"Query expansion + reranking",
                "query_variations": expanded["metadata"]["query_variations"]
            }
        }

    async def _hyde_search(self, query: str, top_k: int, kwargs: Dict) -> Dict:
        """HyDE search."""
        num_hypothetical = kwargs.get("num_hypothetical", 1)

        result = await self.hyde.search_with_hyde(
            query,
            top_k=top_k,
            num_hypothetical_docs=num_hypothetical
        )

        return {
            "strategy": "hyde",
            "query": query,
            "results": result["results"],
            "metadata": {
                "description": "HyDE (Hypothetical Document Embeddings)",
                "hypothetical_docs": result["hypothetical_docs"]
            }
        }

    async def _self_rag_search(self, query: str, top_k: int, kwargs: Dict) -> Dict:
        """Self-RAG search."""
        result = await self.self_rag.self_rag_query(query, top_k=top_k)

        return {
            "strategy": "self_rag",
            "query": query,
            "results": result.get("results", []),
            "metadata": {
                "description": "Self-RAG (Self-Reflective Retrieval)",
                "reflections": {
                    "retrieval_needed": result["reflection_1"],
                    "relevance_check": result.get("reflection_2"),
                    "support_check": result.get("reflection_3")
                },
                "confidence": result.get("final_confidence", 0.0)
            }
        }

    async def _rag_fusion_search(self, query: str, top_k: int, kwargs: Dict) -> Dict:
        """RAG Fusion search."""
        num_queries = kwargs.get("num_queries", 4)

        result = await self.rag_fusion.rag_fusion_search(
            query,
            num_queries=num_queries,
            top_k=top_k
        )

        return {
            "strategy": "rag_fusion",
            "query": query,
            "results": result["final_results"],
            "metadata": {
                "description": f"RAG Fusion ({len(result['generated_queries'])} queries)",
                "generated_queries": result["generated_queries"]
            }
        }
```

---

## 2. Comparison Dashboard

### Multi-Strategy Comparison

Add to `evaluation/comparison.py`:

```python
from typing import List, Dict
from rag.unified_service import UnifiedRAGService, RAGStrategy
import asyncio

class ComparisonService:
    """Compare multiple RAG strategies on the same query."""

    def __init__(self):
        self.unified_rag = UnifiedRAGService()

    async def compare_strategies(
        self,
        query: str,
        strategies: List[RAGStrategy],
        top_k: int = 5
    ) -> Dict:
        """
        Run query through multiple strategies and compare results.

        Args:
            query: User query
            strategies: List of strategies to compare
            top_k: Number of results per strategy

        Returns:
            {
                "query": str,
                "comparisons": Dict[strategy -> result],
                "analysis": Dict  # Overlap, performance, etc.
            }
        """
        # Run all strategies in parallel
        tasks = [
            self.unified_rag.search(query, strategy=strat, top_k=top_k)
            for strat in strategies
        ]

        results = await asyncio.gather(*tasks)

        # Build comparison dict
        comparisons = {
            strat: result
            for strat, result in zip(strategies, results)
        }

        # Analyze differences
        analysis = self._analyze_comparisons(comparisons, top_k)

        return {
            "query": query,
            "comparisons": comparisons,
            "analysis": analysis
        }

    def _analyze_comparisons(
        self,
        comparisons: Dict[str, Dict],
        top_k: int
    ) -> Dict:
        """
        Analyze differences between strategy results.

        Returns:
            {
                "overlap_matrix": Dict,
                "performance_comparison": Dict,
                "unique_docs": Dict,
                "ranking_differences": Dict
            }
        """
        strategies = list(comparisons.keys())

        # 1. Overlap matrix (Jaccard similarity)
        overlap_matrix = {}
        for strat1 in strategies:
            overlap_matrix[strat1] = {}
            ids1 = {doc["id"] for doc in comparisons[strat1]["results"][:top_k]}

            for strat2 in strategies:
                ids2 = {doc["id"] for doc in comparisons[strat2]["results"][:top_k]}
                overlap = len(ids1 & ids2)
                union = len(ids1 | ids2)
                jaccard = overlap / union if union > 0 else 0

                overlap_matrix[strat1][strat2] = {
                    "overlap_count": overlap,
                    "jaccard_similarity": jaccard
                }

        # 2. Performance comparison
        performance_comparison = {}
        for strat, result in comparisons.items():
            perf = result["performance"]
            performance_comparison[strat] = {
                "latency_ms": perf["latency_ms"],
                "rank": 0  # Will be set below
            }

        # Rank by latency
        sorted_by_latency = sorted(
            performance_comparison.items(),
            key=lambda x: x[1]["latency_ms"]
        )
        for rank, (strat, data) in enumerate(sorted_by_latency, 1):
            performance_comparison[strat]["rank"] = rank

        # 3. Unique documents per strategy
        unique_docs = {}
        for strat in strategies:
            ids = {doc["id"] for doc in comparisons[strat]["results"][:top_k]}
            other_ids = set()
            for other_strat in strategies:
                if other_strat != strat:
                    other_ids.update(
                        doc["id"] for doc in comparisons[other_strat]["results"][:top_k]
                    )

            unique = ids - other_ids
            unique_docs[strat] = {
                "count": len(unique),
                "doc_ids": list(unique)
            }

        # 4. Ranking differences (how much do ranks differ?)
        ranking_differences = {}
        for strat in strategies:
            doc_ranks = {
                doc["id"]: rank
                for rank, doc in enumerate(comparisons[strat]["results"][:top_k], 1)
            }

            total_rank_diff = 0
            comparisons_made = 0

            for other_strat in strategies:
                if other_strat == strat:
                    continue

                other_doc_ranks = {
                    doc["id"]: rank
                    for rank, doc in enumerate(comparisons[other_strat]["results"][:top_k], 1)
                }

                # Compare ranks for shared documents
                shared_ids = set(doc_ranks.keys()) & set(other_doc_ranks.keys())
                for doc_id in shared_ids:
                    rank_diff = abs(doc_ranks[doc_id] - other_doc_ranks[doc_id])
                    total_rank_diff += rank_diff
                    comparisons_made += 1

            avg_rank_diff = total_rank_diff / comparisons_made if comparisons_made > 0 else 0
            ranking_differences[strat] = {
                "avg_rank_difference": avg_rank_diff
            }

        return {
            "overlap_matrix": overlap_matrix,
            "performance_comparison": performance_comparison,
            "unique_docs": unique_docs,
            "ranking_differences": ranking_differences
        }

    async def compare_all_strategies(self, query: str, top_k: int = 5) -> Dict:
        """
        Run query through ALL available strategies.

        Convenience method for comprehensive comparison.
        """
        all_strategies: List[RAGStrategy] = [
            "vector",
            "keyword",
            "hybrid",
            "hybrid_rerank",
            "query_expansion",
            "query_expansion_rerank",
            "hyde",
            "self_rag",
            "rag_fusion"
        ]

        return await self.compare_strategies(query, all_strategies, top_k)
```

---

## 3. Performance Metrics

### Metrics Tracker

Create `evaluation/metrics_tracker.py`:

```python
from typing import Dict, List
from collections import defaultdict
import time
import json

class MetricsTracker:
    """Track and analyze RAG performance metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record_query(
        self,
        strategy: str,
        query: str,
        latency_ms: float,
        num_results: int,
        relevance_score: float = None
    ):
        """Record a query execution."""
        self.metrics[strategy].append({
            "timestamp": time.time(),
            "query": query,
            "latency_ms": latency_ms,
            "num_results": num_results,
            "relevance_score": relevance_score
        })

    def get_strategy_stats(self, strategy: str) -> Dict:
        """Get statistics for a strategy."""
        if strategy not in self.metrics:
            return {}

        data = self.metrics[strategy]
        latencies = [d["latency_ms"] for d in data]

        return {
            "total_queries": len(data),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": self._percentile(latencies, 50),
            "p95_latency_ms": self._percentile(latencies, 95),
            "p99_latency_ms": self._percentile(latencies, 99)
        }

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all strategies."""
        return {
            strategy: self.get_strategy_stats(strategy)
            for strategy in self.metrics.keys()
        }

    def compare_performance(self) -> List[Dict]:
        """
        Compare all strategies by performance.

        Returns ranked list from fastest to slowest.
        """
        stats = self.get_all_stats()

        ranked = [
            {
                "strategy": strategy,
                "avg_latency_ms": data["avg_latency_ms"],
                "total_queries": data["total_queries"]
            }
            for strategy, data in stats.items()
        ]

        ranked.sort(key=lambda x: x["avg_latency_ms"])

        return ranked

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        with open(filepath, "w") as f:
            json.dump(dict(self.metrics), f, indent=2)

    def load_metrics(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, "r") as f:
            loaded = json.load(f)
            self.metrics = defaultdict(list, loaded)
```

---

## 4. Strategy Recommendation Engine

### Automatic Strategy Selection

Create `evaluation/recommender.py`:

```python
from typing import Dict, List
from rag.unified_service import RAGStrategy

class StrategyRecommender:
    """Recommend best RAG strategy based on query characteristics."""

    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query characteristics.

        Returns:
            {
                "length": int,
                "is_question": bool,
                "complexity": str,  # simple, medium, complex
                "has_keywords": bool,
                "category": str  # factual, exploratory, conversational
            }
        """
        words = query.split()
        length = len(words)

        # Check if question
        is_question = query.strip().endswith("?") or any(
            query.lower().startswith(q)
            for q in ["what", "how", "why", "when", "where", "who", "which"]
        )

        # Estimate complexity
        if length <= 5:
            complexity = "simple"
        elif length <= 15:
            complexity = "medium"
        else:
            complexity = "complex"

        # Check for specific keywords
        has_keywords = any(word.isupper() for word in words)  # Acronyms
        has_keywords |= any(char.isdigit() for char in query)  # Numbers/dates
        has_keywords |= '"' in query  # Quoted terms

        # Categorize
        if is_question and length <= 10:
            category = "factual"
        elif any(word in query.lower() for word in ["best", "compare", "overview", "explain"]):
            category = "exploratory"
        else:
            category = "conversational"

        return {
            "length": length,
            "is_question": is_question,
            "complexity": complexity,
            "has_keywords": has_keywords,
            "category": category
        }

    def recommend_strategy(self, query: str) -> Dict:
        """
        Recommend best strategy for query.

        Returns:
            {
                "recommended": RAGStrategy,
                "alternatives": List[RAGStrategy],
                "reasoning": str
            }
        """
        analysis = self.analyze_query(query)

        # Decision tree
        if analysis["has_keywords"]:
            # Keywords present → use keyword or hybrid search
            recommended = "hybrid_rerank"
            alternatives = ["keyword", "hybrid"]
            reasoning = "Query contains specific keywords/terms → Hybrid search with reranking"

        elif analysis["category"] == "factual" and analysis["complexity"] == "simple":
            # Simple factual question → fast vector search
            recommended = "vector"
            alternatives = ["hybrid"]
            reasoning = "Simple factual question → Fast vector search sufficient"

        elif analysis["is_question"] and analysis["complexity"] == "medium":
            # Question format → HyDE works well
            recommended = "hyde"
            alternatives = ["hybrid_rerank", "query_expansion"]
            reasoning = "Question format → HyDE bridges semantic gap"

        elif analysis["category"] == "exploratory":
            # Exploratory query → RAG Fusion for comprehensive coverage
            recommended = "rag_fusion"
            alternatives = ["query_expansion_rerank", "hybrid_rerank"]
            reasoning = "Exploratory query → RAG Fusion for comprehensive results"

        elif analysis["complexity"] == "complex":
            # Complex query → Full pipeline
            recommended = "query_expansion_rerank"
            alternatives = ["rag_fusion", "self_rag"]
            reasoning = "Complex query → Full optimization pipeline"

        else:
            # Default: hybrid with reranking
            recommended = "hybrid_rerank"
            alternatives = ["hybrid", "vector"]
            reasoning = "General query → Balanced hybrid approach"

        return {
            "recommended": recommended,
            "alternatives": alternatives,
            "reasoning": reasoning,
            "query_analysis": analysis
        }

    def get_strategy_profile(self, strategy: RAGStrategy) -> Dict:
        """Get detailed profile of a strategy."""
        profiles = {
            "vector": {
                "name": "Vector Search",
                "speed": "⚡⚡⚡ Very Fast",
                "accuracy": "⭐⭐⭐ Good",
                "cost": "💰 Low",
                "best_for": "Simple queries, high-volume scenarios",
                "avoid_for": "Keyword-specific searches, complex reasoning"
            },
            "keyword": {
                "name": "Keyword Search (BM25)",
                "speed": "⚡⚡⚡ Very Fast",
                "accuracy": "⭐⭐ Fair",
                "cost": "💰 Very Low",
                "best_for": "Exact keyword matching, names, codes",
                "avoid_for": "Semantic understanding, paraphrases"
            },
            "hybrid": {
                "name": "Hybrid Search",
                "speed": "⚡⚡ Fast",
                "accuracy": "⭐⭐⭐⭐ Very Good",
                "cost": "💰 Low",
                "best_for": "Balanced retrieval, general queries",
                "avoid_for": "When you need top precision"
            },
            "hybrid_rerank": {
                "name": "Hybrid + Reranking",
                "speed": "⚡ Medium",
                "accuracy": "⭐⭐⭐⭐⭐ Excellent",
                "cost": "💰💰 Medium",
                "best_for": "High-precision needs, important queries",
                "avoid_for": "High-volume, latency-sensitive scenarios"
            },
            "query_expansion": {
                "name": "Query Expansion",
                "speed": "⚡ Medium",
                "accuracy": "⭐⭐⭐⭐ Very Good",
                "cost": "💰💰 Medium",
                "best_for": "Vocabulary mismatch, diverse phrasings",
                "avoid_for": "Very specific keyword queries"
            },
            "query_expansion_rerank": {
                "name": "Query Expansion + Reranking",
                "speed": "⚡ Slow",
                "accuracy": "⭐⭐⭐⭐⭐ Excellent",
                "cost": "💰💰💰 High",
                "best_for": "Complex queries, best possible results",
                "avoid_for": "Real-time/interactive scenarios"
            },
            "hyde": {
                "name": "HyDE",
                "speed": "⚡ Medium",
                "accuracy": "⭐⭐⭐⭐ Very Good",
                "cost": "💰💰 Medium",
                "best_for": "Question-answering, semantic gap",
                "avoid_for": "Keyword searches, FAQ databases"
            },
            "self_rag": {
                "name": "Self-RAG",
                "speed": "⚡ Slow",
                "accuracy": "⭐⭐⭐⭐⭐ Excellent",
                "cost": "💰💰💰 High",
                "best_for": "High-stakes queries, quality assurance",
                "avoid_for": "High-volume scenarios"
            },
            "rag_fusion": {
                "name": "RAG Fusion",
                "speed": "⚡ Slow",
                "accuracy": "⭐⭐⭐⭐⭐ Excellent",
                "cost": "💰💰💰 High",
                "best_for": "Exploratory queries, comprehensive coverage",
                "avoid_for": "Simple factual questions"
            }
        }

        return profiles.get(strategy, {})
```

---

## 5. Discord Commands

### Comparison Commands

Add to `bot/cogs/comparison_cog.py`:

```python
import discord
from discord.ext import commands
from rag.unified_service import UnifiedRAGService
from evaluation.comparison import ComparisonService
from evaluation.recommender import StrategyRecommender

class ComparisonCog(commands.Cog):
    """RAG Strategy Comparison Commands."""

    def __init__(self, bot):
        self.bot = bot
        self.unified_rag = UnifiedRAGService()
        self.comparison = ComparisonService()
        self.recommender = StrategyRecommender()

    @commands.command(name="rag_recommend")
    async def rag_recommend(self, ctx, *, query: str):
        """
        Get strategy recommendation for a query.

        Usage: !rag_recommend what are the best python frameworks?
        """
        recommendation = self.recommender.recommend_strategy(query)

        embed = discord.Embed(
            title="🎯 Strategy Recommendation",
            description=f"**Query:** {query}",
            color=discord.Color.blue()
        )

        # Recommended strategy
        rec_profile = self.recommender.get_strategy_profile(
            recommendation["recommended"]
        )

        embed.add_field(
            name=f"✅ Recommended: {rec_profile['name']}",
            value=f"{rec_profile['speed']} | {rec_profile['accuracy']}\n"
                  f"**Best for:** {rec_profile['best_for']}\n"
                  f"**Reasoning:** {recommendation['reasoning']}",
            inline=False
        )

        # Alternatives
        alt_text = []
        for alt in recommendation["alternatives"][:2]:
            alt_profile = self.recommender.get_strategy_profile(alt)
            alt_text.append(f"• {alt_profile['name']}: {alt_profile['best_for']}")

        embed.add_field(
            name="🔄 Alternatives",
            value="\n".join(alt_text),
            inline=False
        )

        # Query analysis
        analysis = recommendation["query_analysis"]
        embed.set_footer(
            text=f"Category: {analysis['category']} | "
                 f"Complexity: {analysis['complexity']} | "
                 f"Length: {analysis['length']} words"
        )

        await ctx.send(embed=embed)

    @commands.command(name="rag_compare")
    async def rag_compare(self, ctx, *, query: str):
        """
        Compare multiple RAG strategies on same query.

        Usage: !rag_compare what causes rain?
        """
        await ctx.send(f"🔬 Comparing strategies for: `{query}`\nThis may take 30-60 seconds...")

        # Compare subset of strategies (all would take too long)
        strategies = ["vector", "hybrid", "hybrid_rerank", "hyde", "rag_fusion"]

        comparison = await self.comparison.compare_strategies(
            query,
            strategies,
            top_k=3
        )

        embed = discord.Embed(
            title="🔬 Strategy Comparison",
            description=f"**Query:** {query}",
            color=discord.Color.purple()
        )

        # Show top result from each strategy
        for strategy in strategies:
            result = comparison["comparisons"][strategy]
            top_doc = result["results"][0] if result["results"] else None

            if top_doc:
                profile = self.recommender.get_strategy_profile(strategy)
                latency = result["performance"]["latency_ms"]

                embed.add_field(
                    name=f"{profile['name']} ({latency:.0f}ms)",
                    value=top_doc["content"][:100] + "...",
                    inline=False
                )

        # Performance summary
        perf = comparison["analysis"]["performance_comparison"]
        fastest = min(perf.items(), key=lambda x: x[1]["latency_ms"])
        embed.set_footer(
            text=f"Fastest: {fastest[0]} ({fastest[1]['latency_ms']:.0f}ms)"
        )

        await ctx.send(embed=embed)

    @commands.command(name="rag_overlap")
    async def rag_overlap(self, ctx, *, query: str):
        """
        Show overlap between different strategies.

        Usage: !rag_overlap machine learning basics
        """
        await ctx.send(f"📊 Analyzing strategy overlap for: `{query}`")

        strategies = ["vector", "hybrid_rerank", "hyde"]

        comparison = await self.comparison.compare_strategies(
            query,
            strategies,
            top_k=5
        )

        embed = discord.Embed(
            title="📊 Strategy Overlap Analysis",
            description=f"**Query:** {query}",
            color=discord.Color.gold()
        )

        # Overlap matrix
        overlap_matrix = comparison["analysis"]["overlap_matrix"]

        for strat1 in strategies:
            overlap_text = []
            for strat2 in strategies:
                if strat1 != strat2:
                    data = overlap_matrix[strat1][strat2]
                    jaccard = data["jaccard_similarity"]
                    overlap_count = data["overlap_count"]
                    overlap_text.append(
                        f"vs {strat2}: {overlap_count}/5 docs ({jaccard*100:.0f}% similar)"
                    )

            embed.add_field(
                name=f"{strat1}",
                value="\n".join(overlap_text),
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.command(name="rag_profiles")
    async def rag_profiles(self, ctx):
        """
        Show all RAG strategy profiles.

        Usage: !rag_profiles
        """
        strategies = [
            "vector", "keyword", "hybrid", "hybrid_rerank",
            "query_expansion", "hyde", "self_rag", "rag_fusion"
        ]

        embed = discord.Embed(
            title="📚 RAG Strategy Profiles",
            description="Overview of all available strategies",
            color=discord.Color.blue()
        )

        for strategy in strategies:
            profile = self.recommender.get_strategy_profile(strategy)

            embed.add_field(
                name=profile["name"],
                value=f"{profile['speed']} | {profile['accuracy']} | {profile['cost']}\n"
                      f"**Best for:** {profile['best_for']}",
                inline=False
            )

        await ctx.send(embed=embed)
```

---

## 6. Interactive Comparison Tools

### A/B Testing

Create `evaluation/ab_testing.py`:

```python
from typing import Dict, List
from rag.unified_service import UnifiedRAGService, RAGStrategy
import random

class ABTestingService:
    """A/B testing for RAG strategies."""

    def __init__(self):
        self.unified_rag = UnifiedRAGService()
        self.test_results = []

    async def run_ab_test(
        self,
        query: str,
        strategy_a: RAGStrategy,
        strategy_b: RAGStrategy,
        top_k: int = 5
    ) -> Dict:
        """
        Run A/B test between two strategies.

        Returns:
            {
                "query": str,
                "strategy_a": {...},
                "strategy_b": {...},
                "winner": str,  # or "tie"
                "metrics": {...}
            }
        """
        # Run both strategies
        result_a = await self.unified_rag.search(query, strategy=strategy_a, top_k=top_k)
        result_b = await self.unified_rag.search(query, strategy=strategy_b, top_k=top_k)

        # Compare
        winner = self._determine_winner(result_a, result_b)

        return {
            "query": query,
            "strategy_a": {
                "name": strategy_a,
                "results": result_a["results"],
                "latency_ms": result_a["performance"]["latency_ms"]
            },
            "strategy_b": {
                "name": strategy_b,
                "results": result_b["results"],
                "latency_ms": result_b["performance"]["latency_ms"]
            },
            "winner": winner,
            "metrics": {
                "overlap": self._calculate_overlap(result_a["results"], result_b["results"]),
                "latency_diff_ms": abs(
                    result_a["performance"]["latency_ms"] -
                    result_b["performance"]["latency_ms"]
                )
            }
        }

    def _determine_winner(self, result_a: Dict, result_b: Dict) -> str:
        """
        Determine winner based on simple heuristics.

        In production, you'd use human evaluation or relevance labels.
        """
        # For now, just use latency
        latency_a = result_a["performance"]["latency_ms"]
        latency_b = result_b["performance"]["latency_ms"]

        if abs(latency_a - latency_b) < 100:  # Less than 100ms difference
            return "tie"
        elif latency_a < latency_b:
            return "strategy_a"
        else:
            return "strategy_b"

    def _calculate_overlap(self, results_a: List[Dict], results_b: List[Dict]) -> float:
        """Calculate Jaccard similarity between result sets."""
        ids_a = {doc["id"] for doc in results_a}
        ids_b = {doc["id"] for doc in results_b}

        intersection = len(ids_a & ids_b)
        union = len(ids_a | ids_b)

        return intersection / union if union > 0 else 0.0
```

---

## 7. Visualization & Analytics

### Summary Dashboard Command

Add final command to `comparison_cog.py`:

```python
@commands.command(name="rag_dashboard")
async def rag_dashboard(self, ctx, *, query: str):
    """
    Complete dashboard with all comparisons.

    Usage: !rag_dashboard how does Python's GIL work?
    """
    await ctx.send(f"📊 Generating complete dashboard for: `{query}`\n⏳ This will take 60-90 seconds...")

    # Get recommendation
    recommendation = self.recommender.recommend_strategy(query)

    # Run recommended strategy
    result = await self.unified_rag.search(
        query,
        strategy=recommendation["recommended"],
        top_k=5
    )

    # Create comprehensive embed
    embed = discord.Embed(
        title="📊 Complete RAG Dashboard",
        description=f"**Query:** {query}",
        color=discord.Color.blue()
    )

    # Recommended strategy
    embed.add_field(
        name="🎯 Recommended Strategy",
        value=f"{recommendation['recommended']}\n*{recommendation['reasoning']}*",
        inline=False
    )

    # Top 3 results
    results_text = []
    for idx, doc in enumerate(result["results"][:3], 1):
        results_text.append(f"{idx}. {doc['content'][:80]}...")

    embed.add_field(
        name="📄 Top Results",
        value="\n".join(results_text),
        inline=False
    )

    # Performance
    embed.add_field(
        name="⚡ Performance",
        value=f"Latency: {result['performance']['latency_ms']:.0f}ms",
        inline=True
    )

    # Query analysis
    analysis = recommendation["query_analysis"]
    embed.add_field(
        name="🔍 Query Analysis",
        value=f"Type: {analysis['category']}\nComplexity: {analysis['complexity']}",
        inline=True
    )

    embed.set_footer(text="Use !rag_compare to compare multiple strategies")

    await ctx.send(embed=embed)
```

---

## Summary

### What We Built

1. **Unified RAG Service**: Single interface for all 9+ RAG strategies
2. **Comparison Tools**: Side-by-side strategy comparison
3. **Performance Metrics**: Latency tracking and analysis
4. **Strategy Recommender**: Automatic strategy selection based on query
5. **Interactive Commands**: Discord commands for exploration
6. **A/B Testing**: Framework for comparing strategies
7. **Analytics Dashboard**: Comprehensive query analysis

### Key Commands

| Command | Description |
|---------|-------------|
| `!rag_recommend <query>` | Get strategy recommendation |
| `!rag_compare <query>` | Compare multiple strategies |
| `!rag_overlap <query>` | Analyze result overlap |
| `!rag_profiles` | View all strategy profiles |
| `!rag_dashboard <query>` | Complete analysis dashboard |

### Decision Matrix

Use this to choose the right strategy:

```
Your Priority             → Use This Strategy
═══════════════════════════════════════════════════
Speed (< 100ms)           → vector or keyword
Accuracy (best results)   → rag_fusion or self_rag
Balance                   → hybrid_rerank
Question-answering        → hyde
Exploratory queries       → rag_fusion
Simple factual            → vector
Complex reasoning         → query_expansion_rerank
Keywords important        → hybrid
Vocabulary mismatch       → query_expansion or hyde
Quality assurance         → self_rag
```

---

## Congratulations! 🎉

You've completed the entire Advanced RAG series (Phases 14-17)! You now understand:

✅ Vector vs Keyword vs Hybrid search
✅ Query optimization and expansion
✅ Cross-encoder reranking
✅ HyDE (Hypothetical Document Embeddings)
✅ Self-RAG (Self-Reflective Retrieval)
✅ RAG Fusion (Multi-Query Synthesis)
✅ How to compare and choose strategies
✅ Performance optimization trade-offs

### Next Steps

- Implement evaluation framework (Phase 6.5) to measure which strategy works best
- Build conversational chatbot (Phase 11) using your RAG system
- Add user emulation (Phase 12) and debate analysis (Phase 13)
- Deploy to production (see DEPLOYMENT_GUIDE.md)

### Further Learning

- Read the research papers linked in Phases 14-16
- Experiment with different models (GPT-4, Claude, local LLMs)
- Try domain-specific embeddings
- Explore graph-based RAG (coming soon!)

---

**Happy RAG building! 🚀**
