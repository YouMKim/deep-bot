# Phase 6.5: Strategy Evaluation & Comparison

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Strategy Evaluation & Comparison

### Learning Objectives
- Understand retrieval evaluation metrics (Precision, Recall, MRR)
- Learn how to create test datasets for RAG systems
- Practice A/B testing for chunking strategies
- Measure retrieval quality objectively
- Compare strategies systematically

### Design Principles
- **Quantitative Evaluation**: Use metrics, not gut feeling
- **Test-Driven Development**: Create test queries before optimizing
- **Reproducibility**: Same tests should give same results
- **Comparative Analysis**: Always compare against a baseline

### Why Evaluation Matters

You've implemented multiple chunking strategies, but **which one is best?**

Without evaluation, you're flying blind:
- ❌ "This strategy *feels* better" (subjective)
- ❌ "More chunks = better retrieval" (not always true)
- ❌ "Overlap is good" (depends on your data)

With evaluation:
- ✅ "Strategy A has 85% recall vs Strategy B's 62%" (objective)
- ✅ "Sliding window improved MRR by 23%" (measurable)
- ✅ "Token-aware reduces storage by 40% with <5% quality loss" (trade-off analysis)

---

## Implementation Steps

### Step 6.5.1: Create Test Query Dataset

Create `evaluation/datasets/test_queries.py`:

```python
"""
Test queries for evaluating chunking strategies.

Learning: Good test queries should:
1. Have clear, known correct answers
2. Cover different query types (factual, conceptual, temporal)
3. Test edge cases (multi-message answers, rare terms)
"""

from typing import List, Dict

class TestQueryDataset:
    """
    Dataset of test queries with labeled relevant messages.

    Learning: This is similar to how commercial RAG systems are evaluated.
    You manually label a small set (~10-50 queries) and measure performance.
    """

    def __init__(self):
        self.queries = self._create_test_queries()

    def _create_test_queries(self) -> List[Dict]:
        """
        Create test queries with ground truth.

        Structure:
        {
            "query": "What did Alice say about the project deadline?",
            "relevant_message_ids": ["123", "456"],  # Messages that answer this
            "category": "factual",  # factual, conceptual, temporal
            "difficulty": "easy"  # easy, medium, hard
        }

        How to create:
        1. Read through your Discord messages
        2. Pick 10-20 interesting questions
        3. Manually find which messages answer each question
        4. Label them here
        """

        return [
            {
                "query": "What time is the team meeting?",
                "relevant_message_ids": ["1234567890"],
                "category": "factual",
                "difficulty": "easy",
                "notes": "Direct answer in single message"
            },
            {
                "query": "How did we decide to handle the database migration?",
                "relevant_message_ids": ["1234567891", "1234567892", "1234567893"],
                "category": "conceptual",
                "difficulty": "medium",
                "notes": "Answer spans multiple messages in a conversation"
            },
            {
                "query": "Who suggested using Docker for deployment?",
                "relevant_message_ids": ["1234567894"],
                "category": "factual",
                "difficulty": "easy",
                "notes": "Single message with specific person"
            },
            {
                "query": "What were the main concerns about the API design?",
                "relevant_message_ids": ["1234567895", "1234567896", "1234567897", "1234567898"],
                "category": "conceptual",
                "difficulty": "hard",
                "notes": "Multiple people raised different concerns across conversation"
            },
            {
                "query": "When did we last discuss the budget?",
                "relevant_message_ids": ["1234567899", "1234567900"],
                "category": "temporal",
                "difficulty": "medium",
                "notes": "Requires temporal understanding"
            }
            # Add 5-15 more queries based on YOUR actual messages
        ]

    def get_queries(self, category: str = None, difficulty: str = None) -> List[Dict]:
        """Filter queries by category or difficulty"""
        filtered = self.queries

        if category:
            filtered = [q for q in filtered if q["category"] == category]

        if difficulty:
            filtered = [q for q in filtered if q["difficulty"] == difficulty]

        return filtered

    def get_all_relevant_message_ids(self) -> set:
        """Get all message IDs that are relevant to any query"""
        all_ids = set()
        for query in self.queries:
            all_ids.update(query["relevant_message_ids"])
        return all_ids
```

### Step 6.5.2: Implement Evaluation Metrics

Create `evaluation/metrics.py`:

```python
"""
Evaluation metrics for RAG retrieval quality.

Learning: These metrics are standard in Information Retrieval (IR):
- Precision: % of retrieved docs that are relevant
- Recall: % of relevant docs that are retrieved
- F1: Harmonic mean of precision and recall
- MRR: Mean Reciprocal Rank (position of first relevant result)
- NDCG: Normalized Discounted Cumulative Gain (rewards ranking)
"""

from typing import List, Dict, Set
import logging

class RetrievalEvaluator:
    """
    Evaluate retrieval quality using IR metrics.

    Learning: These metrics help answer:
    - Does the system find the right information? (Recall)
    - Is the retrieved information mostly correct? (Precision)
    - Are the best results ranked first? (MRR, NDCG)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = 5
    ) -> float:
        """
        Precision@K: % of top-K retrieved docs that are relevant.

        Formula: (# relevant in top-K) / K

        Example:
            Retrieved (top-5): [A, B, C, D, E]
            Relevant: {A, C, E, F}
            Precision@5 = 3/5 = 0.60 (60%)

        Learning: High precision = low false positives
        """
        if not retrieved_ids or k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_retrieved / k

    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = 5
    ) -> float:
        """
        Recall@K: % of relevant docs found in top-K.

        Formula: (# relevant in top-K) / (total # relevant)

        Example:
            Retrieved (top-5): [A, B, C, D, E]
            Relevant: {A, C, E, F}
            Recall@5 = 3/4 = 0.75 (75% of relevant docs found)

        Learning: High recall = low false negatives
        """
        if not relevant_ids:
            return 0.0

        if not retrieved_ids:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_retrieved / len(relevant_ids)

    def f1_score(
        self,
        precision: float,
        recall: float
    ) -> float:
        """
        F1 Score: Harmonic mean of precision and recall.

        Formula: 2 * (P * R) / (P + R)

        Learning: F1 balances precision and recall.
        Useful when you care about both equally.
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def mean_reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        MRR: Average of reciprocal ranks of first relevant result.

        Formula: 1 / (rank of first relevant doc)

        Example:
            Retrieved: [A, B, C, D, E]
            Relevant: {C, E}
            First relevant is C at rank 3
            MRR = 1/3 = 0.333

        Learning: MRR rewards putting relevant docs early.
        Good for "I'm feeling lucky" style search.
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def evaluate_query(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate a single query with all metrics.

        Returns:
            {
                "precision@5": 0.60,
                "recall@5": 0.75,
                "f1@5": 0.667,
                "mrr": 0.333
            }
        """
        precision = self.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = self.recall_at_k(retrieved_ids, relevant_ids, k)
        f1 = self.f1_score(precision, recall)
        mrr = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)

        return {
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"f1@{k}": f1,
            "mrr": mrr
        }

    def evaluate_strategy(
        self,
        query_results: List[Dict[str, any]]
    ) -> Dict[str, float]:
        """
        Evaluate a strategy across all test queries.

        Args:
            query_results: List of:
                {
                    "query": "...",
                    "retrieved_ids": ["id1", "id2", ...],
                    "relevant_ids": {"id1", "id3", ...}
                }

        Returns:
            Average metrics across all queries
        """
        if not query_results:
            return {}

        all_metrics = []
        for result in query_results:
            metrics = self.evaluate_query(
                result["retrieved_ids"],
                result["relevant_ids"],
                k=5
            )
            all_metrics.append(metrics)

        # Average all metrics
        avg_metrics = {}
        metric_names = all_metrics[0].keys()
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            avg_metrics[metric_name] = sum(values) / len(values)

        return avg_metrics
```

### Step 6.5.3: Create Strategy Comparison Tool

Create `evaluation/comparison.py`:

```python
"""
Compare chunking strategies using test queries.

Learning: This is how you scientifically choose the best strategy.
"""

from rag import ChunkedMemoryService
from evaluation.metrics import RetrievalEvaluator
from evaluation.datasets.test_queries import TestQueryDataset
from typing import List, Dict
import logging

class StrategyComparator:
    """
    Compare multiple chunking strategies.

    Learning: A/B testing for RAG systems.
    """

    def __init__(
        self,
        chunked_memory: ChunkedMemoryService
    ):
        self.memory = chunked_memory
        self.evaluator = RetrievalEvaluator()
        self.logger = logging.getLogger(__name__)

    def compare_strategies(
        self,
        strategies: List[str],
        test_dataset: TestQueryDataset,
        top_k: int = 5
    ) -> Dict[str, Dict]:
        """
        Compare strategies on test queries.

        Returns:
            {
                "temporal": {
                    "precision@5": 0.65,
                    "recall@5": 0.80,
                    ...
                },
                "sliding_window": {
                    "precision@5": 0.70,
                    "recall@5": 0.85,
                    ...
                },
                ...
            }
        """
        results = {}

        for strategy in strategies:
            self.logger.info(f"Evaluating strategy: {strategy}")
            strategy_results = self._evaluate_strategy(
                strategy,
                test_dataset,
                top_k
            )
            results[strategy] = strategy_results

        return results

    def _evaluate_strategy(
        self,
        strategy: str,
        test_dataset: TestQueryDataset,
        top_k: int
    ) -> Dict[str, float]:
        """Evaluate a single strategy on all test queries"""
        query_results = []

        for test_query in test_dataset.get_queries():
            # Retrieve chunks for this query
            retrieved_chunks = self.memory.search(
                query=test_query["query"],
                strategy=strategy,
                top_k=top_k
            )

            # Extract message IDs from chunks
            retrieved_ids = []
            for chunk in retrieved_chunks:
                # Chunks contain multiple message IDs
                msg_ids = chunk.get("metadata", {}).get("message_ids", [])
                if isinstance(msg_ids, list):
                    retrieved_ids.extend(msg_ids)
                else:
                    retrieved_ids.append(str(msg_ids))

            # Remove duplicates while preserving order
            seen = set()
            unique_retrieved_ids = []
            for msg_id in retrieved_ids:
                if msg_id not in seen:
                    seen.add(msg_id)
                    unique_retrieved_ids.append(msg_id)

            query_results.append({
                "query": test_query["query"],
                "retrieved_ids": unique_retrieved_ids,
                "relevant_ids": set(test_query["relevant_message_ids"])
            })

        # Calculate average metrics
        avg_metrics = self.evaluator.evaluate_strategy(query_results)
        return avg_metrics

    def print_comparison(self, results: Dict[str, Dict]):
        """
        Pretty-print comparison results.

        Example output:

        Strategy Comparison Results:
        ============================

        Strategy: temporal
          Precision@5: 0.65
          Recall@5:    0.80
          F1@5:        0.72
          MRR:         0.55

        Strategy: sliding_window
          Precision@5: 0.70 (+7.7%)
          Recall@5:    0.85 (+6.3%)
          F1@5:        0.77 (+6.9%)
          MRR:         0.62 (+12.7%)

        WINNER: sliding_window (highest F1@5)
        """
        if not results:
            print("No results to display")
            return

        # Find baseline (first strategy)
        strategies = list(results.keys())
        baseline = strategies[0]
        baseline_metrics = results[baseline]

        print("\n" + "="*60)
        print("Strategy Comparison Results")
        print("="*60)

        for strategy, metrics in results.items():
            print(f"\nStrategy: {strategy}")
            print("-" * 40)

            for metric_name, value in metrics.items():
                if strategy == baseline:
                    print(f"  {metric_name:15s}: {value:.3f}")
                else:
                    baseline_value = baseline_metrics[metric_name]
                    if baseline_value > 0:
                        pct_change = ((value - baseline_value) / baseline_value) * 100
                        sign = "+" if pct_change >= 0 else ""
                        print(f"  {metric_name:15s}: {value:.3f} ({sign}{pct_change:+.1f}%)")
                    else:
                        print(f"  {metric_name:15s}: {value:.3f}")

        # Find winner (highest F1)
        winner = max(results.items(), key=lambda x: x[1].get("f1@5", 0))
        print(f"\n{'='*60}")
        print(f"WINNER: {winner[0]} (highest F1@5: {winner[1].get('f1@5', 0):.3f})")
        print(f"{'='*60}\n")
```

### Step 6.5.4: Create Evaluation Bot Command

Add to `cogs/admin.py`:

```python
@commands.command(name='evaluate_strategies')
@commands.is_owner()
async def evaluate_strategies(self, ctx, top_k: int = 5):
    """
    Evaluate all chunking strategies on test queries.

    Usage: !evaluate_strategies [top_k]

    Learning: This shows you which strategy works best for your data.
    """
    if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
        await ctx.send("🚫 **Access Denied!**")
        return

    try:
        status_msg = await ctx.send("📊 Evaluating strategies...")

        # Initialize services
        from retrieval import VectorStoreFactory
        from embedding import EmbeddingServiceFactory
        from rag import ChunkedMemoryService
        from evaluation.comparison import StrategyComparator
        from evaluation.datasets.test_queries import TestQueryDataset

        vector_store = VectorStoreFactory.create()
        embedding_provider = EmbeddingServiceFactory.create()
        chunked_memory = ChunkedMemoryService(vector_store, embedding_provider)

        comparator = StrategyComparator(chunked_memory)
        test_dataset = TestQueryDataset()

        # Get available strategies
        stats = chunked_memory.get_strategy_stats()
        strategies = [s for s, count in stats.items() if count > 0]

        if not strategies:
            await status_msg.edit(content="❌ No strategies found. Run !chunk_channel first.")
            return

        await status_msg.edit(
            content=f"📊 Evaluating {len(strategies)} strategies on "
                   f"{len(test_dataset.get_queries())} test queries..."
        )

        # Run comparison
        results = comparator.compare_strategies(strategies, test_dataset, top_k)

        # Format results as Discord embed
        embed = discord.Embed(
            title="📊 Strategy Evaluation Results",
            description=f"Tested on {len(test_dataset.get_queries())} queries, top-{top_k} results",
            color=discord.Color.blue()
        )

        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1].get("f1@5", 0))

        for strategy, metrics in results.items():
            is_best = (strategy == best_strategy[0])
            emoji = "🏆 " if is_best else ""

            value = (
                f"Precision@{top_k}: **{metrics.get(f'precision@{top_k}', 0):.3f}**\n"
                f"Recall@{top_k}: **{metrics.get(f'recall@{top_k}', 0):.3f}**\n"
                f"F1@{top_k}: **{metrics.get(f'f1@{top_k}', 0):.3f}**\n"
                f"MRR: **{metrics.get('mrr', 0):.3f}**"
            )

            embed.add_field(
                name=f"{emoji}{strategy.title()}",
                value=value,
                inline=True
            )

        embed.set_footer(text=f"Winner: {best_strategy[0]} (highest F1@{top_k})")

        await status_msg.edit(content="", embed=embed)

    except Exception as e:
        self.logger.error(f"Error evaluating strategies: {e}", exc_info=True)
        await ctx.send(f"❌ Error: {e}")
```

---

## Usage Example

```python
# 1. Create test dataset (one-time setup)
test_dataset = TestQueryDataset()
print(f"Created {len(test_dataset.get_queries())} test queries")

# 2. Run evaluation
from evaluation.comparison import StrategyComparator

comparator = StrategyComparator(chunked_memory)
results = comparator.compare_strategies(
    strategies=["temporal", "sliding_window", "token_aware"],
    test_dataset=test_dataset,
    top_k=5
)

# 3. Print results
comparator.print_comparison(results)

# Example output:
# Strategy Comparison Results:
# ============================
#
# Strategy: temporal
#   precision@5:    0.650
#   recall@5:       0.800
#   f1@5:           0.717
#   mrr:            0.550
#
# Strategy: sliding_window
#   precision@5:    0.700 (+7.7%)
#   recall@5:       0.850 (+6.3%)
#   f1@5:           0.768 (+7.1%)
#   mrr:            0.620 (+12.7%)
#
# WINNER: sliding_window (highest F1@5: 0.768)
```

---

## Common Pitfalls - Phase 6.5

1. **Too few test queries**: Need at least 10-20 for meaningful results
2. **Biased test set**: Don't only test easy queries
3. **Overfitting**: Don't optimize strategy based on test set, then test on same set
4. **Ignoring variance**: Run multiple times to check consistency
5. **Wrong message IDs**: Ensure labeled message IDs actually exist in database

## Debugging Tips - Phase 6.5

- **Verify test queries**: Manually check that labeled messages do answer the query
- **Check retrieval**: Print retrieved chunks to see what's being found
- **Test one query**: Debug single query before running full evaluation
- **Compare manually**: For a few queries, manually verify metrics are correct

## Key Insights - Phase 6.5

**What the metrics tell you:**

- **High Precision, Low Recall**: System is accurate but misses relevant docs
  - Fix: Increase top_k, add overlap, use broader chunking

- **Low Precision, High Recall**: System finds relevant docs but also junk
  - Fix: Better embedding model, metadata filtering, reranking

- **Low MRR**: Relevant docs ranked low (buried in results)
  - Fix: Reranking, better embedding model, query expansion

- **All metrics low**: Strategy or embedding doesn't work for your data
  - Fix: Try completely different approach (semantic chunking, hybrid search)

---

## Next Steps

After evaluating strategies:
1. **Choose best strategy** based on F1@5 or metric that matters most to you
2. **Set as default** in your bot commands
3. **Monitor in production**: Track which queries fail, add to test set
4. **Iterate**: Add more test queries, try new strategies, re-evaluate

**Remember**: The "best" strategy depends on your data and use case. Always evaluate!

