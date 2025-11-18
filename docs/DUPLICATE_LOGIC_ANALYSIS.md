# Duplicate Logic Analysis
*Generated: 2025-11-18*

## Executive Summary

After analyzing the codebase, I found **7 key areas** with duplicate or near-duplicate logic that could be consolidated. Most are in `storage/chunked_memory.py`, `rag/hybrid_search.py`, and `bot/cogs/rag.py`.

**Impact:** Medium priority - not critical but would improve maintainability and reduce bugs.

---

## 1. Collection Name Generation (Duplicated 4+ times)

### Current State

**Pattern appears in:**
- `storage/chunked_memory.py` line 229, 331, 421, 565
- Multiple other files

```python
# Repeated 4 times in chunked_memory.py alone:
collection_name = f"discord_chunks_{strategy_name}"
collection_name = f"discord_chunks_{strategy_value}"
```

### Issue
- Magic string format repeated everywhere
- If naming convention changes, must update all locations
- Error-prone

### Recommended Fix

**Create a utility method:**

```python
# storage/chunked_memory.py
class ChunkedMemoryService:
    @staticmethod
    def _get_collection_name(strategy: str) -> str:
        """Get collection name for a strategy."""
        return f"discord_chunks_{strategy}"

    # Then use everywhere:
    collection_name = self._get_collection_name(strategy_value)
```

**Effort:** 10 minutes
**Impact:** Low risk, high maintainability gain

---

## 2. Strategy Value Extraction (Duplicated 2 times)

### Current State

**Lines 330 and 420 in `storage/chunked_memory.py`:**

```python
# In search() method (line 330)
strategy_value = (strategy or ChunkStrategy(self.active_strategy)).value

# In search_bm25() method (line 420)
strategy_value = (strategy or ChunkStrategy(self.active_strategy)).value
```

### Issue
- Exact same logic duplicated
- If default strategy logic changes, must update both places

### Recommended Fix

```python
class ChunkedMemoryService:
    def _resolve_strategy(self, strategy: Optional[ChunkStrategy]) -> str:
        """Resolve strategy to its string value, using active strategy as default."""
        return (strategy or ChunkStrategy(self.active_strategy)).value

    # Then use in both methods:
    def search(self, query: str, strategy: Optional[ChunkStrategy] = None, ...):
        strategy_value = self._resolve_strategy(strategy)
        collection_name = self._get_collection_name(strategy_value)
        ...

    def search_bm25(self, query: str, strategy: Optional[ChunkStrategy] = None, ...):
        strategy_value = self._resolve_strategy(strategy)
        collection_name = self._get_collection_name(strategy_value)
        ...
```

**Effort:** 5 minutes
**Impact:** Medium maintainability gain

---

## 3. Result Formatting Logic (Duplicated 2 times)

### Current State

**Very similar logic in `search()` and `search_bm25()`:**

```python
# In search() method (lines 391-397)
formatted_results.append({
    "content": document,
    "metadata": metadata,
    "similarity": similarity,
})

# In search_bm25() method (lines 482-487)
results.append({
    "content": document,
    "metadata": metadata,
    "bm25_score": float(score),
    "similarity": float(score) / 100.0  # Normalize for consistency
})
```

### Issue
- Near-duplicate result formatting
- Inconsistent key names (`bm25_score` vs just using `similarity`)
- Hard to keep formats aligned

### Recommended Fix

**Option 1: Extract formatter (simple)**

```python
def _format_search_result(
    self,
    document: str,
    metadata: Dict,
    score: float,
    score_type: str = "similarity"
) -> Dict:
    """Format a search result consistently."""
    result = {
        "content": document,
        "metadata": metadata,
        "similarity": score,
    }

    # Add score-specific fields
    if score_type == "bm25":
        result["bm25_score"] = score
    elif score_type == "vector":
        result["vector_score"] = score

    return result
```

**Option 2: Just standardize (better)**

```python
# In both methods, use the same format:
result = {
    "content": document,
    "metadata": metadata,
    "similarity": similarity,  # Always use this key
    # Optional: add source type
    "search_type": "vector" or "bm25" or "hybrid"
}
```

**Effort:** 15 minutes
**Impact:** High - makes result handling more consistent

---

## 4. Fetch K Calculation (Duplicated 3 times)

### Current State

**Pattern appears in:**
- `storage/chunked_memory.py` line 342 (search)
- `storage/chunked_memory.py` line 528 (search_hybrid)
- `rag/pipeline.py` line 151 (for reranking)

```python
# In search() - conditional based on filtering
fetch_k = top_k * 3 if needs_filtering else top_k

# In search_hybrid() - always 3x
fetch_k = top_k * 3

# In RAG pipeline - for reranking
fetch_k = config.top_k * 3 if config.use_reranking else config.top_k
```

### Issue
- Magic number `3` repeated
- Different logic in different places (conditional vs always)
- No explanation WHY 3x

### Recommended Fix

```python
# config.py
class Config:
    # Fetch multiplier for filtering/reranking
    # We fetch 3x the requested results to account for:
    # - Author filtering (may remove many results)
    # - Reranking (need more candidates)
    # - Hybrid search fusion (deduplication)
    SEARCH_FETCH_MULTIPLIER: int = 3

# storage/chunked_memory.py
def _calculate_fetch_k(
    self,
    top_k: int,
    needs_filtering: bool = False,
    needs_reranking: bool = False
) -> int:
    """
    Calculate how many results to fetch from vector store.

    Fetches more than requested to account for:
    - Filtering (blacklist, author whitelist)
    - Reranking (cross-encoder needs candidates)
    - Hybrid search fusion (deduplication)
    """
    if needs_filtering or needs_reranking:
        return top_k * self.config.SEARCH_FETCH_MULTIPLIER
    return top_k

# Then use:
fetch_k = self._calculate_fetch_k(top_k, needs_filtering=True)
```

**Effort:** 20 minutes
**Impact:** Medium - better self-documentation

---

## 5. Duplicate RRF Implementation (CRITICAL - 2 implementations!)

### Current State

**TWO separate RRF implementations in `rag/hybrid_search.py`:**

1. **`reciprocal_rank_fusion()` function (lines 7-45)** - Generic RRF
2. **`HybridSearchService.hybrid_search()` method (lines 56-104)** - Weighted RRF

Both do essentially the same thing with slightly different logic!

```python
# Function version (lines 7-45)
def reciprocal_rank_fusion(ranked_lists: List[List[Dict]], ...) -> List[Dict]:
    # Generic RRF for multiple lists
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k_constant + rank)
    ...

# Method version (lines 74-85)
class HybridSearchService:
    def hybrid_search(self, ...) -> List[Dict]:
        # Weighted RRF for BM25 + vector
        for rank, result in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = bm25_weight / (k + rank)

        for rank, result in enumerate(vector_results, start=1):
            rrf_scores[doc_id] += vector_weight / (k + rank)
        ...
```

### Issue
- **DUPLICATE IMPLEMENTATIONS** of same algorithm
- Function version is more generic
- Method version adds weights but reinvents the wheel
- Confusing which one to use

### Recommended Fix

**Consolidate into ONE implementation:**

```python
# rag/hybrid_search.py
def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    top_k: int = 10,
    k_constant: int = 60,
    weights: Optional[List[float]] = None
) -> List[Dict]:
    """
    Generic RRF implementation with optional weights.

    Args:
        ranked_lists: Multiple ranked lists to fuse
        top_k: Number of results to return
        k_constant: RRF constant (default: 60)
        weights: Optional weights for each list (default: equal weights)

    Returns:
        Fused and re-ranked results
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists length")

    all_docs = {}
    rrf_scores = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc.get('metadata', {}).get('first_message_id', f"doc_{id(doc)}")

            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                rrf_scores[doc_id] = 0

            # RRF formula with weight
            rrf_scores[doc_id] += weight / (k_constant + rank)

    # Sort and return top-k
    sorted_docs = sorted(
        [(doc_id, score) for doc_id, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for doc_id, score in sorted_docs[:top_k]:
        doc = all_docs[doc_id]
        doc['rrf_score'] = score
        doc['fusion_rank'] = len(results) + 1
        results.append(doc)

    return results


class HybridSearchService:
    """Combines BM25 and vector search."""

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
        Merge BM25 and vector search using RRF.

        Now just a thin wrapper around the generic RRF function!
        """
        # Use the generic RRF function
        results = reciprocal_rank_fusion(
            ranked_lists=[bm25_results, vector_results],
            top_k=top_k,
            k_constant=60,
            weights=[bm25_weight, vector_weight]
        )

        self.logger.info(
            f"Hybrid search: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"→ {len(results)} fused results"
        )

        return results
```

**Effort:** 30 minutes
**Impact:** HIGH - eliminates duplicate algorithm implementation

---

## 6. RAGConfig Creation (Duplicated 2 times)

### Current State

**In `bot/cogs/rag.py`, both `ask()` and `ask_hybrid()` create nearly identical RAGConfig:**

```python
# In ask() command (lines 49-57)
config = RAGConfig(
    top_k=self.config.RAG_DEFAULT_TOP_K,
    similarity_threshold=self.config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
    max_context_tokens=self.config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
    temperature=self.config.RAG_DEFAULT_TEMPERATURE,
    strategy=self.config.RAG_DEFAULT_STRATEGY,
    show_sources=False,
    filter_authors=mentioned_users if mentioned_users else None,
)

# In ask_hybrid() command (lines 110-120)
config = RAGConfig(
    top_k=self.config.RAG_DEFAULT_TOP_K,
    similarity_threshold=self.config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
    max_context_tokens=self.config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
    temperature=self.config.RAG_DEFAULT_TEMPERATURE,
    strategy=self.config.RAG_DEFAULT_STRATEGY,
    use_hybrid_search=True,   # ONLY DIFFERENCE!
    bm25_weight=0.5,           # ONLY DIFFERENCE!
    vector_weight=0.5,         # ONLY DIFFERENCE!
    filter_authors=mentioned_users if mentioned_users else None,
)
```

### Issue
- 90% identical code
- If defaults change, must update both places
- Easy to introduce inconsistencies

### Recommended Fix

```python
class RAG(commands.Cog):

    def _create_base_config(
        self,
        filter_authors: Optional[List[str]] = None,
        **overrides
    ) -> RAGConfig:
        """Create RAGConfig with defaults from Config."""
        config_dict = {
            'top_k': self.config.RAG_DEFAULT_TOP_K,
            'similarity_threshold': self.config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
            'max_context_tokens': self.config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
            'temperature': self.config.RAG_DEFAULT_TEMPERATURE,
            'strategy': self.config.RAG_DEFAULT_STRATEGY,
            'filter_authors': filter_authors,
        }
        # Override with any custom settings
        config_dict.update(overrides)
        return RAGConfig(**config_dict)

    @commands.command(name='ask')
    async def ask(self, ctx, *, question: str):
        ...
        config = self._create_base_config(
            filter_authors=mentioned_users,
            show_sources=False
        )
        ...

    @commands.command(name='ask_hybrid')
    async def ask_hybrid(self, ctx, *, question: str):
        ...
        config = self._create_base_config(
            filter_authors=mentioned_users,
            use_hybrid_search=True,
            bm25_weight=0.5,
            vector_weight=0.5
        )
        ...
```

**Effort:** 15 minutes
**Impact:** Medium - easier to add new commands with different configs

---

## 7. Cooldown Error Handlers (Duplicated 2 times)

### Current State

**In `bot/cogs/rag.py`, nearly identical error handlers:**

```python
# ask.error (lines 196-217)
@ask.error
async def ask_error(self, ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        minutes, seconds = divmod(int(error.retry_after), 60)
        embed = discord.Embed(...)  # Custom message for ask
        await ctx.send(embed=embed)
    else:
        raise error

# ask_hybrid.error (lines 219-238)
@ask_hybrid.error
async def ask_hybrid_error(self, ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        minutes, seconds = divmod(int(error.retry_after), 60)
        embed = discord.Embed(...)  # Custom message for ask_hybrid
        await ctx.send(embed=embed)
    else:
        raise error
```

### Issue
- Identical structure, only message differs
- Time formatting logic duplicated

### Recommended Fix

```python
class RAG(commands.Cog):

    async def _handle_cooldown_error(
        self,
        ctx,
        error,
        command_name: str,
        rate_limit_msg: str
    ):
        """Generic cooldown error handler."""
        if isinstance(error, commands.CommandOnCooldown):
            minutes, seconds = divmod(int(error.retry_after), 60)

            embed = discord.Embed(
                title="⏰ Rate Limit Reached",
                description=(
                    f"{rate_limit_msg}\n\n"
                    f"Please wait **{minutes}m {seconds}s** before trying again."
                ),
                color=discord.Color.orange()
            )
            embed.set_footer(text=f"Limit: {error.cooldown.rate} per {error.cooldown.per}s")

            await ctx.send(embed=embed)
        else:
            raise error

    @ask.error
    async def ask_error(self, ctx, error):
        await self._handle_cooldown_error(
            ctx, error,
            command_name="ask",
            rate_limit_msg="You're asking questions too quickly! This helps prevent API cost overruns."
        )

    @ask_hybrid.error
    async def ask_hybrid_error(self, ctx, error):
        await self._handle_cooldown_error(
            ctx, error,
            command_name="ask_hybrid",
            rate_limit_msg="Hybrid search is expensive! Consider using regular !ask for faster queries."
        )
```

**Effort:** 20 minutes
**Impact:** Medium - easier to add more commands with consistent error handling

---

## Summary & Prioritization

### High Priority (Do These First)

1. **[CRITICAL] Consolidate RRF Implementation** (30 min)
   - Two separate implementations of same algorithm
   - Confusing and error-prone
   - High impact on code quality

2. **Standardize Result Formatting** (15 min)
   - Makes all search methods return consistent formats
   - Easier to work with results downstream

### Medium Priority (Nice to Have)

3. **Extract Collection Name Generation** (10 min)
   - Low effort, decent maintainability gain

4. **Extract Strategy Resolution** (5 min)
   - Very quick win

5. **Consolidate RAGConfig Creation** (15 min)
   - Makes adding new commands easier

6. **Extract Cooldown Error Handler** (20 min)
   - Clean up error handling

### Low Priority (Optional)

7. **Extract Fetch K Calculation** (20 min)
   - Mostly for documentation
   - Current code works fine

---

## Estimated Total Effort

- **High Priority:** 45 minutes
- **Medium Priority:** 50 minutes
- **Low Priority:** 20 minutes
- **Total:** ~2 hours for all consolidations

---

## Implementation Order

**Recommended sequence:**

1. Fix RRF duplication (hybrid_search.py) - 30 min
2. Standardize result formats (chunked_memory.py) - 15 min
3. Extract collection name helper (chunked_memory.py) - 10 min
4. Extract strategy resolver (chunked_memory.py) - 5 min
5. Consolidate RAGConfig creation (rag.py) - 15 min
6. Extract cooldown handler (rag.py) - 20 min
7. Extract fetch_k calculator (chunked_memory.py) - 20 min

**All changes are low-risk** - they don't change behavior, just reorganize code.

---

## Testing Strategy

For each consolidation:

1. **Before:** Run existing tests - all should pass
2. **During:** Extract logic to helper method
3. **After:** Run tests again - all should still pass
4. **Verify:** No behavior change, just cleaner code

No new tests needed - existing tests validate behavior doesn't change.

---

## Benefits After Consolidation

✅ **Easier maintenance** - Change logic in one place
✅ **Fewer bugs** - No copy-paste mistakes
✅ **Better readability** - Clear helper methods with good names
✅ **Self-documenting** - Method names explain what code does
✅ **Easier testing** - Can test helpers in isolation

---

*End of Analysis*
