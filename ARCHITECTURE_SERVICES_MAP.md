# Architecture & Services Map - Phases 8-15

**Generated:** 2025-11-07
**Purpose:** Map all services from phases 8-15 and show how they integrate with the core architecture

---

## Executive Summary

Phases 8-15 define **advanced services** that sit on top of your core RAG infrastructure (Phases 1-7). Here's what you need to know:

✅ **Good news:** Your recent work (Phases 4, 6.8, 19) provides the **foundation** these services need
⚠️ **Action needed:** Some services need updates to use the new domain-based architecture
🎯 **Integration points:** Clear dependencies mapped below

---

## Service Architecture Map

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER-FACING FEATURES                        │
│  (Phases 8-15: Built on top of core RAG infrastructure)         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 8: Summary Enhancement      (Caching, Performance)       │
│  Phase 8.5: Engagement Features    (Leaderboards, Bookmarks)    │
│  Phase 9: Configuration & Polish   (Config validation)          │
│  Phase 10: RAG Query Pipeline      (Complete RAG queries)       │
│  Phase 10.5: Smart Context         (Thread detection)           │
│  Phase 11: Conversational Chatbot  (Multi-turn memory)          │
│  Phase 12: User Emulation          (Style analysis)             │
│  Phase 13: Debate Analysis         (Argument analysis)          │
│  Phase 14: Hybrid Search           (Vector + BM25)              │
│  Phase 15: Reranking & Optimization (Cross-encoders)            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CORE RAG INFRASTRUCTURE                         │
│      (Phases 1-7: Your foundation - recently updated)           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: storage/messages.py       (SQLite storage)            │
│  Phase 2: storage/loader.py         (Message fetching)          │
│  Phase 3: ai/embedding.py           (Embeddings)                │
│  Phase 4: chunking/service.py       (NEW: Production ready!)    │
│  Phase 5: storage/vectors/          (Vector stores)             │
│  Phase 6: storage/chunked_memory.py (Multi-strategy)            │
│  Phase 6.8: storage/sync_tracker.py (NEW: Incremental sync!)    │
│  Phase 7: bot/cogs/                 (Bot commands)              │
│  Phase 19: bot/tasks/               (NEW: Background tasks!)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase-by-Phase Service Breakdown

### Phase 8: Summary Enhancement
**Status:** ✅ Compatible with current architecture
**Updates needed:** Minor

#### Services Defined:
```python
# File: bot/cogs/summary.py
class Summary(commands.Cog):
    async def _fetch_messages_with_fallback(...)
        # Uses: storage/messages.py (Phase 1) ✅
        # Strategy: DB-first with Discord API fallback
```

#### Integration Points:
- **Dependencies:**
  - `storage/messages.py` - MessageStorage ✅ Already compatible
  - `config.py` - SUMMARY_USE_STORED_MESSAGES ✅ Already defined in Phase 9

- **Usage:**
  - Check if messages are in DB (from Phase 2 loader)
  - If gaps detected, fetch from Discord API
  - Cache-first pattern improves performance

#### Action Items:
- ✅ No changes needed - already uses your Phase 1 MessageStorage
- 💡 Consider: Add Phase 6.8 checkpoint tracking to detect gaps better

---

### Phase 8.5: Engagement Features (Leaderboards & Bookmarking)
**Status:** ✅ Compatible
**Updates needed:** None

#### Services Defined:

```python
# File: ai/tracker.py (Enhanced)
class UserAITracker:
    def get_leaderboard(category, limit) -> List[Tuple]
    def get_user_rank(user, category) -> int
    def get_global_stats() -> Dict
    # Storage: data/user_ai_stats.json

# File: storage/bookmarks.py (NEW)
class BookmarkDatabase:
    def add_bookmark(user_id, message_id, ...) -> int
    def get_user_bookmarks(user_id, tag=None) -> List[Dict]
    def search_bookmarks(user_id, query) -> List[Dict]
    # Storage: data/bookmarks.db (SQLite)

# File: bot/cogs/bookmarks.py (NEW)
class Bookmarks(commands.Cog):
    @commands.command(name='bookmark')
    @commands.command(name='bookmarks')
    @commands.command(name='bookmark_search')
```

#### Integration Points:
- **Dependencies:**
  - ai/tracker.py (Phase 7.5) - tracks AI usage, now adds leaderboards
  - New SQLite database: `data/bookmarks.db`

- **Independent:** No dependencies on RAG/chunking systems
- **Optional:** Gamification layer on top of core bot

#### Action Items:
- ✅ Ready to implement as-is
- 💡 Consider: Link bookmarks with RAG - bookmark chunks, not just messages

---

### Phase 9: Configuration & Polish
**Status:** ✅ Compatible
**Updates needed:** ⚠️ Needs sync with Phase 4 updates

#### Services Defined:

```python
# File: config.py (Enhanced)
class Config:
    # Embedding Configuration
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_MAX_TOKENS: int = 512

    # Chunking Configuration (Phase 4)
    CHUNKING_TEMPORAL_WINDOW: int = 300
    CHUNKING_CONVERSATION_GAP: int = 1800
    CHUNKING_WINDOW_SIZE: int = 10
    CHUNKING_MAX_TOKENS: int = 512  # ✅ Already in Phase 4!
    CHUNKING_MIN_CHUNK_SIZE: int = 3  # ✅ Already in Phase 4!

    # RAG Configuration (NEW in Phase 9)
    RAG_TOP_K: int = 5
    RAG_MIN_SIMILARITY: float = 0.7
    RAG_DEFAULT_STRATEGY: str = "token_aware"  # ✅ Matches Phase 4!
    RAG_MAX_CONTEXT_TOKENS: int = 2000

    @classmethod
    def validate(cls) -> bool:
        # Validation logic
```

#### Integration Points:
- **Dependencies:** ALL phases use this for configuration
- **Critical:** Central config validation

#### Action Items:
- ✅ Phase 4 config already added: CHUNKING_MAX_TOKENS, CHUNKING_MIN_CHUNK_SIZE
- ✅ Phase 9 defines RAG_* configs used by Phase 10
- ⚠️ **TODO:** Add Phase 6.8 sync configs:
  ```python
  # Add to config.py:
  SYNC_AUTO_INTERVAL_MINUTES: int = 30  # From Phase 19
  SYNC_BATCH_SIZE: int = 100
  ```

---

### Phase 10: RAG Query Pipeline
**Status:** ✅ Core compatible
**Updates needed:** ⚠️ Integrate with Phase 6.8 incremental sync

#### Services Defined:

```python
# File: rag/pipeline.py (NEW)
class RAGQueryService:
    def __init__(chunked_memory, ai_service):
        self.memory = chunked_memory  # Uses Phase 6!
        self.ai = ai_service

    async def query(
        user_question,
        strategy=None,  # ✅ Supports Phase 4 strategies!
        top_k=None,
        min_similarity=None
    ) -> Dict:
        # 1. Retrieve chunks (uses Phase 6 ChunkedMemoryService)
        # 2. Filter by similarity
        # 3. Build context
        # 4. Generate answer with LLM
        # 5. Format sources

# File: bot/cogs/rag.py (NEW)
class RAGCommands(commands.Cog):
    @commands.command(name='ask')
    async def ask(ctx, *, question: str)
        # User-facing RAG query command
```

#### Integration Points:
- **Dependencies:**
  - `storage/chunked_memory.py` (Phase 6) ✅ Compatible
  - `ai/service.py` (Phase 3/7.5) ✅ Compatible
  - `config.py` (Phase 9) ✅ Uses RAG_* configs

- **Used by:**
  - Phase 11 (Conversational RAG)
  - Phase 13 (Debate Analysis - fact checking)

#### Action Items:
- ✅ Phase 4 token-aware chunking ensures context fits in RAG_MAX_CONTEXT_TOKENS
- ✅ Phase 6.8 ensures chunks are up-to-date via incremental sync
- 💡 **Improvement:** Add cache warming using Phase 7.5 cache:
  ```python
  # In RAG query service:
  from utils.cache import SmartCache
  self.query_cache = SmartCache()
  ```

---

### Phase 10.5: Smart Context Building
**Status:** ✅ Compatible
**Updates needed:** ⚠️ Update to use Phase 4 improvements

#### Services Defined:

```python
# File: rag/thread_detector.py (NEW)
class ConversationThread:
    def __init__(messages: List[Dict])
    def get_relevance_score(query_author=None) -> float

class ThreadDetector:
    def detect_threads(
        documents: List[Dict],
        query_author=None
    ) -> List[ConversationThread]
    # Algorithm:
    # 1. Sort by timestamp
    # 2. Group by time gaps (300s threshold)
    # 3. Split long threads (3600s max)
    # 4. Rank by relevance

# File: rag/context_builder.py (NEW)
class SmartContextBuilder:
    def __init__(max_tokens=2000, model="gpt-3.5-turbo"):
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.thread_detector = ThreadDetector()

    def build_context(
        query,
        retrieved_docs,
        query_author=None
    ) -> str:
        # 1. Detect threads
        # 2. Deduplicate
        # 3. Format chronologically
        # 4. Respect token limits
```

#### Integration Points:
- **Dependencies:**
  - tiktoken (same as Phase 4!) ✅
  - Retrieved documents from Phase 10 RAG

- **Enhances:** Phase 10 RAG by improving context quality

#### Action Items:
- ✅ Already uses tiktoken (consistent with Phase 4)
- ⚠️ **TODO:** Update to use Phase 4's token counting:
  ```python
  # In context_builder.py, use ChunkingService's count_tokens():
  from chunking.service import ChunkingService
  chunking_service = ChunkingService()
  token_count = chunking_service.count_tokens(text, model)
  ```
- 💡 **Improvement:** Cache thread detection results (expensive operation)

---

### Phase 11: Conversational Chatbot with Memory
**Status:** ✅ Compatible
**Updates needed:** None

#### Services Defined:

```python
# File: rag/conversation_memory.py (NEW)
class ConversationMemory:
    def __init__(max_history=10, memory_timeout=3600):
        self.conversations: Dict[str, Dict] = {}
        # Storage: in-memory (user_id:channel_id -> messages)

    def add_message(channel_id, user_id, role, content)
    def get_history(channel_id, user_id, limit=None) -> List[Dict]
    def get_conversation_summary(channel_id, user_id) -> str

# File: rag/conversational.py (NEW)
class ConversationalRAGService:
    def __init__(rag_service, conversation_memory, ai_service):
        self.rag = rag_service  # Uses Phase 10!
        self.memory = conversation_memory

    async def chat(
        user_message,
        channel_id,
        user_id,
        use_rag=True
    ) -> Dict:
        # 1. Add to conversation history
        # 2. Decide if RAG needed (keyword detection)
        # 3. Retrieve from RAG if needed
        # 4. Build prompt (conversation + RAG)
        # 5. Generate response
        # 6. Save to history

# File: bot/cogs/chatbot.py (NEW)
class Chatbot(commands.Cog):
    @commands.command(name='chat')
    @commands.command(name='clear_chat')
```

#### Integration Points:
- **Dependencies:**
  - Phase 10 RAG (for knowledge retrieval) ✅
  - Phase 3 AI service (for generation) ✅
  - In-memory storage (ephemeral, timeout-based)

- **Used by:** Phase 12 (User Emulation)

#### Action Items:
- ✅ Ready to implement
- 💡 **Improvement:** Persist conversations to SQLite instead of in-memory:
  ```python
  # Option: Use storage/messages.py pattern
  # Create: storage/conversations.py
  class ConversationStorage:
      def __init__(db_path="data/conversations.db")
  ```

---

### Phase 12: User Emulation Mode
**Status:** ✅ Compatible
**Updates needed:** ⚠️ Integrate with Phase 1 storage

#### Services Defined:

```python
# File: ai/user_style_analyzer.py (NEW)
class UserStyleAnalyzer:
    def analyze_user_style(messages: List[Dict]) -> Dict:
        # Analyzes:
        # - avg_message_length
        # - vocabulary (top 50 words)
        # - common_phrases ("imo", "tbh")
        # - emoji_usage
        # - punctuation_style (casual/formal)
        # - formality_score (0-1)
        # - signature_phrases

# File: ai/user_emulation.py (NEW)
class UserEmulationService:
    def __init__(message_storage, chunked_memory, ai_service):
        self.storage = message_storage  # Uses Phase 1! ✅
        self.memory = chunked_memory    # Uses Phase 6! ✅
        self.style_analyzer = UserStyleAnalyzer()

    async def emulate_response(
        target_user_id,
        target_user_name,
        context
    ) -> Dict:
        # 1. Get user's messages from storage
        # 2. Analyze style
        # 3. Find relevant messages via RAG
        # 4. Generate in user's style

# File: bot/cogs/chatbot.py (Enhanced)
@commands.command(name='emulate')
@commands.command(name='analyze_user')
```

#### Integration Points:
- **Dependencies:**
  - `storage/messages.py` (Phase 1) ✅ Gets user messages
  - `storage/chunked_memory.py` (Phase 6) ✅ User-filtered RAG
  - `ai/service.py` (Phase 3) ✅ Style transfer generation

- **Ethical:** Requires EMULATION_ENABLED config flag

#### Action Items:
- ✅ Already uses Phase 1 MessageStorage
- ⚠️ **TODO:** Implement user filtering in RAG queries:
  ```python
  # In ChunkedMemoryService, add:
  def search(query, strategy, top_k, filter_user_id=None):
      # Filter chunks by author metadata
  ```
- 💡 **Improvement:** Cache style profiles (expensive to analyze)

---

### Phase 13: Debate & Rhetoric Analyzer
**Status:** ✅ Compatible
**Updates needed:** None

#### Services Defined:

```python
# File: ai/argument_analyzer.py (NEW)
class ArgumentAnalyzer:
    def analyze_argument(text: str) -> Dict:
        # Extracts:
        # - main_claim
        # - supporting_reasons
        # - evidence
        # - fallacies (appeal to popularity, ad hominem, etc.)
        # - qualifiers ("should", "must")
        # - rebuttals

    def _detect_fallacies(text) -> List[Dict]:
        # Pattern matching for 8 common fallacies

# File: ai/debate_analyzer.py (NEW)
class DebateAnalyzerService:
    def __init__(rag_service, ai_service):
        self.rag = rag_service  # Uses Phase 10 for fact-checking! ✅
        self.argument_analyzer = ArgumentAnalyzer()

    async def analyze_statement(statement, context=None) -> Dict:
        # 1. Analyze argument structure
        # 2. Fact-check using RAG (chat history)
        # 3. Identify strengths
        # 4. Identify weaknesses
        # 5. Generate suggestions (via LLM)
        # 6. Generate revised argument

# File: bot/cogs/chatbot.py (Enhanced)
@commands.command(name='analyze_debate')
@commands.command(name='compare_arguments')
```

#### Integration Points:
- **Dependencies:**
  - Phase 10 RAG (for fact-checking claims against chat history) ✅
  - Phase 3 AI service (for generating suggestions) ✅

- **Independent:** Fallacy detection is pattern-based (no ML)

#### Action Items:
- ✅ Ready to implement
- 💡 **Improvement:** Use Phase 6.8 to ensure recent messages are indexed for fact-checking

---

### Phase 14: Hybrid Search (Vector + Keyword)
**Status:** ✅ Compatible
**Updates needed:** ⚠️ Needs integration with Phase 6

#### Services Defined:

```python
# File: retrieval/keyword.py (NEW)
class BM25Retriever:
    def __init__():
        self.bm25 = None
        self.documents = []

    def index_documents(documents: List[Dict]):
        # Build BM25 index (keyword search)
        # Uses: rank_bm25 library

    def search(query, top_k=10) -> List[Tuple[str, float, Dict]]:
        # BM25 keyword search
        # Returns: (doc_id, bm25_score, document)

# File: retrieval/hybrid.py (NEW)
class HybridSearchService:
    def __init__(chunked_memory, bm25_retriever):
        self.vector_search = chunked_memory  # Phase 6! ✅
        self.keyword_search = bm25_retriever

    def search(
        query,
        strategy="token_aware",  # Phase 4 strategies! ✅
        top_k=10,
        alpha=0.5,  # 0=keyword, 1=vector
        rrf_k=60    # RRF constant
    ) -> List[Dict]:
        # 1. Vector search (via ChunkedMemoryService)
        # 2. Keyword search (via BM25)
        # 3. Reciprocal Rank Fusion (RRF)
        # 4. Return top-k fused results

# File: bot/cogs/chatbot.py (Enhanced)
@commands.command(name='hybrid_search')
@commands.command(name='compare_search')
```

#### Integration Points:
- **Dependencies:**
  - `storage/chunked_memory.py` (Phase 6) ✅ Vector search
  - `rank_bm25` library (NEW dependency)

- **Enhances:** Phase 10 RAG with hybrid retrieval

#### Action Items:
- ⚠️ **TODO:** Add BM25 indexing to Phase 6.8 sync process:
  ```python
  # In storage/chunked_memory.py:
  def sync_strategy_incremental(...):
      # After embedding chunks:
      bm25_retriever.index_documents(new_chunks)
  ```
- ⚠️ **TODO:** Add dependency: `pip install rank-bm25`
- 💡 **Improvement:** Cache BM25 index (rebuild only when new chunks added)

---

### Phase 15: Reranking & Query Optimization
**Status:** ✅ Compatible
**Updates needed:** None

#### Services Defined:

```python
# File: retrieval/reranking.py (NEW)
class RerankingService:
    def __init__(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        # Uses: sentence-transformers cross-encoders

    def rerank(
        query,
        documents: List[Dict],
        top_k=10
    ) -> List[Tuple[Dict, float]]:
        # Cross-encoder reranking
        # Returns: (document, rerank_score) sorted

# File: retrieval/query_optimizer.py (NEW)
class QueryOptimizer:
    def expand_query_llm(query, num_variations=3) -> List[str]:
        # LLM-based query expansion
        # Generates synonymous queries

    def expand_query_keywords(query) -> List[str]:
        # Keyword-based expansion
        # Extracts and combines keywords

    def merge_results(results_list, top_k=10) -> List[Dict]:
        # RRF fusion of multiple result sets

# File: retrieval/advanced_rag.py (NEW)
class AdvancedRAGService:
    def __init__(query_optimizer, hybrid_search, reranker):
        # Complete pipeline:
        # Query opt → Expansion → Hybrid search → Fusion → Reranking

    async def search_with_optimization(...) -> Dict:
        # Full advanced RAG pipeline

# File: bot/cogs/rag_cog.py (NEW)
@commands.command(name='rerank_search')
@commands.command(name='expand_search')
@commands.command(name='advanced_search')
@commands.command(name='compare_pipelines')
```

#### Integration Points:
- **Dependencies:**
  - Phase 14 Hybrid Search ✅
  - `sentence-transformers` (cross-encoders)
  - OpenAI API (for query expansion)

- **Enhances:** Phase 10/14 with better retrieval quality

#### Action Items:
- ✅ Ready to implement
- ⚠️ **TODO:** Add dependencies:
  ```bash
  pip install sentence-transformers
  ```
- 💡 **Cost optimization:** Cache query expansions (LLM calls expensive)

---

## Dependency Graph

```
Phase 8 (Summary)
  └─ Uses: Phase 1 (MessageStorage)

Phase 8.5 (Leaderboards)
  └─ Uses: Phase 7.5 (UserAITracker)
  └─ NEW: storage/bookmarks.py

Phase 9 (Config)
  └─ Used by: ALL phases

Phase 10 (RAG Query)
  └─ Uses: Phase 6 (ChunkedMemoryService)
  └─ Uses: Phase 3 (AIService)
  └─ Uses: Phase 9 (Config)

Phase 10.5 (Smart Context)
  └─ Uses: Phase 10 (RAG results)
  └─ Uses: Phase 4 (tiktoken)

Phase 11 (Conversational)
  └─ Uses: Phase 10 (RAG)
  └─ Uses: Phase 3 (AIService)
  └─ NEW: ConversationMemory

Phase 12 (User Emulation)
  └─ Uses: Phase 1 (MessageStorage)
  └─ Uses: Phase 6 (ChunkedMemoryService)
  └─ Uses: Phase 3 (AIService)

Phase 13 (Debate Analysis)
  └─ Uses: Phase 10 (RAG for fact-checking)
  └─ Uses: Phase 3 (AIService)

Phase 14 (Hybrid Search)
  └─ Uses: Phase 6 (ChunkedMemoryService - vector search)
  └─ NEW: BM25Retriever (keyword search)

Phase 15 (Reranking)
  └─ Uses: Phase 14 (HybridSearchService)
  └─ NEW: Cross-encoder reranking
  └─ NEW: Query optimization
```

---

## Integration Checklist

### ✅ Already Compatible
- [x] Phase 8: Summary (uses Phase 1 MessageStorage)
- [x] Phase 8.5: Leaderboards (independent)
- [x] Phase 9: Config (Phase 4 configs already added)
- [x] Phase 10: RAG Query (works with Phase 6)
- [x] Phase 10.5: Smart Context (uses tiktoken like Phase 4)
- [x] Phase 11: Conversational (works with Phase 10)
- [x] Phase 12: User Emulation (uses Phase 1 + 6)
- [x] Phase 13: Debate Analysis (uses Phase 10)

### ⚠️ Needs Updates
- [ ] **Phase 9 Config:** Add Phase 6.8 sync configs
- [ ] **Phase 10 RAG:** Add Phase 7.5 cache integration
- [ ] **Phase 10.5 Context:** Use Phase 4 token counting method
- [ ] **Phase 11 Conversational:** Optional - persist to SQLite
- [ ] **Phase 12 Emulation:** Add user filtering to RAG queries
- [ ] **Phase 14 Hybrid:** Integrate BM25 indexing with Phase 6.8 sync
- [ ] **Phase 15 Reranking:** Add query expansion caching

### 🆕 New Dependencies Needed
- [ ] `rank-bm25` (Phase 14)
- [ ] `sentence-transformers[cross-encoder]` (Phase 15)

---

## Recommended Implementation Order

Based on dependencies and value:

### **Tier 1: High Value, Low Complexity** (Implement First)
1. **Phase 9:** Config validation ⏱️ 1 hour
2. **Phase 8:** Summary enhancement ⏱️ 2 hours
3. **Phase 8.5:** Leaderboards & bookmarks ⏱️ 4 hours
4. **Phase 10:** RAG Query Pipeline ⏱️ 6 hours

### **Tier 2: Core RAG Improvements** (Implement Second)
5. **Phase 10.5:** Smart Context Building ⏱️ 8 hours
6. **Phase 14:** Hybrid Search ⏱️ 6 hours
7. **Phase 15:** Reranking ⏱️ 4 hours

### **Tier 3: Advanced Features** (Implement Third)
8. **Phase 11:** Conversational Chatbot ⏱️ 8 hours
9. **Phase 13:** Debate Analysis ⏱️ 6 hours
10. **Phase 12:** User Emulation ⏱️ 10 hours (ethical considerations)

---

## Architecture Benefits

These phases give you:

### 🎯 **User-Facing Features**
- Smart conversational chatbot (Phase 11)
- Leaderboards & gamification (Phase 8.5)
- Debate analysis & fact-checking (Phase 13)
- User style emulation (Phase 12)

### 🚀 **Performance Improvements**
- DB-first caching (Phase 8)
- Hybrid search (Phase 14)
- Cross-encoder reranking (Phase 15)
- Smart context building (Phase 10.5)

### 🏗️ **Infrastructure**
- Complete RAG pipeline (Phase 10)
- Config validation (Phase 9)
- Query optimization (Phase 15)

---

## Next Steps

**Priority 1:** Update config (Phase 9)
```bash
# Edit config.py
# Add Phase 6.8 sync configs
# Add Phase 19 background task configs
```

**Priority 2:** Implement RAG Query (Phase 10)
```bash
# Create rag/pipeline.py
# Test with your existing Phase 6 ChunkedMemoryService
# Verify it works with Phase 4 token-aware strategy
```

**Priority 3:** Add Hybrid Search (Phase 14)
```bash
pip install rank-bm25
# Create retrieval/keyword.py
# Integrate with Phase 6.8 sync
```

---

## Questions to Consider

1. **Conversational Memory (Phase 11):**
   - In-memory (ephemeral) or SQLite (persistent)?
   - User privacy concerns with conversation history?

2. **User Emulation (Phase 12):**
   - Enable by default or opt-in only?
   - Add blacklist for users who don't want to be emulated?

3. **Hybrid Search (Phase 14):**
   - Alpha parameter: 0.5 (balanced) or tune per use case?
   - Rebuild BM25 index: on every sync or daily batch?

4. **Query Expansion (Phase 15):**
   - Use LLM (expensive, better) or keywords (free, okay)?
   - Cache expansions or generate fresh each time?

---

## Summary

**The good news:**
Your recent work (Phase 4 production chunking, Phase 6.8 incremental sync, Phase 19 background tasks) provides a **solid foundation** for all these advanced features!

**The action items:**
1. Minor config updates (Phase 9)
2. BM25 integration with sync (Phase 14)
3. Cache integration for performance (Phases 10, 15)

**The payoff:**
A production-ready Discord bot with RAG, conversational AI, fact-checking, and advanced search capabilities!
