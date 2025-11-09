# RAG Pipeline Implementation Guide

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [Architecture Overview](#architecture-overview)
3. [Component Breakdown](#component-breakdown)
4. [Implementation Steps](#implementation-steps)
5. [Code Examples](#code-examples)
6. [Testing Strategy](#testing-strategy)
7. [Best Practices](#best-practices)

---

## What is RAG?

**RAG (Retrieval Augmented Generation)** combines two powerful AI techniques:

1. **Retrieval**: Finding relevant information from a knowledge base (your Discord messages)
2. **Generation**: Using an LLM to generate natural language answers based on that information

### Why RAG?

Without RAG, an LLM can only answer based on:
- Its training data (outdated, generic)
- The immediate conversation context (limited)

With RAG, you can:
- ✅ Answer questions about YOUR specific Discord conversations
- ✅ Provide sources/citations for transparency
- ✅ Stay up-to-date with recent discussions
- ✅ Handle domain-specific knowledge

### The RAG Pipeline Flow

```
User Question: "What did we decide about the database schema?"
     ↓
1. RETRIEVE: Search vector store for relevant chunks
   → Finds 5 chunks mentioning "database schema"
     ↓
2. FILTER: Remove low-similarity results
   → Keeps 3 chunks with >0.5 similarity
     ↓
3. BUILD CONTEXT: Format chunks with metadata
   → "[2025-01-15 14:23] Alice: Let's use PostgreSQL..."
   → "[2025-01-15 14:25] Bob: Agreed, with UUID primary keys"
     ↓
4. GENERATE: LLM reads context and answers
   → "Based on your conversation, you decided to use PostgreSQL
      with UUID primary keys (discussed on Jan 15th)"
     ↓
5. RETURN: Answer + sources to user
```

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Discord User                      │
└────────────────┬────────────────────────────────────┘
                 │ !ask "What was decided?"
                 ↓
┌─────────────────────────────────────────────────────┐
│              RAG Cog (bot/cogs/rag.py)              │
│  - Parse command                                     │
│  - Validate parameters                               │
│  - Call RAGPipeline                                  │
└────────────────┬────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────┐
│           RAGPipeline (rag/pipeline.py)             │
│  ┌─────────────────────────────────────────────┐   │
│  │ 1. Retrieve (ChunkedMemoryService)          │   │
│  │    → Vector similarity search                │   │
│  ├─────────────────────────────────────────────┤   │
│  │ 2. Filter (similarity threshold)             │   │
│  │    → Remove irrelevant chunks                │   │
│  ├─────────────────────────────────────────────┤   │
│  │ 3. Build Context (with metadata)             │   │
│  │    → Format: [timestamp] author: content     │   │
│  ├─────────────────────────────────────────────┤   │
│  │ 4. Generate (AIService)                      │   │
│  │    → LLM reads context + generates answer    │   │
│  ├─────────────────────────────────────────────┤   │
│  │ 5. Return RAGResult                          │   │
│  │    → Answer, sources, metadata               │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Data Flow

```
Question → Embedding → Vector Search → Chunks → Context → LLM → Answer
   ↓           ↓            ↓            ↓         ↓        ↓      ↓
 "schema"   [0.1,      ChromaDB      [chunk1]  "[time]  GPT-4  "You
            0.5,...]   similarity    [chunk2]   Alice:          decided
                       score >0.5    [chunk3]   ..."            on..."
```

---

## Component Breakdown

### 1. RAGConfig (Configuration Object)

**Purpose**: Encapsulate all RAG parameters in one place

**Why?**
- Makes testing easier (pass different configs)
- Allows presets (conservative, aggressive, balanced)
- Clear separation of concerns

**Parameters Explained**:

```python
@dataclass
class RAGConfig:
    # How many chunks to retrieve from vector store
    top_k: int = 5
    # Range: 1-20 typical
    # Higher = more context, but potentially noisier
    # Lower = more focused, but might miss relevant info
    
    # Minimum similarity score to include chunk
    similarity_threshold: float = 0.5
    # Range: 0.0-1.0 (cosine similarity)
    # 0.0 = include everything (noisy)
    # 0.5 = balanced (recommended)
    # 0.7 = strict (might miss relevant chunks)
    
    # Maximum tokens to send to LLM as context
    max_context_tokens: int = 4000
    # Most models have 8k-128k context windows
    # 4000 leaves room for: system prompt + question + answer
    # Prevents hitting token limits and high costs
    
    # LLM creativity parameter
    temperature: float = 0.7
    # Range: 0.0-2.0 (OpenAI), 0.0-1.0 (Anthropic)
    # 0.0 = deterministic, factual (good for Q&A)
    # 0.7 = balanced creativity (default)
    # 1.5+ = very creative, less predictable
    
    # Which chunking strategy to search
    strategy: str = "tokens"
    # "single" = individual messages (precise)
    # "tokens" = token-aware chunks (balanced)
    # "temporal" = time-based groups (good for events)
    
    # Which LLM model to use
    model: Optional[str] = None
    # None = use provider default (gpt-4o-mini)
    # Override for specific needs (gpt-4, claude-opus)
    
    # Whether to show source chunks in response
    show_sources: bool = False
    # False = answer only (clean)
    # True = answer + source chunks (transparent)
```

### 2. RAGResult (Output Object)

**Purpose**: Return structured results with metadata

**Why?**
- Transparency (user sees what sources were used)
- Debugging (track token usage, costs)
- Analytics (measure performance over time)

```python
@dataclass
class RAGResult:
    # The generated answer from the LLM
    answer: str
    
    # List of source chunks used
    sources: List[Dict]
    # Each source contains:
    # - content: chunk text
    # - metadata: timestamp, author, channel_id
    # - similarity: how relevant (0.0-1.0)
    
    # Configuration used (for reproducibility)
    config_used: RAGConfig
    
    # Total tokens sent to LLM
    tokens_used: int
    
    # Cost in USD
    cost: float
    
    # Which model generated the answer
    model: str
```

### 3. RAGPipeline (Orchestrator)

**Purpose**: Coordinate all RAG steps in the right order

**Why not just put this in the Cog?**
- **Separation of concerns**: Cog handles Discord, Pipeline handles RAG logic
- **Testability**: Can unit test pipeline without Discord bot
- **Reusability**: Could use pipeline in web API, CLI, etc.

**Architecture Pattern**: This follows the **Service Pattern**
- Cogs = Presentation Layer (Discord UI)
- Pipeline = Business Logic Layer (RAG orchestration)
- ChunkedMemoryService = Data Access Layer (vector store)

---

## Implementation Steps

### Step 1: Create Data Models (`rag/models.py`)

**Learning Objective**: Understand dataclasses and type safety

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class RAGConfig:
    """
    Configuration for RAG pipeline execution.
    
    Using a dataclass provides:
    - Automatic __init__, __repr__, __eq__
    - Type hints for IDE autocomplete
    - Immutability options (frozen=True)
    - Default values
    """
    top_k: int = 5
    similarity_threshold: float = 0.5
    max_context_tokens: int = 4000
    temperature: float = 0.7
    strategy: str = "tokens"
    model: Optional[str] = None
    show_sources: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.max_context_tokens < 100:
            raise ValueError("max_context_tokens must be at least 100")


@dataclass
class RAGResult:
    """
    Result of RAG pipeline execution.
    
    Separating the result from the process provides:
    - Clear API boundaries
    - Easy serialization (to JSON, DB, etc.)
    - Testability (mock results)
    """
    answer: str
    sources: List[Dict] = field(default_factory=list)
    config_used: Optional[RAGConfig] = None
    tokens_used: int = 0
    cost: float = 0.0
    model: str = "unknown"
    
    def format_for_discord(self, include_sources: bool = False) -> str:
        """
        Format result for Discord message.
        
        Why separate formatting from data?
        - Same data can be formatted differently (Discord, web, CLI)
        - Keeps RAGResult platform-agnostic
        """
        output = f"**Answer:**\n{self.answer}\n\n"
        output += f"*Model: {self.model} | Tokens: {self.tokens_used} | Cost: ${self.cost:.4f}*"
        
        if include_sources and self.sources:
            output += "\n\n**Sources:**\n"
            for i, source in enumerate(self.sources, 1):
                similarity = source.get('similarity', 0)
                metadata = source.get('metadata', {})
                author = metadata.get('author', 'Unknown')
                timestamp = metadata.get('timestamp', 'Unknown')
                output += f"{i}. [{timestamp}] {author} (similarity: {similarity:.2f})\n"
        
        return output
```

**Key Concepts**:
- **Dataclasses**: Python 3.7+ feature for creating data containers
- **Type hints**: Help catch bugs before runtime
- **Default factories**: `field(default_factory=list)` avoids mutable default bug
- **Validation**: `__post_init__` runs after `__init__` to validate inputs
- **Separation of concerns**: Data model separate from presentation logic

---

### Step 2: Build RAG Pipeline Core (`rag/pipeline.py`)

**Learning Objective**: Orchestration patterns and async programming

#### 2.1 Class Structure

```python
import logging
from typing import List, Dict, Optional
from storage.chunked_memory import ChunkedMemoryService
from ai.service import AIService
from storage.messages.messages import MessageStorage
from .models import RAGConfig, RAGResult
from chunking.constants import ChunkStrategy

class RAGPipeline:
    """
    Orchestrates the complete RAG (Retrieval Augmented Generation) process.
    
    Design Pattern: Service/Orchestrator
    - Coordinates multiple services (ChunkedMemory, AI)
    - Implements business logic (filtering, context building)
    - Returns structured results
    
    Why dependency injection?
    - Testability: Can inject mocks
    - Flexibility: Can swap implementations
    - Loose coupling: Pipeline doesn't create its dependencies
    """
    
    def __init__(
        self,
        chunked_memory_service: Optional[ChunkedMemoryService] = None,
        ai_service: Optional[AIService] = None,
        message_storage: Optional[MessageStorage] = None,
    ):
        """
        Initialize pipeline with dependencies.
        
        Optional parameters with defaults = dependency injection pattern
        Production: uses real services
        Testing: injects mocks
        """
        self.chunked_memory = chunked_memory_service or ChunkedMemoryService()
        self.ai_service = ai_service or AIService()
        self.message_storage = message_storage or MessageStorage()
        self.logger = logging.getLogger(__name__)
```

#### 2.2 Main Pipeline Method

```python
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
        
        Why async?
        - API calls (embedding, LLM) are I/O bound
        - async allows other Discord operations to proceed
        - Better resource utilization
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
                max_tokens=1000,  # Reserve tokens for answer
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
```

**Key Concepts**:
- **Error handling**: Try/except prevents crashes, returns graceful errors
- **Logging**: Track pipeline execution for debugging
- **Graceful degradation**: No results? Return helpful message
- **Structured returns**: Always return RAGResult, even on error

#### 2.3 Retrieval Stage

```python
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
        
        Why async?
        - Embedding API calls are I/O bound
        - ChunkedMemoryService.search() is async
        """
        self.logger.info(f"Retrieving top {config.top_k} chunks with strategy: {config.strategy}")
        
        try:
            strategy = ChunkStrategy(config.strategy)
        except ValueError:
            self.logger.warning(f"Invalid strategy '{config.strategy}', using default")
            strategy = ChunkStrategy.TOKENS
        
        chunks = self.chunked_memory.search(
            query=query,
            strategy=strategy,
            top_k=config.top_k,
        )
        
        self.logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks
```

#### 2.4 Filtering Stage

```python
    def _filter_by_similarity(
        self,
        chunks: List[Dict],
        threshold: float,
    ) -> List[Dict]:
        """
        Filter chunks below similarity threshold.
        
        Why filter?
        - Low-similarity chunks add noise
        - Reduces token usage (cheaper, faster)
        - Improves answer quality
        
        Similarity scores:
        - 1.0 = identical match
        - 0.7-0.9 = very relevant
        - 0.5-0.7 = somewhat relevant
        - <0.5 = probably not relevant
        """
        filtered = [
            chunk for chunk in chunks
            if chunk.get('similarity', 0) >= threshold
        ]
        
        removed = len(chunks) - len(filtered)
        if removed > 0:
            self.logger.info(f"Filtered out {removed} chunks below threshold {threshold}")
        
        return filtered
```

#### 2.5 Context Building Stage

```python
    def _build_context_with_metadata(
        self,
        chunks: List[Dict],
        max_tokens: int,
    ) -> str:
        """
        Build context string with timestamp and author metadata.
        
        Format: [timestamp] author: content
        
        Why this format?
        - Natural language (LLM-friendly)
        - Temporal context (when was it said?)
        - Attribution (who said it?)
        - Conversational flow
        
        Token management:
        - Estimate 4 chars ≈ 1 token (rough heuristic)
        - Truncate if exceeds max_tokens
        - Prioritize most relevant chunks (already sorted by similarity)
        """
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough token estimation
        
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
```

**Key Concepts**:
- **Token estimation**: Rough heuristic (4 chars = 1 token) prevents overruns
- **Prioritization**: Most relevant chunks included first
- **Metadata enrichment**: Adds temporal and social context
- **Budget management**: Respects token limits

#### 2.6 Prompt Engineering Stage

```python
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create prompt for LLM with context and question.
        
        Prompt engineering principles:
        1. Clear role definition (system message)
        2. Context before question (establishes knowledge base)
        3. Specific instructions (answer format, citation style)
        4. Question at the end (focus attention)
        
        Why this structure?
        - LLMs perform better with clear instructions
        - Context establishes "ground truth"
        - Reduces hallucination (making up facts)
        """
        system_message = """You are a helpful assistant that answers questions based on Discord conversation history.

Instructions:
- Answer ONLY using information from the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Reference specific messages when relevant
- Do NOT make up information not in the context"""

        user_message = f"""Context from Discord conversations:

{context}

---

Question: {question}

Answer:"""

        # Combine into single prompt (simple format for AIService)
        full_prompt = f"{system_message}\n\n{user_message}"
        
        return full_prompt
```

**Key Concepts**:
- **System messages**: Define assistant behavior
- **Context separation**: Clear boundary between context and question
- **Explicit constraints**: "ONLY using information from context" reduces hallucination
- **Format consistency**: Structured format helps LLM understand task

---

### Step 3: Discord Commands (`bot/cogs/rag.py`)

**Learning Objective**: Discord bot patterns and user interaction

#### 3.1 Cog Structure

```python
import discord
from discord.ext import commands
import logging
from rag.pipeline import RAGPipeline
from rag.models import RAGConfig, RAGResult
from config import Config

class RAG(commands.Cog):
    """
    Discord cog for RAG (Retrieval Augmented Generation) commands.
    
    Commands:
    - !ask: Simple Q&A for all users
    - !ask_detailed: Advanced Q&A with parameters (admin only)
    
    Why separate from RAGPipeline?
    - Cog = Discord-specific UI logic
    - Pipeline = Reusable business logic
    - Clean separation of concerns
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.pipeline = RAGPipeline()
        self.logger = logging.getLogger(__name__)
```

#### 3.2 Simple Ask Command

```python
    @commands.command(
        name='ask',
        help='Ask a question about Discord conversations'
    )
    async def ask(self, ctx, *, question: str):
        """
        Simple RAG command for all users.
        
        Usage: !ask What was decided about the database?
        
        Uses default configuration from Config.
        Returns clean answer without technical details.
        
        Why keep it simple?
        - Lower barrier for users
        - Most users don't need advanced controls
        - Clean, focused UX
        """
        # Show typing indicator (user knows bot is working)
        async with ctx.typing():
            self.logger.info(f"User {ctx.author} asked: {question}")
            
            # Create config from environment defaults
            config = RAGConfig(
                top_k=Config.RAG_DEFAULT_TOP_K,
                similarity_threshold=Config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
                max_context_tokens=Config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
                temperature=Config.RAG_DEFAULT_TEMPERATURE,
                strategy=Config.RAG_DEFAULT_STRATEGY,
                show_sources=False,  # Simple mode = no sources
            )
            
            # Execute pipeline
            result = await self.pipeline.answer_question(question, config)
            
            # Create Discord embed (prettier than plain text)
            embed = discord.Embed(
                title="💡 Answer",
                description=result.answer,
                color=discord.Color.blue()
            )
            
            # Add footer with metadata (subtle, not overwhelming)
            embed.set_footer(
                text=f"Model: {result.model} | Cost: ${result.cost:.4f}"
            )
            
            # Send response
            message = await ctx.send(embed=embed)
            
            # Add reaction for optional source viewing
            await message.add_reaction("📚")
            
            # Store result for reaction handler
            self.bot._rag_cache = getattr(self.bot, '_rag_cache', {})
            self.bot._rag_cache[message.id] = result
```

**Key Concepts**:
- **Typing indicator**: Visual feedback during processing
- **Discord embeds**: Rich formatting (title, description, footer)
- **Reaction-based UI**: Optional sources without cluttering response
- **Caching**: Store result for reaction handler

#### 3.3 Reaction Handler for Sources

```python
    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        """
        Handle 📚 reaction to show sources.
        
        Why reactions instead of always showing?
        - Most users just want the answer
        - Advanced users can opt-in to sources
        - Keeps responses clean
        """
        # Ignore bot's own reactions
        if user.bot:
            return
        
        # Check if it's the sources emoji
        if str(reaction.emoji) != "📚":
            return
        
        # Check if this message has cached results
        rag_cache = getattr(self.bot, '_rag_cache', {})
        result = rag_cache.get(reaction.message.id)
        
        if not result or not result.sources:
            return
        
        # Create sources embed
        embed = discord.Embed(
            title="📚 Sources Used",
            description=f"Top {len(result.sources)} relevant chunks:",
            color=discord.Color.green()
        )
        
        for i, source in enumerate(result.sources, 1):
            content = source.get('content', '')
            metadata = source.get('metadata', {})
            similarity = source.get('similarity', 0)
            
            # Truncate long content
            preview = content[:200] + "..." if len(content) > 200 else content
            
            embed.add_field(
                name=f"{i}. Similarity: {similarity:.2f}",
                value=(
                    f"**Author:** {metadata.get('author', 'Unknown')}\n"
                    f"**Time:** {metadata.get('timestamp', 'Unknown')}\n"
                    f"**Content:** {preview}"
                ),
                inline=False
            )
        
        # Send as DM to avoid channel clutter
        try:
            await user.send(embed=embed)
            await reaction.message.add_reaction("✅")  # Confirm sent
        except discord.Forbidden:
            # User has DMs disabled
            await reaction.message.channel.send(
                f"{user.mention} Please enable DMs to receive sources!",
                delete_after=10
            )
```

#### 3.4 Advanced Ask Command (Admin Only)

```python
    @commands.command(
        name='ask_detailed',
        help='Ask with advanced RAG parameters (admin only)'
    )
    @commands.has_permissions(administrator=True)
    async def ask_detailed(
        self,
        ctx,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        max_context_tokens: int = 4000,
        temperature: float = 0.7,
        strategy: str = "tokens",
        show_sources: bool = False,
        *,
        question: str
    ):
        """
        Advanced RAG command with full parameter control.
        
        Usage:
            !ask_detailed 10 0.7 4000 0.5 tokens true What was decided?
            
        Parameters:
            top_k: Number of chunks to retrieve (1-20)
            similarity_threshold: Min similarity (0.0-1.0)
            max_context_tokens: Max context size (100-8000)
            temperature: LLM creativity (0.0-2.0)
            strategy: Chunking strategy (single, tokens, temporal, etc.)
            show_sources: Show source chunks inline (true/false)
            question: Your question (rest of message)
        
        Why admin-only?
        - Prevents abuse (high token usage)
        - Requires understanding of parameters
        - Testing and tuning use case
        """
        async with ctx.typing():
            self.logger.info(
                f"Admin {ctx.author} asked with params: "
                f"top_k={top_k}, threshold={similarity_threshold}, "
                f"strategy={strategy}"
            )
            
            try:
                # Create custom config
                config = RAGConfig(
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    max_context_tokens=max_context_tokens,
                    temperature=temperature,
                    strategy=strategy,
                    show_sources=show_sources,
                )
            except ValueError as e:
                await ctx.send(f"❌ Invalid configuration: {e}")
                return
            
            # Execute pipeline
            result = await self.pipeline.answer_question(question, config)
            
            # Create detailed response
            embed = discord.Embed(
                title="🔧 Detailed RAG Answer",
                description=result.answer,
                color=discord.Color.gold()
            )
            
            # Show configuration used
            embed.add_field(
                name="Configuration",
                value=(
                    f"**Strategy:** {config.strategy}\n"
                    f"**Top K:** {config.top_k}\n"
                    f"**Threshold:** {config.similarity_threshold}\n"
                    f"**Temperature:** {config.temperature}\n"
                    f"**Max Tokens:** {config.max_context_tokens}"
                ),
                inline=False
            )
            
            # Show performance metrics
            embed.add_field(
                name="Metrics",
                value=(
                    f"**Tokens Used:** {result.tokens_used}\n"
                    f"**Cost:** ${result.cost:.4f}\n"
                    f"**Model:** {result.model}\n"
                    f"**Sources Found:** {len(result.sources)}"
                ),
                inline=False
            )
            
            await ctx.send(embed=embed)
            
            # Show sources if requested
            if show_sources and result.sources:
                sources_embed = discord.Embed(
                    title="📚 Sources",
                    color=discord.Color.green()
                )
                
                for i, source in enumerate(result.sources[:5], 1):  # Limit to 5
                    content = source.get('content', '')[:200]
                    metadata = source.get('metadata', {})
                    similarity = source.get('similarity', 0)
                    
                    sources_embed.add_field(
                        name=f"{i}. {metadata.get('author', 'Unknown')} ({similarity:.2f})",
                        value=f"*{metadata.get('timestamp', 'Unknown')}*\n{content}...",
                        inline=False
                    )
                
                await ctx.send(embed=sources_embed)


async def setup(bot):
    """Load the RAG cog"""
    await bot.add_cog(RAG(bot))
```

**Key Concepts**:
- **Permission checks**: `@commands.has_permissions(administrator=True)`
- **Parameter validation**: Try/except with helpful error messages
- **Detailed feedback**: Show config and metrics for tuning
- **Field limits**: Discord embeds have max 25 fields, limit sources
- **Admin-only**: Prevents abuse, enables testing

---

### Step 4: Configuration (`config.py`)

Add RAG defaults to your existing `Config` class:

```python
    # RAG Configuration
    RAG_DEFAULT_TOP_K: int = int(os.getenv("RAG_DEFAULT_TOP_K", "5"))
    RAG_DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_DEFAULT_SIMILARITY_THRESHOLD", "0.5"))
    RAG_DEFAULT_MAX_CONTEXT_TOKENS: int = int(os.getenv("RAG_DEFAULT_MAX_CONTEXT_TOKENS", "4000"))
    RAG_DEFAULT_TEMPERATURE: float = float(os.getenv("RAG_DEFAULT_TEMPERATURE", "0.7"))
    RAG_DEFAULT_STRATEGY: str = os.getenv("RAG_DEFAULT_STRATEGY", "tokens")
```

Add to your `.env` file (optional overrides):

```env
# RAG Settings
RAG_DEFAULT_TOP_K=5
RAG_DEFAULT_SIMILARITY_THRESHOLD=0.5
RAG_DEFAULT_MAX_CONTEXT_TOKENS=4000
RAG_DEFAULT_TEMPERATURE=0.7
RAG_DEFAULT_STRATEGY=tokens
```

---

### Step 5: Bot Integration (`bot/bot.py`)

Load the RAG cog in your bot initialization:

```python
async def load_extensions(self):
    """Load all cogs"""
    extensions = [
        "bot.cogs.admin",
        "bot.cogs.summary",
        "bot.cogs.rag",  # Add this line
    ]
    
    for extension in extensions:
        try:
            await self.load_extension(extension)
            self.logger.info(f"Loaded extension: {extension}")
        except Exception as e:
            self.logger.error(f"Failed to load {extension}: {e}")
```

---

## Testing Strategy

### Unit Tests (`tests/test_rag_pipeline.py`)

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from rag.pipeline import RAGPipeline
from rag.models import RAGConfig, RAGResult

@pytest.fixture
def mock_chunks():
    """Sample chunks for testing"""
    return [
        {
            'content': 'We decided to use PostgreSQL',
            'metadata': {
                'timestamp': '2025-01-15 14:23:00',
                'author': 'Alice',
                'channel_id': '123',
            },
            'similarity': 0.85
        },
        {
            'content': 'UUID primary keys are better',
            'metadata': {
                'timestamp': '2025-01-15 14:25:00',
                'author': 'Bob',
                'channel_id': '123',
            },
            'similarity': 0.75
        },
        {
            'content': 'Not very relevant message',
            'metadata': {
                'timestamp': '2025-01-15 14:30:00',
                'author': 'Charlie',
                'channel_id': '123',
            },
            'similarity': 0.35
        }
    ]


class TestContextBuilding:
    """Test context building with metadata"""
    
    def test_build_context_with_metadata(self, mock_chunks):
        """Should format chunks with [timestamp] author: content"""
        pipeline = RAGPipeline()
        
        context = pipeline._build_context_with_metadata(
            mock_chunks[:2],  # Just first 2 chunks
            max_tokens=10000  # High limit, no truncation
        )
        
        # Check format
        assert '[2025-01-15 14:23:00] Alice: We decided to use PostgreSQL' in context
        assert '[2025-01-15 14:25:00] Bob: UUID primary keys are better' in context
        
        # Check structure
        lines = context.split('\n\n')
        assert len(lines) == 2
    
    def test_token_limit_truncation(self, mock_chunks):
        """Should truncate context at token limit"""
        pipeline = RAGPipeline()
        
        # Very low token limit (should fit only 1 chunk)
        context = pipeline._build_context_with_metadata(
            mock_chunks[:2],
            max_tokens=50  # ~200 chars = 50 tokens
        )
        
        # Should only include first chunk
        assert 'Alice' in context
        assert 'Bob' not in context


class TestFiltering:
    """Test similarity filtering"""
    
    def test_filter_by_similarity(self, mock_chunks):
        """Should filter out low similarity chunks"""
        pipeline = RAGPipeline()
        
        filtered = pipeline._filter_by_similarity(
            mock_chunks,
            threshold=0.5
        )
        
        # Should keep first 2, filter out 3rd (0.35 < 0.5)
        assert len(filtered) == 2
        assert filtered[0]['similarity'] == 0.85
        assert filtered[1]['similarity'] == 0.75
    
    def test_no_filtering_with_zero_threshold(self, mock_chunks):
        """Should keep all chunks with 0.0 threshold"""
        pipeline = RAGPipeline()
        
        filtered = pipeline._filter_by_similarity(
            mock_chunks,
            threshold=0.0
        )
        
        assert len(filtered) == 3


class TestFullPipeline:
    """Integration tests for complete pipeline"""
    
    @pytest.mark.asyncio
    async def test_answer_question_success(self, mock_chunks):
        """Should successfully answer with mocked dependencies"""
        # Mock ChunkedMemoryService
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=mock_chunks)
        
        # Mock AIService
        mock_ai = MagicMock()
        mock_ai.generate = AsyncMock(return_value={
            'content': 'You decided to use PostgreSQL with UUID keys.',
            'tokens_total': 150,
            'cost': 0.002,
            'model': 'gpt-4o-mini',
        })
        
        pipeline = RAGPipeline(
            chunked_memory_service=mock_memory,
            ai_service=mock_ai,
        )
        
        config = RAGConfig(
            top_k=5,
            similarity_threshold=0.5,
            temperature=0.7,
        )
        
        result = await pipeline.answer_question(
            "What database did we choose?",
            config
        )
        
        # Verify result
        assert isinstance(result, RAGResult)
        assert 'PostgreSQL' in result.answer
        assert len(result.sources) == 2  # 2 chunks above threshold
        assert result.tokens_used == 150
        assert result.cost == 0.002
        assert result.model == 'gpt-4o-mini'
    
    @pytest.mark.asyncio
    async def test_no_relevant_chunks(self):
        """Should handle case with no relevant results"""
        mock_memory = MagicMock()
        mock_memory.search = MagicMock(return_value=[])
        
        pipeline = RAGPipeline(chunked_memory_service=mock_memory)
        
        result = await pipeline.answer_question("Unrelated question")
        
        assert "couldn't find" in result.answer.lower()
        assert len(result.sources) == 0
        assert result.model == "none"
```

---

## Best Practices

### 1. Error Handling

**Always handle errors gracefully in user-facing code:**

```python
try:
    result = await pipeline.answer_question(question)
except Exception as e:
    logger.error(f"Pipeline failed: {e}", exc_info=True)
    await ctx.send("Sorry, something went wrong. Please try again.")
```

### 2. Logging

**Log at appropriate levels:**

```python
self.logger.debug("Building context...")  # Detailed dev info
self.logger.info("Retrieved 5 chunks")    # Important milestones
self.logger.warning("No chunks found")     # Potential issues
self.logger.error("API failed", exc_info=True)  # Errors with stacktrace
```

### 3. Token Management

**Always respect token limits:**

- Estimate tokens (4 chars = 1 token rough)
- Leave buffer for answer (context + answer < model limit)
- Monitor costs in production

### 4. Prompt Engineering

**Test and iterate on prompts:**

- Start with clear, simple instructions
- Add constraints to reduce hallucination
- Use examples if needed (few-shot prompting)
- Test with edge cases

### 5. User Experience

**Make commands intuitive:**

- Simple command for common use case (!ask)
- Advanced command for power users (!ask_detailed)
- Show progress (typing indicator)
- Provide feedback (reactions, embeds)

### 6. Testing

**Test each component:**

- Unit tests for pure functions (filtering, formatting)
- Integration tests with mocks (full pipeline)
- Manual testing on Discord (UX validation)

---

## Advanced Topics

### 1. Hybrid Search

Combine vector search with keyword search:

```python
# Vector search for semantic similarity
vector_results = chunked_memory.search(query, top_k=10)

# Keyword search for exact matches
keyword_results = message_storage.search_text(query)

# Merge and deduplicate
combined = merge_results(vector_results, keyword_results)
```

### 2. Re-ranking

Use a dedicated re-ranker model to improve relevance:

```python
# Get more candidates initially
chunks = chunked_memory.search(query, top_k=20)

# Re-rank with cross-encoder
reranked = reranker_model.rerank(query, chunks)

# Take top 5 after reranking
final_chunks = reranked[:5]
```

### 3. Multi-turn Conversations

Track conversation history for follow-up questions:

```python
conversation_history = [
    {"role": "user", "content": "What database?"},
    {"role": "assistant", "content": "PostgreSQL"},
    {"role": "user", "content": "Why?"},  # Needs context
]
```

### 4. Source Citation

Format answers with inline citations:

```python
answer = "You decided to use PostgreSQL [1] with UUID primary keys [2]."
sources = [
    "[1] Message from Alice on Jan 15: 'Use PostgreSQL'",
    "[2] Message from Bob on Jan 15: 'UUID keys are better'",
]
```

---

## Troubleshooting

### Common Issues

1. **"I couldn't find any relevant information"**
   - Check if channel has been ingested (!chunk_status)
   - Try lowering similarity_threshold
   - Verify search strategy has data

2. **Answers are hallucinated (made up)**
   - Lower temperature (0.3-0.5 for factual)
   - Strengthen prompt constraints
   - Increase similarity threshold

3. **Answers miss relevant info**
   - Increase top_k
   - Lower similarity threshold
   - Try different chunking strategy

4. **Token limit exceeded**
   - Reduce max_context_tokens
   - Reduce top_k
   - Implement smarter truncation

---

## Next Steps

1. Implement basic pipeline (models, core logic)
2. Add Discord commands
3. Test manually on Discord
4. Add unit tests
5. Tune parameters based on results
6. Add advanced features (re-ranking, hybrid search)

---

## Further Reading

- **RAG Papers**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- **Prompt Engineering**: https://platform.openai.com/docs/guides/prompt-engineering
- **Discord.py**: https://discordpy.readthedocs.io/
- **Vector Databases**: ChromaDB documentation
- **Token Estimation**: tiktoken library

---

**Happy coding! 🚀**

