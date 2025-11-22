# How Bot Knowledge Detection Works

## Overview

The bot doesn't explicitly "detect" questions about itself. Instead, **bot knowledge is automatically included in every RAG search**, and the semantic search naturally retrieves relevant bot documentation when the query is about the bot.

## How It Works

### 1. **Automatic Inclusion in All Searches**

When a user asks any question via `!ask` or `!rag`, the RAG pipeline:

1. **Searches bot documentation collection** (`discord_chunks_bot_docs`) alongside regular message collections
2. **Combines results** from both bot docs and regular messages
3. **Ranks all results** by semantic similarity to the query

This happens in `rag/pipeline.py` in the `_retrieve_chunks()` method:

```python
# Also search bot knowledge collection
bot_docs_results = []
try:
    bot_docs_collection = get_collection_name('bot_docs')
    collections = self.chunked_memory.vector_store.list_collections()
    if bot_docs_collection in collections:
        # Search bot docs collection directly
        query_embedding = await loop.run_in_executor(
            None,
            self.chunked_memory.embedder.encode,
            search_query
        )
        
        bot_docs_raw = self.vector_store.query(
            collection_name=bot_docs_collection,
            query_embeddings=[query_embedding],
            n_results=min(config.top_k // 2, 5)  # Get fewer from bot docs
        )
        # ... format results ...
except Exception as e:
    self.logger.debug(f"Could not search bot knowledge: {e}")

# Combine bot docs with regular chunks
if bot_docs_results:
    chunks = bot_docs_results + chunks
```

### 2. **Semantic Search Does the "Detection"**

The bot uses **semantic similarity** (vector embeddings) to find relevant chunks. When you ask:

- **"What commands do you have?"** → Semantic search finds chunks about "commands" in bot docs
- **"How does RAG work?"** → Finds chunks explaining the RAG system
- **"What AI providers are supported?"** → Finds chunks listing AI providers

The embeddings naturally match queries about the bot to bot documentation because:
- Bot docs contain words like "commands", "Deep-Bot", "RAG", "AI providers"
- These words are semantically similar to user queries about the bot
- Vector search ranks bot doc chunks highly when they're relevant

### 3. **No Explicit Detection Needed**

We don't need keyword matching like:
- ❌ "if 'bot' in query or 'you' in query"
- ❌ "if query starts with 'what can you'"

Instead, **semantic search automatically handles it** because:
- Bot documentation is indexed with embeddings
- User queries are embedded the same way
- Similar embeddings = relevant results

## Example Flow

### User asks: "What commands do you have?"

1. **Query embedding**: "What commands do you have?" → `[0.123, -0.456, ...]` (vector)

2. **Search both collections**:
   - Regular messages: Finds 0 relevant chunks (no messages about commands)
   - Bot docs: Finds 3 relevant chunks about "Available Commands" section

3. **Combine results**:
   ```
   Results: [
     {content: "Available Commands\n\n- !ping - Check bot latency\n...", similarity: 0.85, source: 'bot_documentation'},
     {content: "Basic Commands\n\n- !hello - Say hello...", similarity: 0.82, source: 'bot_documentation'},
     ...
   ]
   ```

4. **Generate answer**: LLM uses bot doc chunks to answer: "Deep-Bot has several commands including !ping, !hello, !ask, !summary..."

## When Bot Knowledge is Retrieved

Bot knowledge chunks are retrieved when:

✅ **Query is semantically similar to bot documentation**
- "What commands do you have?"
- "How does your RAG system work?"
- "What AI providers do you support?"
- "Tell me about Deep-Bot"

❌ **Query is about something else**
- "What did we discuss yesterday?" → Only regular messages
- "Who said X?" → Only regular messages
- "What's the weather?" → No relevant chunks from either source

## Advantages of This Approach

1. **No brittle keyword matching** - Works with natural language variations
2. **Automatic relevance** - Semantic search handles synonyms and related concepts
3. **Graceful degradation** - If bot docs aren't relevant, regular messages still work
4. **Combines sources** - Can answer questions using both bot docs AND past messages

## Testing

To verify bot knowledge works:

1. **Index bot knowledge**: `!reindex_bot_knowledge` (admin)
2. **Ask bot questions**: 
   - `!ask What commands do you have?`
   - `!ask How does RAG work?`
   - `!ask What AI providers are supported?`
3. **Check sources**: Bot should cite bot documentation in answers

## Configuration

- Bot knowledge is automatically initialized on bot startup
- Stored in collection: `discord_chunks_bot_docs`
- Metadata includes: `source: 'bot_documentation'`, `channel_id: 'system'`, `author: 'system'`
- Can be re-indexed with: `!reindex_bot_knowledge` (admin command)

