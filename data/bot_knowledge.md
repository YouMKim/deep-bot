# Deep-Bot Documentation

## Overview

Deep-Bot is a Discord bot that indexes your server's conversation history for search and question-answering. It uses vector embeddings, semantic search, hybrid retrieval, and language models to answer questions based on your server's conversation history.

## Bot Name and Purpose

Deep-Bot is a Retrieval-Augmented Generation (RAG) Discord bot designed to help users find information from past conversations. It can answer questions about what was discussed in the server, generate summaries, and provide contextual responses using AI.

## Available Commands

### Basic Commands

- `!ping` - Check bot latency
- `!hello` - Say hello to the bot (AI-powered greeting)
- `!info` - Get bot information (guilds, users, latency, prefix)
- `!echo <message>` - Echo a message back
- `!server` - Get server information (members, channels, roles, creation date)
- `!mystats` - View your AI usage stats and social credit score
- `!help` - Show available commands for regular users
- `!help_admin` - Show available admin commands (Admin only)

### RAG Commands

- `!ask <question>` - Ask a question about past conversations. The bot will search through indexed messages and provide an answer based on what was discussed.
- `!rag <question>` - Alternative command for asking questions (same as !ask)

### Summary Commands

- `!summary [count]` - Generate a summary of recent messages from local storage. Default count is 50 messages. Rate limit: 3 summaries per 2 minutes per user.

### Chatbot Commands

- `!chatbot_reset` - Reset channel conversation history
- `!chatbot_stats` - View channel chatbot usage statistics
- `!chatbot_mode` - Check current chatbot settings (Admin only)

### Social Credit Commands

- `!socialcredit view [@user]` or `!sc view [@user]` - View your or another user's social credit score
- `!socialcredit give @user <points>` - Give points to a user (Admin only)
- `!socialcredit take @user <points>` - Take points from a user (Admin only)
- `!socialcredit set @user <score>` - Set user's score (Admin only)
- `!socialcredit reset @user` - Reset user's score to 0 (Admin only)
- `!socialcredit leaderboard [top|bottom] [limit]` - Show leaderboard (Admin only)

### Admin Commands

- `!whoami` - Check who the bot thinks is the owner
- `!check_blacklist` - Check the current blacklist configuration
- `!reload_blacklist` - Reload the blacklist from environment variables (Admin only)
- `!load_channel` - Load all messages from current channel into memory (Admin only)
- `!check_storage` - Check message storage statistics for current channel
- `!checkpoint_info` - Show checkpoint information for current channel
- `!chunk_status` - Show chunking progress and statistics for current channel
- `!reset_chunk_checkpoint` - Delete chunk checkpoint to force re-processing (Admin only)
- `!rechunk` - Re-run chunking from last checkpoint (Admin only)
- `!rag_settings` - Show current RAG technique settings (Admin only)
- `!rag_set <setting> <value>` - Set a RAG technique setting (Admin only)
- `!rag_reset` - Reset all RAG settings to defaults (Admin only)
- `!rag_enable_all` - Enable all RAG techniques at once (Admin only)
- `!compare_rag <question>` - Compare RAG output with different technique combinations (Admin only)
- `!ai_provider <provider>` - Switch AI provider (Admin only). Options: openai, anthropic, gemini
- `!reindex_bot_knowledge [true]` - Re-index bot documentation into RAG system (Admin only)
  - Without argument: Re-index if not already indexed
  - With "true": Force re-index even if already indexed

## Technical Stack

### Core Technologies

- **Python 3.12+**: Modern Python with async/await support
- **discord.py**: Discord API wrapper with full async support
- **ChromaDB**: Vector database for embedding storage and retrieval
- **SQLite**: Relational database for message metadata and statistics
- **SentenceTransformers**: Local embedding generation (all-MiniLM-L6-v2 model)
- **rank-bm25**: BM25 keyword search implementation

### AI & ML Libraries

- **OpenAI SDK**: GPT model integration (GPT-5, GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo)
- **Anthropic SDK**: Claude model integration (Claude Sonnet 4.5, Claude Haiku 4.5, Claude Opus 4.1)
- **Google Generative AI**: Gemini model integration (Gemini 2.5 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash)
- **sentence-transformers**: Embedding models for semantic search
- **transformers**: Cross-encoder reranking models

### Data Processing

- **tiktoken**: Fast token counting for OpenAI models
- **numpy**: Numerical operations for embeddings
- **asyncio**: Asynchronous I/O for high concurrency

## How RAG Works

Deep-Bot uses a Retrieval-Augmented Generation (RAG) system to answer questions about past conversations. Here's how it works:

### 1. Message Ingestion

When messages are loaded into the bot:
- Messages are stored in SQLite database for fast retrieval
- Messages are chunked using various strategies (temporal, author-based, token-based)
- Chunks are embedded using sentence-transformers or OpenAI embeddings
- Embeddings are stored in ChromaDB vector database

### 2. Retrieval Strategies

The bot uses multiple retrieval strategies:

- **Vector Semantic Search**: Uses dense embeddings to find semantically similar content
- **BM25 Keyword Search**: Keyword-based information retrieval using BM25 algorithm
- **Hybrid Search**: Combines vector and BM25 using Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: Re-ranks results using cross-encoder models for better accuracy
- **Multi-Query Expansion**: Generates multiple query variations to improve recall
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers for better retrieval

### 3. Answer Generation

When you ask a question:
- The bot searches through indexed chunks using the configured retrieval strategies
- Relevant chunks are filtered by similarity threshold
- Context is built from the most relevant chunks
- An AI model generates an answer based on the retrieved context

## Chunking Strategies

The bot uses multiple chunking strategies to organize messages. Each strategy serves different purposes:

### Single Strategy
- **Description**: One message per chunk
- **Use Case**: Fast, simple chunking for quick searches
- **Pros**: Fastest, preserves individual message context
- **Cons**: May miss multi-message context
- **Best For**: Simple queries, when speed is priority

### Temporal Strategy
- **Description**: Groups messages by time windows (default: 1 hour)
- **Use Case**: Finding conversations within specific time periods
- **Pros**: Captures conversations that happened together
- **Cons**: May exceed token limits, may split related conversations
- **Best For**: Time-based queries, chronological context

### Author Strategy
- **Description**: Groups messages by author for conversation threads
- **Use Case**: Finding what specific users said
- **Pros**: Great for author filtering, captures user's full thoughts
- **Cons**: May create very long chunks
- **Best For**: Author-specific queries, user-focused searches
- **Default**: Yes (part of default strategies)

### Tokens Strategy
- **Description**: Token-aware chunking that respects token limits
- **Use Case**: Ensuring chunks fit within LLM context windows
- **Pros**: Guarantees chunks stay within token limits
- **Cons**: May split sentences or thoughts
- **Best For**: Production use, when token limits matter
- **Default**: Yes (part of default strategies)

### Sliding Window Strategy
- **Description**: Overlapping windows for better context
- **Use Case**: Capturing context across chunk boundaries
- **Pros**: Prevents information loss at boundaries
- **Cons**: Creates duplicate chunks, uses more storage
- **Best For**: When context continuity is critical

### Conversation Strategy
- **Description**: Gap detection between conversations
- **Use Case**: Grouping messages that are part of the same conversation
- **Pros**: Identifies natural conversation boundaries
- **Cons**: May miss related conversations with gaps
- **Best For**: Conversation-level queries

### Default Strategies

Default strategies are: **single, tokens, and author**. This combination provides:
- Fast retrieval (single)
- Token safety (tokens)
- Author context (author)

You can change this with `CHUNKING_DEFAULT_STRATEGIES` environment variable.

## AI Providers

Deep-Bot supports multiple AI providers:

### OpenAI
- Models: GPT-5, GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo
- Default model: GPT-5-mini-2025-08-07

### Anthropic
- Models: Claude Sonnet 4.5, Claude Haiku 4.5, Claude Opus 4.1
- Default model: claude-haiku-4-5

### Google Gemini
- Models: Gemini 2.5 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash
- Default model: gemini-2.5-flash

You can switch providers using the `!ai_provider` command (Admin only) or set `AI_DEFAULT_PROVIDER` environment variable.

## Configuration Options

### Required Environment Variables

- `DISCORD_TOKEN` - Discord bot token (required)
- `OPENAI_API_KEY` - OpenAI API key (required for basic functionality)

### Optional API Keys

- `ANTHROPIC_API_KEY` - Anthropic API key (optional, enables Claude models)
- `GEMINI_API_KEY` - Google Gemini API key (optional, enables Gemini models)

### Discord Configuration

- `DISCORD_CLIENT_ID` - Discord application client ID (optional)
- `DISCORD_GUILD_ID` - Specific guild ID to restrict bot to (optional)
- `BOT_PREFIX` - Command prefix (default: "!")
- `BOT_OWNER_ID` - Bot owner Discord user ID (optional, for admin commands)

### Embedding Configuration

- `EMBEDDING_PROVIDER` - Embedding provider: "sentence-transformers" or "openai" (default: sentence-transformers)
  - `sentence-transformers`: Uses local model "all-MiniLM-L6-v2" (free, runs locally)
  - `openai`: Uses OpenAI's "text-embedding-3-small" (requires API key, better quality)
- `EMBEDDING_MODEL` - Specific embedding model name (optional, auto-selected if not set)
- `EMBEDDING_BATCH_SIZE` - Number of documents to embed per batch (default: 100)
- `EMBEDDING_BATCH_DELAY` - Seconds to wait between batches for rate limiting (default: 0.1)

### RAG Configuration

#### Core RAG Settings

- `RAG_DEFAULT_TOP_K` - Number of chunks to retrieve (default: 8)
- `RAG_DEFAULT_SIMILARITY_THRESHOLD` - Minimum similarity score (default: 0.01)
  - Lower values = more results (better recall)
  - Higher values = fewer but more relevant results (better precision)
- `RAG_DEFAULT_MAX_CONTEXT_TOKENS` - Maximum context tokens sent to LLM (default: 4000)
- `RAG_DEFAULT_STRATEGY` - Default chunking strategy (default: author)
- `RAG_DEFAULT_TEMPERATURE` - LLM temperature for answer generation (default: 0.7)
- `RAG_MAX_OUTPUT_TOKENS` - Maximum tokens in generated answer (default: 1200)

#### RAG Technique Toggles

- `RAG_USE_HYBRID_SEARCH` - Enable hybrid search combining vector + BM25 (default: true)
- `RAG_USE_MULTI_QUERY` - Enable multi-query expansion (default: true)
  - Generates multiple query variations to improve recall
- `RAG_USE_HYDE` - Enable HyDE (Hypothetical Document Embeddings) (default: true)
  - Generates hypothetical answers for better retrieval
- `RAG_USE_RERANKING` - Enable cross-encoder reranking (default: true)
  - Re-ranks results using more accurate but slower models

#### Advanced RAG Settings

- `RAG_FETCH_MULTIPLIER` - Multiplier for fetching more candidates before filtering (default: 2)
- `RAG_MULTI_QUERY_MULTIPLIER` - Multiplier for multi-query expansion (default: 1.5)

### Chunking Configuration

- `CHUNKING_DEFAULT_STRATEGIES` - Comma-separated list of strategies to use (default: "single,tokens,author")
- `CHUNKING_TEMPORAL_WINDOW` - Time window in seconds for temporal chunking (default: 3600 = 1 hour)
- `CHUNKING_CONVERSATION_GAP` - Gap in seconds to detect conversation boundaries (default: 1800 = 30 minutes)
- `CHUNKING_WINDOW_SIZE` - Messages per window for sliding window strategy (default: 10)
- `CHUNKING_OVERLAP` - Overlapping messages between windows (default: 2)
- `CHUNKING_MAX_TOKENS` - Maximum tokens per chunk (default: 512)
- `CHUNKING_MIN_CHUNK_SIZE` - Minimum messages per chunk (default: 3)

### Chatbot Configuration

- `CHATBOT_CHANNEL_ID` - Specific channel ID for chatbot (0 = all channels, default: 0)
- `CHATBOT_MAX_HISTORY` - Maximum conversation history messages (default: 15)
- `CHATBOT_SESSION_TIMEOUT` - Session timeout in seconds (default: 1800 = 30 minutes)
- `CHATBOT_MAX_TOKENS` - Default max tokens for chatbot responses (default: 400)
- `CHATBOT_CHAT_MAX_TOKENS` - Max tokens for conversational chat (default: 250)
- `CHATBOT_RAG_MAX_TOKENS` - Max tokens for RAG responses (default: 1500)
- `CHATBOT_TEMPERATURE` - Temperature for chatbot responses (default: 0.8)
- `CHATBOT_USE_RAG` - Enable RAG in chatbot mode (default: true)
- `CHATBOT_RAG_THRESHOLD` - Similarity threshold for chatbot RAG (default: 0.01)
- `CHATBOT_RATE_LIMIT_MESSAGES` - Max messages per rate limit window (default: 10)
- `CHATBOT_RATE_LIMIT_WINDOW` - Rate limit window in seconds (default: 60)
- `CHATBOT_INCLUDE_CONTEXT_MESSAGES` - Number of context messages to include (default: 5)

### AI Provider Configuration

- `AI_DEFAULT_PROVIDER` - Default AI provider: "openai", "anthropic", or "gemini" (default: openai)
- `OPENAI_DEFAULT_MODEL` - Default OpenAI model (default: gpt-5-mini-2025-08-07)
- `ANTHROPIC_DEFAULT_MODEL` - Default Anthropic model (default: claude-haiku-4-5)
- `GEMINI_DEFAULT_MODEL` - Default Gemini model (default: gemini-2.5-flash)

### Social Credit System Configuration

- `SOCIAL_CREDIT_ENABLED` - Enable social credit system (default: true)
- `SOCIAL_CREDIT_INITIAL_MEAN` - Initial score mean for new users (default: 0)
- `SOCIAL_CREDIT_INITIAL_STD` - Initial score standard deviation (default: 200)
- `SOCIAL_CREDIT_PENALTY_ADMIN_COMMAND` - Penalty for unauthorized admin command attempts (default: -500)
- `SOCIAL_CREDIT_PENALTY_QUERY_FILTER` - Penalty for filtered queries (default: -500)
- `SOCIAL_CREDIT_DECAY_NEGATIVE` - Daily decay for negative scores (default: -10)
- `SOCIAL_CREDIT_GROWTH_POSITIVE` - Daily growth for positive scores (default: 10)

### Message Loading Configuration

- `MESSAGE_FETCH_DELAY` - Delay between message fetches in seconds (default: 0.2)
- `MESSAGE_FETCH_BATCH_SIZE` - Messages per batch when loading (default: 100)
- `MESSAGE_FETCH_PROGRESS_INTERVAL` - Progress update interval (default: 100)
- `MESSAGE_FETCH_MAX_RETRIES` - Maximum retries for failed fetches (default: 5)

### Security & Debugging

- `DEBUG_MODE` - Enable debug mode (default: false)
- `TEST_MODE` - Enable test mode (default: false)
- `LOG_LEVEL` - Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `BLACKLIST_IDS` - Comma-separated list of user IDs to exclude from AI usage

### Vector Store Configuration

- `VECTOR_STORE_PROVIDER` - Vector store provider (default: chroma)
  - Currently supports: chroma (ChromaDB)
  - Future: pinecone, weaviate support planned

## Architecture

Deep-Bot uses a modular, event-driven architecture designed for scalability and maintainability:

### Layer 1: Discord Gateway Layer
- Handles Discord WebSocket events
- Manages connection and reconnection
- Processes incoming messages and commands
- Built on discord.py library

### Layer 2: Command & Event Router
- Cog-based command system for modularity
- Separate cogs for different functionalities:
  - **Basic Cog**: Utility commands (ping, hello, info)
  - **RAG Cog**: Question-answering commands (!ask, !rag)
  - **Summary Cog**: Message summarization (!summary)
  - **Chatbot Cog**: Conversational AI with RAG
  - **Admin Cog**: Administrative commands
  - **Social Credit Cog**: Social credit system commands
- Event handlers for message processing

### Layer 3: AI Service Layer
- Provider abstraction supporting multiple AI providers
- Unified interface for OpenAI, Anthropic, and Gemini
- Handles API calls, retries, rate limiting
- Cost tracking and usage statistics
- Social credit tone injection

### Layer 4: RAG Pipeline
- **Query Validation**: Sanitizes and validates user queries
- **Query Enhancement**: Multi-query expansion, HyDE generation
- **Retrieval**: Hybrid search (vector + BM25)
- **Reranking**: Cross-encoder reranking for accuracy
- **Context Building**: Assembles context from retrieved chunks
- **Answer Generation**: LLM generates final answer

### Layer 5: Storage Layer
- **ChromaDB**: Vector database for embeddings
  - Stores dense embeddings for semantic search
  - Multiple collections per chunking strategy
  - Metadata indexing for filtering
- **SQLite**: Relational database for messages
  - Raw message storage
  - Metadata and timestamps
  - Chunking checkpoints
  - Usage statistics

### Data Flow

1. **Message Ingestion**: Discord messages → SQLite → Chunking → Embeddings → ChromaDB
2. **Query Processing**: User query → Validation → Enhancement → Retrieval → Reranking → Context → Answer
3. **Bot Knowledge**: Bot docs → Chunking → Embeddings → ChromaDB (separate collection)

## Features

### Conversation Memory Management

- **Session Management**: Channel-level conversation sessions with TTL expiration
- **Context Window Management**: Intelligent trimming to stay within token limits
- **Message History**: Maintains conversation context across multiple interactions
- **Rate Limiting**: Sliding window rate limiting to prevent abuse

### Search Features

- **Author Filtering**: Filter results by Discord users
- **Temporal Filtering**: Search within time ranges
- **Blacklist Support**: Exclude messages from blacklisted users
- **Similarity Thresholds**: Configurable relevance thresholds

### Performance Optimizations

- Async embedding generation using thread pools
- Database connection pooling
- LRU caches for frequently accessed data
- Batch processing for embeddings
- Background tasks for non-critical operations

## Usage Examples

### Asking Questions

```
!ask What did we discuss about the project yesterday?
!rag How does the authentication system work?
```

### Getting Summaries

```
!summary 100
```

### Checking Stats

```
!mystats
!chatbot_stats
```

### Admin Operations

```
!load_channel
!chunk_status
!rag_settings
```

## Limitations

- Bot can only answer questions about messages that have been indexed
- Messages must be loaded using `!load_channel` before they can be searched
- Rate limits apply to prevent API quota exhaustion
- Social credit system is optional and can be disabled

## How to Use

### Initial Setup (Admin)

1. **Load messages**: Use `!load_channel` to load messages from a channel
2. **Wait for chunking**: Check progress with `!chunk_status`
3. **Index bot knowledge**: Use `!reindex_bot_knowledge` to index bot documentation
4. **Configure RAG**: Adjust settings with `!rag_settings` and `!rag_set` if needed

### Using RAG Commands

1. **Ask questions**: Use `!ask <question>` or `!rag <question>`
2. **Be specific**: More specific questions get better answers
3. **Check sources**: Bot will cite sources when available
4. **Try variations**: If first query doesn't work, try rephrasing

### Using Chatbot Mode

1. **Start conversation**: Just mention the bot or use commands
2. **Maintain context**: Bot remembers conversation history
3. **Use RAG**: Bot automatically uses RAG when relevant
4. **Reset if needed**: Use `!chatbot_reset` to clear history

### Using Summary Command

1. **Specify count**: `!summary 100` for last 100 messages
2. **Wait for processing**: Summary generation takes a few seconds
3. **Check rate limits**: Limited to 3 summaries per 2 minutes
4. **Review results**: Summary includes key points and topics

## Social Credit System

Deep-Bot includes an optional social credit system that tracks user behavior and adjusts AI response tone accordingly.

### How It Works

- Users start with a randomized score (mean: 0, std: 200)
- Scores change based on user actions
- AI responses adjust tone based on score tiers
- Scores decay/grow over time automatically

### Score Tiers

- **Very High** (800+): Respectful, formal tone
- **High** (400-799): Friendly, helpful tone
- **Medium** (0-399): Standard conversational tone
- **Low** (-399 to -1): Slightly dismissive tone
- **Very Low** (-400 and below): Sarcastic, dismissive tone

### Penalties and Rewards

- **Penalties**: Unauthorized admin command attempts (-500), filtered queries (-500)
- **Rewards**: Normal usage, positive interactions (+10 daily growth)
- **Decay**: Negative scores decay by -10 per day

### Commands

- `!socialcredit view [@user]` - View scores
- `!socialcredit leaderboard` - See top/bottom users (Admin only)
- `!socialcredit give/take/set/reset` - Admin score management

## Troubleshooting

### RAG Not Finding Results

1. **Check if messages are loaded**: Use `!check_storage` to verify messages exist
2. **Check chunking status**: Use `!chunk_status` to see if chunking completed
3. **Verify collection exists**: Check ChromaDB collections
4. **Try different query**: Some queries may not match any content
5. **Lower similarity threshold**: Use `!rag_set similarity_threshold 0.005` for more results

### Messages Not Loading

1. **Check permissions**: Bot needs "Read Message History" permission
2. **Check rate limits**: Discord has rate limits, bot waits between fetches
3. **Check logs**: Look for error messages in bot logs
4. **Retry**: Use `!load_channel` again if it failed

### Chunking Issues

1. **Reset checkpoint**: Use `!reset_chunk_checkpoint` to force re-chunking
2. **Check strategy**: Verify chunking strategies are correct
3. **Check token limits**: Very long messages may cause issues
4. **Re-chunk**: Use `!rechunk` to re-process from checkpoint

### Performance Issues

1. **Reduce top_k**: Lower `RAG_DEFAULT_TOP_K` for faster searches
2. **Disable techniques**: Turn off reranking or multi-query if too slow
3. **Check embedding provider**: OpenAI embeddings are faster than local
4. **Reduce batch size**: Lower `EMBEDDING_BATCH_SIZE` if memory issues

### Bot Not Responding

1. **Check bot status**: Use `!ping` to verify bot is online
2. **Check permissions**: Bot needs "Send Messages" permission
3. **Check rate limits**: Bot may be rate limited
4. **Check logs**: Look for error messages
5. **Restart bot**: If all else fails, restart the bot

## Best Practices

### For Administrators

1. **Load channels gradually**: Don't load all channels at once, start with important ones
2. **Monitor chunking**: Use `!chunk_status` to track progress
3. **Adjust RAG settings**: Tune similarity thresholds and top_k for your use case
4. **Use hybrid search**: Enable hybrid search for better results
5. **Set appropriate rate limits**: Prevent abuse with rate limiting

### For Users

1. **Be specific**: More specific questions get better answers
2. **Use natural language**: The bot understands conversational queries
3. **Check context**: Bot answers based on indexed messages only
4. **Be patient**: Complex queries may take a few seconds
5. **Provide feedback**: Report issues to administrators

### Performance Optimization

1. **Use appropriate chunking strategies**: Match strategies to your query patterns
2. **Tune similarity thresholds**: Balance recall vs precision
3. **Monitor token usage**: Keep context sizes reasonable
4. **Use caching**: Bot caches embeddings and results
5. **Batch operations**: Load messages in batches, not all at once

## Deployment

### Requirements

- Python 3.12+
- Discord bot token
- OpenAI API key (minimum)
- Sufficient disk space for ChromaDB and SQLite
- Network access to Discord API and AI provider APIs

### Environment Setup

1. Create `.env` file with required variables
2. Install dependencies: `pip install -r requirements.txt`
3. Run bot: `python bot.py`
4. Load channels: Use `!load_channel` in Discord
5. Start asking questions: Use `!ask` or `!rag` commands

### Docker Deployment

The bot includes a Dockerfile for containerized deployment:
- Multi-stage build for optimized image size
- Includes all dependencies
- Configured for Railway deployment
- Handles Rust compilation for tokenizers

### Railway Deployment

The bot is configured for Railway deployment:
- Automatic builds from Git
- Environment variable configuration
- Persistent storage for databases
- Health checks and monitoring

## Advanced Features

### Query Enhancement

- **Multi-Query Expansion**: Generates multiple query variations automatically
- **HyDE**: Creates hypothetical answers to improve retrieval
- **Query Rewriting**: Improves query quality before search

### Hybrid Search

- Combines semantic (vector) and keyword (BM25) search
- Uses Reciprocal Rank Fusion (RRF) to merge results
- Provides better recall than either method alone
- Configurable weights for each method

### Reranking

- Uses cross-encoder models for more accurate ranking
- Slower but more precise than initial retrieval
- Re-ranks top candidates from hybrid search
- Significantly improves answer quality

### Session Management

- Maintains conversation context across messages
- Channel-level sessions with TTL expiration
- Intelligent context window management
- Prevents token limit overflow

## Error Handling

The bot includes comprehensive error handling:

- **Graceful degradation**: Continues working if optional features fail
- **Retry logic**: Automatically retries failed operations
- **Error messages**: Clear error messages for users
- **Logging**: Detailed logs for debugging
- **Validation**: Input validation prevents errors

## Security Features

- **Input sanitization**: Protects against injection attacks
- **Rate limiting**: Prevents abuse and API quota exhaustion
- **Blacklist support**: Exclude users from AI usage
- **Permission checks**: Admin commands require owner permissions
- **Secure storage**: API keys stored in environment variables

