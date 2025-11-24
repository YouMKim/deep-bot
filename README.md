# Deep-Bot 🤖

A Discord bot with Retrieval-Augmented Generation (RAG) capabilities, multi-provider AI integration, and conversation memory management.

## 🎯 Overview

Deep-Bot is a Discord bot that indexes your server's conversation history for search and question-answering. It uses vector embeddings, semantic search, hybrid retrieval, and language models to answer questions based on your server's conversation history.

## 🏗️ Architecture

Deep-Bot is built on a modular, event-driven architecture designed for scalability and extensibility. The system is organized into distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Discord Gateway Layer                      │
│              (discord.py event handlers)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Command & Event Router                    │
│              (Cog-based command system)                     │
└──────┬──────────┬──────────┬──────────┬──────────┬─────────┘
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
   │Basic │  │Chat  │  │ RAG  │  │Summary│  │Admin │
   │ Cog  │  │ Bot  │  │ Cog  │  │  Cog │  │ Cog  │
   └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
      │         │          │         │         │
      └─────────┴──────────┴─────────┴─────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Service Layer                         │
│  (OpenAI | Anthropic | Gemini Provider Abstraction)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              RAG Pipeline (Retrieval-Augmented Gen)          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Retrieve │→ │  Filter  │→ │  Build   │→ │ Generate │   │
│  │  Chunks  │  │  Results │  │ Context  │  │  Answer  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────┐            ┌──────────────┐
│   Vector     │            │   Message    │
│   Storage    │            │   Storage    │
│  (ChromaDB)  │            │   (SQLite)   │
└──────────────┘            └──────────────┘
```

## 🔄 System Flow

### Message Ingestion Pipeline

```
Discord Message Event
        │
        ▼
┌───────────────────────┐
│  Message Preprocessing│
│  - Sanitization       │
│  - Author extraction  │
│  - Metadata capture   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Message Storage     │
│   (SQLite Database)   │
│  - Raw message data   │
│  - Timestamps         │
│  - Author info        │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Chunking Service    │
│  - Temporal windows   │
│  - Author grouping    │
│  - Token-based splits │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Embedding Generation │
│  - SentenceTransformers│
│  - OpenAI Embeddings  │
│  - Batch processing   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Vector Storage      │
│   (ChromaDB)          │
│  - Dense embeddings   │
│  - Metadata indexing  │
│  - Collection mgmt    │
└───────────────────────┘
```

### RAG Query Pipeline

```
User Query
    │
    ▼
┌───────────────────────┐
│  Query Validation     │
│  - Length checks      │
│  - Content filtering  │
│  - Intent detection   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Query Enhancement    │
│  - Multi-query exp.   │
│  - HyDE generation    │
│  - Query rewriting    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Hybrid Retrieval     │
│  ┌─────────┐ ┌──────┐│
│  │ Vector  │ │ BM25 ││
│  │ Search  │ │Search││
│  └────┬────┘ └───┬──┘│
│       └─────┬────┘   │
│             │         │
│      ┌──────▼──────┐  │
│      │ Reciprocal  │  │
│      │ Rank Fusion │  │
│      └──────┬──────┘  │
└─────────────┼─────────┘
              │
              ▼
┌───────────────────────┐
│  Reranking            │
│  - Cross-encoder      │
│  - Relevance scoring  │
│  - Top-K selection    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Context Assembly     │
│  - Metadata injection │
│  - Token budgeting    │
│  - Chunk ordering     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  LLM Generation       │
│  - Prompt engineering │
│  - Temperature tuning │
│  - Response streaming │
└───────────┬───────────┘
            │
            ▼
      Final Answer
```

## 🚀 Core Capabilities

### RAG (Retrieval-Augmented Generation)

Deep-Bot implements a RAG system with multiple retrieval strategies:

- **Vector Semantic Search**: Uses dense embeddings (SentenceTransformers, OpenAI) to find semantically similar content
- **BM25 Keyword Search**: Keyword-based information retrieval
- **Hybrid Search**: Combines vector and BM25 using Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: Reranking using cross-encoder models
- **Multi-Query Expansion**: Generates multiple query variations
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers for retrieval

### Multi-Provider AI Integration

Seamlessly supports multiple AI providers with a unified interface:

- **OpenAI**: GPT-5, GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude Sonnet 4.5, Claude Haiku 4.5, Claude Opus 4.1
- **Google**: Gemini 2.5 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash

Provider abstraction allows hot-swapping models without code changes.

### Chunking Strategies

Multiple chunking strategies for different use cases:

- **Temporal Chunking**: Groups messages by time windows (default: 1 hour)
- **Author-Based Chunking**: Groups messages by author for conversation threads
- **Token-Based Chunking**: Splits by token count for optimal context windows
- **Hybrid Strategies**: Combines multiple approaches for best results

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

## 🛠️ Technical Stack

### Core Technologies

- **Python 3.12+**: Modern Python with async/await support
- **discord.py**: Discord API wrapper with full async support
- **ChromaDB**: Vector database for embedding storage and retrieval
- **SQLite**: Relational database for message metadata and statistics
- **SentenceTransformers**: Local embedding generation (all-MiniLM-L6-v2)
- **rank-bm25**: BM25 keyword search implementation

### AI & ML Libraries

- **OpenAI SDK**: GPT model integration
- **Anthropic SDK**: Claude model integration
- **Google Generative AI**: Gemini model integration
- **sentence-transformers**: Embedding models
- **transformers**: Cross-encoder reranking models

### Data Processing

- **tiktoken**: Fast token counting for OpenAI models
- **numpy**: Numerical operations for embeddings
- **asyncio**: Asynchronous I/O for high concurrency

## 📊 Performance Optimizations

Performance optimizations implemented:

- **Async Embedding Generation**: Non-blocking embedding computation using thread pools
- **Connection Pooling**: Database connection management
- **Caching**: LRU caches for frequently accessed data
- **Batch Processing**: Batch embedding generation
- **Background Tasks**: Fire-and-forget operations for non-critical tasks
- **Vector Search**: ChromaDB queries with indexing

## 🔐 Security & Privacy

- **Local-First Architecture**: All data stored locally, no external data transmission
- **API Key Management**: Secure environment variable handling
- **User Blacklisting**: Configurable user exclusion
- **Rate Limiting**: Prevents abuse and API quota exhaustion
- **Input Sanitization**: Protects against injection attacks
- **Error Handling**: Graceful degradation on failures

## 📈 Scalability Considerations

The architecture supports horizontal scaling:

- **Stateless Design**: Bot instances can be scaled independently
- **Database Abstraction**: Easy migration to PostgreSQL/MySQL for production
- **Vector DB Scaling**: ChromaDB can be replaced with production vector DBs (Pinecone, Weaviate)
- **Async Architecture**: Handles high concurrency efficiently
- **Resource Management**: Configurable limits prevent resource exhaustion

## 🧪 Testing

Comprehensive test suite covering:

- Unit tests for core components
- Integration tests for RAG pipeline
- End-to-end tests for Discord interactions
- Performance benchmarks
- Embedding quality validation

## 📝 Configuration

Deep-Bot is highly configurable via environment variables:

### Required Configuration
- `DISCORD_TOKEN`: Discord bot token
- `OPENAI_API_KEY`: At least one AI provider API key

### AI Provider Configuration
- `AI_DEFAULT_PROVIDER`: Primary AI provider (openai, anthropic, gemini)
- `AI_DEFAULT_MODEL`: Model to use (e.g., gpt-5-mini-2025-08-07)

### RAG Configuration
- `RAG_DEFAULT_TOP_K`: Number of chunks to retrieve (default: 15)
- `RAG_DEFAULT_SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.01)
- `RAG_USE_HYBRID_SEARCH`: Enable hybrid search (default: True)
- `RAG_USE_RERANKING`: Enable cross-encoder reranking (default: True)
- `RAG_USE_MULTI_QUERY`: Enable multi-query expansion (default: True)
- `RAG_USE_HYDE`: Enable HyDE (default: True)

### Chatbot Configuration
- `CHATBOT_CHANNEL_ID`: Channel ID for chatbot responses
- `CHATBOT_MAX_HISTORY`: Maximum conversation history length
- `CHATBOT_SESSION_TIMEOUT`: Session expiration time (seconds)
- `CHATBOT_CHAT_MAX_TOKENS`: Max tokens for chat responses
- `CHATBOT_RAG_MAX_TOKENS`: Max tokens for RAG responses

### Cronjob Configuration
- `SNAPSHOT_CHANNEL_ID`: Channel ID where snapshot messages will be posted (optional)

## 🚀 Quick Start

```bash
# Clone repository
git clone <repository-url>
cd deep-bot

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run bot
python bot.py
```

## 📚 Usage Examples

### Basic Commands
```
!ping          # Check bot latency
!hello         # Get AI-generated greeting
!help          # Show available commands
```

### RAG Search
```
!ask What did we discuss about the project timeline?
!ask strategy:tokens Who mentioned wanting to learn Python?
!ask @username What did they say about the meeting?
```

### Summaries
```
!summary 50    # Summarize last 50 messages
!summary 100   # Summarize last 100 messages
```

### Chatbot
Simply send messages in the configured chatbot channel. The bot will:
- Maintain conversation context
- Search history when needed
- Provide natural, conversational responses

## 🏛️ Architecture Decisions

### Why ChromaDB?
- Lightweight and embeddable (no separate server needed)
- Fast vector search performance
- Easy to migrate to production vector DBs
- Good Python integration

### Why SQLite?
- Zero-configuration database
- Perfect for single-instance deployments
- Easy to migrate to PostgreSQL for production
- Excellent performance for read-heavy workloads

### Why Multiple Chunking Strategies?
- Different strategies optimize for different query types
- Temporal chunking: Good for "what happened when" queries
- Author chunking: Good for "what did X say" queries
- Token chunking: Good for general semantic search

### Why Hybrid Search?
- Vector search: Excellent for semantic similarity
- BM25 search: Excellent for exact keyword matching
- Combined: Best of both worlds with RRF fusion

## 🔬 Features

### Query Enhancement
- **Multi-Query Expansion**: Generates 3-5 query variations
- **HyDE**: Generates hypothetical answers for retrieval
- **Query Rewriting**: Improves query clarity

### Reranking
- **Cross-Encoder Models**: More accurate than bi-encoders but slower
- **Relevance Scoring**: Combines multiple signals for ranking
- **Top-K Selection**: Configurable result set size

### Embedding Strategies
- **Local Embeddings**: SentenceTransformers for local embedding generation
- **OpenAI Embeddings**: OpenAI API embeddings (costs money)
- **Caching**: Embeddings cached to reduce computation

## 📊 Monitoring & Observability

- **Structured Logging**: JSON-formatted logs for easy parsing
- **Cost Tracking**: Per-user AI usage and cost tracking
- **Performance Metrics**: Latency tracking for all operations
- **Error Tracking**: Comprehensive error logging with stack traces


## ⏰ Cronjob Setup (Railway)

Deep-Bot includes a cronjob script that runs two tasks:

1. **Load Server**: Loads all messages not yet ingested from Discord and processes them through the chunking pipeline
2. **Snapshot**: Posts 5 messages from 1, 2, 3, 4, and 5 years ago on the current day to a configured channel

### Setting up the Cronjob on Railway

1. **Create a new service** in your Railway project:
   - Go to your Railway project dashboard
   - Click "New" → "Service"
   - Select "GitHub Repo" and choose your repository

2. **Configure the service**:
   - Use the same Dockerfile as your main bot service
   - Set the start command to: `python scripts/cronjob.py`
   - Configure the cron schedule (e.g., `0 2 * * *` for daily at 2 AM UTC)

3. **Set environment variables**:
   - Copy all environment variables from your main bot service
   - Ensure `SNAPSHOT_CHANNEL_ID` is set if you want snapshot functionality

4. **Deploy**:
   - Railway will automatically run the cronjob according to the schedule
   - Check logs to verify it's running correctly

### Cronjob Features

- **Load Server**: Automatically processes all channels, loading new messages and creating chunks
- **Snapshot**: Finds and posts historical messages from years past (skips if not found)
- **Error Handling**: Gracefully handles errors and continues processing
- **Logging**: Comprehensive logging for monitoring and debugging

### Manual Testing

You can test the cronjob script locally:

```bash
python scripts/cronjob.py
```

This will run both tasks and exit when complete.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
