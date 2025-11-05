# Deep Bot 🤖

An advanced **Discord RAG (Retrieval-Augmented Generation) chatbot** with conversational memory, user emulation, and debate analysis capabilities. Built for learning deep learning, RAG systems, and modern chatbot architecture.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Discord.py](https://img.shields.io/badge/discord.py-2.0+-blue.svg)](https://discordpy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Features

### **Core RAG System**
- 🔍 **Semantic Search** - Find relevant messages using vector similarity
- 📊 **Multi-Strategy Chunking** - Temporal, conversation-based, sliding window, token-aware
- 🧪 **Strategy Evaluation** - Compare chunking strategies with precision/recall metrics
- 💾 **Persistent Storage** - SQLite for raw messages, ChromaDB for vector embeddings
- ⚡ **Optimized Retrieval** - Top-K search with configurable similarity thresholds

### **Advanced Chatbot Features**
- 💬 **Conversational Memory** - Multi-turn conversations with context awareness
- 🎭 **User Emulation** - Mimic speech patterns and personality of server members
- 🎓 **Debate Analysis** - Detect logical fallacies, fact-check claims, suggest improvements
- 📚 **Source Citations** - All answers include references to chat history

### **Production Ready**
- 🐳 **Docker Support** - Complete Docker & Docker Compose setup
- 🚀 **Multiple Deployment Options** - Railway, Heroku, VPS, AWS
- 📈 **Monitoring & Logging** - Health checks, automated backups
- 🔒 **Security Best Practices** - Environment variables, rate limiting, access control

---

## 🎯 What Can Deep Bot Do?

### 1. Answer Questions from Chat History
```
!ask What trips have people taken in the past 2 years?
```
Bot searches chat history and provides relevant information with sources.

### 2. Have Natural Conversations
```
User: !chat What's the best database for our project?
Bot: Based on the chat history, I see MongoDB and PostgreSQL discussed...

User: !chat Tell me more about MongoDB
Bot: [Remembers context and continues conversation naturally]
```

### 3. Emulate User Speech Patterns
```
!emulate @Alice what do you think about the new API design?
```
Bot responds in Alice's style (vocabulary, tone, emoji usage, etc.)

### 4. Analyze Arguments & Debates
```
!analyze_debate We should use MongoDB because everyone uses it
```
Bot identifies logical fallacies, checks facts against chat history, and suggests improvements.

### 5. Evaluate & Compare Strategies
```
!evaluate_strategies
```
Scientifically compare chunking strategies using retrieval metrics.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Discord Bot                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼────────┐         ┌────────▼────────┐
│  Message Loader│         │  RAG Query      │
│  (Phase 2)     │         │  Pipeline       │
│  - Rate limit  │         │  (Phase 10-13)  │
│  - Batch fetch │         │  - Retrieve     │
│  - Checkpoint  │         │  - Generate     │
└───────┬────────┘         │  - Emulate      │
        │                  │  - Analyze      │
┌───────▼────────┐         └────────┬────────┘
│ SQLite Storage │                  │
│  (Phase 1)     │         ┌────────▼────────┐
│  - Raw msgs    │         │ Conversation    │
│  - Checkpoints │         │ Memory Service  │
└────────────────┘         │  (Phase 11)     │
                           └────────┬────────┘
        ┌───────────────────────────┘
        │
┌───────▼────────┐         ┌─────────────────┐
│ Chunking       │────────▶│  Embeddings     │
│ Service        │         │  (Phase 3)      │
│  (Phase 4)     │         │  - Sentence     │
│  - Temporal    │         │    Transformers │
│  - Sliding     │         │  - OpenAI       │
│  - Token-aware │         └────────┬────────┘
└───────┬────────┘                  │
        │                  ┌────────▼────────┐
        └─────────────────▶│  ChromaDB       │
                           │  (Phase 5-6)    │
                           │  - Vector store │
                           │  - Similarity   │
                           └─────────────────┘
```

---

## 📁 Project Structure

```
deep-bot/
├── 📄 Implementation Phases
│   ├── IMPLEMENTATION_GUIDE.md      # Main guide with learning path
│   ├── PHASE_01.md                  # Message Storage (SQLite)
│   ├── PHASE_02.md                  # Rate Limiting & Loading
│   ├── PHASE_03.md                  # Embedding Service
│   ├── PHASE_04.md                  # Chunking Strategies
│   ├── PHASE_05.md                  # Vector Store Abstraction
│   ├── PHASE_06.md                  # Multi-Strategy Storage
│   ├── PHASE_06_5.md                # Strategy Evaluation ⭐
│   ├── PHASE_07.md                  # Bot Commands
│   ├── PHASE_08.md                  # Summary Enhancement
│   ├── PHASE_09.md                  # Configuration
│   ├── PHASE_10.md                  # RAG Query Pipeline ⭐
│   ├── PHASE_11.md                  # Conversational Chatbot ⭐
│   ├── PHASE_12.md                  # User Emulation ⭐
│   └── PHASE_13.md                  # Debate Analyzer ⭐
│
├── 📄 Deployment
│   └── DEPLOYMENT_GUIDE.md          # Docker, VPS, Cloud deployment
│
├── 🐍 Core Application
│   ├── bot.py                       # Main bot entry point
│   ├── config.py                    # Configuration management
│   └── requirements.txt             # Python dependencies
│
├── 🔧 Services
│   ├── services/
│   │   ├── message_storage.py       # SQLite message storage
│   │   ├── message_loader.py        # Discord message fetching
│   │   ├── chunking_service.py      # Message chunking strategies
│   │   ├── embedding_service.py     # Embedding providers
│   │   ├── chunked_memory_service.py # Vector store management
│   │   ├── rag_query_service.py     # RAG pipeline
│   │   ├── conversational_rag_service.py # Conversational RAG
│   │   ├── conversation_memory.py    # Conversation state
│   │   ├── user_emulation_service.py # User style emulation
│   │   ├── debate_analyzer_service.py # Argument analysis
│   │   └── evaluation_service.py     # Strategy evaluation
│
├── 🎮 Discord Commands
│   ├── cogs/
│   │   ├── basic.py                 # Basic commands
│   │   ├── admin.py                 # Admin/owner commands
│   │   ├── summary.py               # Summary commands
│   │   └── chatbot.py               # Chatbot features
│
├── 💾 Data (auto-created)
│   ├── data/
│   │   ├── raw_messages/            # SQLite databases
│   │   └── chroma/                  # Vector embeddings
│
└── 🔒 Configuration
    ├── .env                         # Environment variables (not committed)
    ├── .env.example                 # Template
    └── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Discord Bot Token ([Get one here](https://discord.com/developers/applications))
- OpenAI or Anthropic API Key (optional for cloud embeddings)

### 1. Clone & Install

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/deep-bot.git
cd deep-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

Create `.env` file:

```bash
# Required
DISCORD_TOKEN=your_discord_bot_token
BOT_OWNER_ID=your_discord_user_id

# AI Provider (choose one)
OPENAI_API_KEY=your_openai_key  # For cloud embeddings/LLM
# OR
ANTHROPIC_API_KEY=your_claude_key

# Optional - use local embeddings (free!)
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Bot Configuration
BOT_PREFIX=!
RAG_TOP_K=5
RAG_DEFAULT_STRATEGY=token_aware
```

### 3. Run

```bash
python bot.py
```

### 4. Start Using

In Discord:
```
# Load messages into bot
!chunk_channel 1000

# Ask questions
!ask What have people discussed about databases?

# Chat naturally
!chat Tell me about the Japan trip

# Emulate users
!emulate @Alice what do you think?

# Analyze arguments
!analyze_debate We should use MongoDB because it's popular
```

---

## 📚 Documentation

### For Learning & Implementation
- **[Implementation Guide](./IMPLEMENTATION_GUIDE.md)** - Complete learning path with 13 phases
- **[Deployment Guide](./DEPLOYMENT_GUIDE.md)** - Production deployment with Docker

### Phase-by-Phase Guides
Each phase has detailed documentation with:
- Learning objectives
- Design principles
- Step-by-step implementation
- Code examples
- Common pitfalls
- Performance considerations

**Start here:** [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)

---

## 🎮 Command Reference

### Core RAG Commands

| Command | Description | Example |
|---------|-------------|---------|
| `!chunk_channel [limit]` | Load & chunk messages | `!chunk_channel 1000` |
| `!chunk_stats` | Show chunking statistics | `!chunk_stats` |
| `!ask <question>` | Query chat history | `!ask What trips have people taken?` |
| `!evaluate_strategies` | Compare chunking strategies | `!evaluate_strategies` |

### Conversational Chatbot

| Command | Description | Example |
|---------|-------------|---------|
| `!chat <message>` | Natural conversation | `!chat What's the best framework?` |
| `!clear_chat` | Clear conversation history | `!clear_chat` |
| `!chat_history` | View conversation | `!chat_history` |

### User Emulation

| Command | Description | Example |
|---------|-------------|---------|
| `!emulate @user <context>` | Emulate user's response | `!emulate @Alice what do you think?` |
| `!analyze_user @user` | Analyze communication style | `!analyze_user @Bob` |

### Debate Analysis

| Command | Description | Example |
|---------|-------------|---------|
| `!analyze_debate <statement>` | Analyze argument quality | `!analyze_debate Python is better` |
| `!compare_arguments <args>` | Compare two arguments | See Phase 13 docs |

### Utility

| Command | Description |
|---------|-------------|
| `!summary [count]` | Summarize recent messages |
| `!ping` | Check bot latency |
| `!help` | Show all commands |

---

## 🛠️ Development

### Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Bot Framework** | discord.py | Discord integration |
| **Storage** | SQLite | Raw message storage |
| **Vector DB** | ChromaDB | Semantic search |
| **Embeddings** | sentence-transformers / OpenAI | Text embeddings |
| **LLM** | OpenAI GPT / Anthropic Claude | Text generation |
| **Deployment** | Docker / Docker Compose | Containerization |

### Key Design Patterns

- **Strategy Pattern** - Multiple chunking/embedding strategies
- **Factory Pattern** - Service creation from config
- **Adapter Pattern** - Vector store & embedding abstraction
- **Observer Pattern** - Progress callbacks
- **Repository Pattern** - Data access abstraction

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_message_storage.py

# With coverage
python -m pytest --cov=services tests/
```

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Create .env file with your credentials
cp .env.example .env
# Edit .env with your tokens

# Start bot
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop bot
docker-compose down
```

### Production Deployment

See **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** for:
- VPS deployment (DigitalOcean, Linode, AWS)
- Cloud platforms (Railway, Heroku)
- Monitoring & logging setup
- Backup strategies
- Security best practices

---

## 📊 Learning Objectives

This project demonstrates:

### RAG & NLP
- ✅ Retrieval-Augmented Generation architecture
- ✅ Vector embeddings & semantic search
- ✅ Chunking strategies for context preservation
- ✅ Evaluation metrics (Precision, Recall, F1, MRR)

### Software Engineering
- ✅ Design patterns (Strategy, Factory, Adapter)
- ✅ Modular architecture
- ✅ Configuration management
- ✅ Testing & evaluation frameworks

### AI/ML Applications
- ✅ Conversational AI with memory
- ✅ Style transfer (user emulation)
- ✅ Computational rhetoric (debate analysis)
- ✅ Multi-model integration

### DevOps
- ✅ Docker containerization
- ✅ Environment management
- ✅ Logging & monitoring
- ✅ Production deployment

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the existing code style
4. Add tests for new features
5. Update documentation
6. Submit a pull request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linters
black .
flake8 .

# Run tests
pytest
```

---

## 📈 Roadmap

### Phase 14: Hybrid Search (Planned)
- Combine vector search + keyword search
- BM25 + semantic similarity
- Reranking with cross-encoders

### Phase 15: Advanced Analytics (Planned)
- Conversation topic clustering
- User interaction patterns
- Sentiment analysis over time

### Phase 16: Multi-Modal (Planned)
- Image understanding (analyze screenshots)
- Voice message transcription
- Document parsing (PDFs, links)

---

## 🔒 Security & Privacy

### Best Practices Implemented
- ✅ Environment variables for secrets (never committed)
- ✅ Input validation & sanitization
- ✅ Rate limiting on API calls
- ✅ Role-based access control for sensitive commands
- ✅ Opt-out support for user emulation

### Privacy Considerations
- Message data stored locally (not sent to third parties except for embeddings/LLM)
- User emulation requires transparency (always labeled)
- No PII stored beyond Discord IDs
- Configurable data retention

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Troubleshooting

### Bot won't start
```bash
# Check logs
tail -f logs/bot.log

# Verify config
python -c "from config import Config; Config.validate()"

# Check dependencies
pip install -r requirements.txt --upgrade
```

### Out of memory
```bash
# Reduce batch sizes in .env
EMBEDDING_BATCH_SIZE=16
CHUNKING_MAX_TOKENS=256
```

### Slow responses
- Use local embeddings (`sentence-transformers`) instead of API
- Reduce `RAG_TOP_K` to 3
- Use `token_aware` chunking strategy

See **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** for more troubleshooting.

---

## 🙏 Acknowledgments

- [discord.py](https://github.com/Rapptz/discord.py) - Discord API wrapper
- [sentence-transformers](https://www.sbert.net/) - State-of-the-art embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) - GPT models & embeddings
- [Anthropic](https://www.anthropic.com/) - Claude models

---

## 📞 Support

- **Documentation**: [Implementation Guide](./IMPLEMENTATION_GUIDE.md)
- **Deployment**: [Deployment Guide](./DEPLOYMENT_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/deep-bot/issues)

---

**Built for learning deep learning, RAG systems, and chatbot architecture** 🚀

**Remember: Keep your tokens and API keys secure! Never share them publicly.**
