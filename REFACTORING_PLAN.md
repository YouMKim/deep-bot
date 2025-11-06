# Refactoring Plan: Domain-Based Architecture 🏗️

**Goal:** Reorganize codebase from flat `services/` structure to domain-based architecture for better maintainability and clarity.

**Estimated Time:** 2-3 hours

**Risk Level:** Low (mostly file moves + import updates)

---

## Current Structure (Before)

```
deep-bot/
├── bot.py
├── config.py
├── services/
│   ├── message_storage.py        # Storage
│   ├── memory_service.py         # RAG
│   ├── message_loader.py         # Data fetching
│   ├── chunking_service.py       # RAG
│   ├── embedding_service.py      # RAG
│   ├── vector_store_base.py      # RAG
│   ├── vector_store_chroma.py    # RAG
│   ├── vector_store_factory.py   # RAG
│   ├── chunked_memory_service.py # RAG
│   └── ai_service.py             # AI/LLM
├── cogs/
│   ├── basic.py
│   ├── admin.py
│   ├── summary.py
│   └── ...
└── utils/
    ├── discord_utils.py
    └── ...
```

**Problems:**
- ❌ All services mixed together (15+ files in one folder)
- ❌ No clear separation between RAG, Discord, and AI concerns
- ❌ Hard to see architectural boundaries
- ❌ Difficult to navigate for new contributors
- ❌ Testing boundaries unclear

---

## New Structure (After)

```
deep-bot/
├── bot.py                        # Main bot entry point
├── config.py                     # Global configuration
│
├── embedding/                    # 🆕 Embedding domain
│   ├── __init__.py              # Exports: EmbeddingProvider, EmbeddingFactory
│   ├── base.py                  # Abstract base class
│   ├── sentence_transformer.py  # Local embeddings
│   ├── openai.py               # OpenAI embeddings
│   └── factory.py              # Factory pattern
│
├── chunking/                    # 🆕 Chunking domain
│   ├── __init__.py             # Exports: ChunkingService, Chunk
│   ├── base.py                 # Chunk data structure
│   ├── service.py              # Main chunking service
│   └── strategies/             # Strategy implementations
│       ├── __init__.py
│       ├── temporal.py         # Time-window chunking
│       ├── conversation.py     # Conversation-gap chunking
│       ├── token_aware.py      # Token-limit aware chunking
│       └── sliding_window.py   # Sliding window chunking
│
├── retrieval/                   # 🆕 Vector retrieval domain
│   ├── __init__.py             # Exports: VectorStore, VectorStoreFactory
│   ├── base.py                 # Abstract base class
│   ├── factory.py              # Factory pattern
│   └── providers/              # Provider implementations
│       ├── __init__.py
│       ├── chroma.py          # ChromaDB adapter
│       ├── pinecone.py        # Pinecone adapter (future)
│       └── qdrant.py          # Qdrant adapter (future)
│
├── storage/                     # 🆕 Data persistence domain
│   ├── __init__.py             # Exports: MessageStorage
│   ├── message_storage.py      # SQLite storage
│   └── checkpoint.py           # Checkpoint management (future split)
│
├── rag/                        # 🆕 RAG orchestration domain
│   ├── __init__.py            # Exports: RAGService, RAGPipeline
│   ├── pipeline.py            # Main RAG pipeline
│   ├── memory_service.py      # Chunked memory service
│   ├── reranking.py          # Reranking logic (Phase 15)
│   ├── query_optimization.py  # Query expansion (Phase 15)
│   └── strategies.py         # RAG strategies (Phase 16)
│
├── ai/                         # 🆕 AI/LLM domain
│   ├── __init__.py            # Exports: AIService
│   ├── service.py             # AI service abstraction
│   ├── openai.py             # OpenAI implementation
│   └── ollama.py             # Ollama implementation
│
├── security/                   # 🆕 Security domain
│   ├── __init__.py            # Exports: SecurityService, RateLimiter
│   ├── input_validator.py    # Input validation (Phase 3)
│   ├── rate_limiter.py       # Rate limiting (Phase 3)
│   ├── prompt_injection.py   # Prompt injection defense (Phase 18)
│   └── audit_log.py          # Security audit logging (Phase 18)
│
├── bot/                        # 🆕 Discord bot domain
│   ├── __init__.py
│   ├── cogs/                  # Discord cogs
│   │   ├── __init__.py
│   │   ├── basic.py
│   │   ├── admin.py
│   │   ├── summary.py
│   │   ├── mvp_chatbot.py    # Phase 2 MVP
│   │   └── ...
│   ├── loaders/               # Data fetching
│   │   ├── __init__.py
│   │   └── message_loader.py
│   └── utils/                 # Bot utilities
│       ├── __init__.py
│       ├── discord_utils.py
│       └── formatting.py
│
├── utils/                      # 🆕 General utilities (non-domain specific)
│   ├── __init__.py
│   ├── error_handler.py
│   ├── secure_logger.py
│   └── secrets_manager.py
│
└── data/                       # Data directories (unchanged)
    ├── raw_messages/
    ├── chroma/
    └── ...
```

---

## Migration Steps (In Order)

### Step 1: Create New Directory Structure

```bash
# Create new directories
mkdir -p embedding
mkdir -p chunking/strategies
mkdir -p retrieval/providers
mkdir -p storage
mkdir -p rag
mkdir -p ai
mkdir -p security
mkdir -p bot/cogs bot/loaders bot/utils
mkdir -p utils

# Create __init__.py files
touch embedding/__init__.py
touch chunking/__init__.py chunking/strategies/__init__.py
touch retrieval/__init__.py retrieval/providers/__init__.py
touch storage/__init__.py
touch rag/__init__.py
touch ai/__init__.py
touch security/__init__.py
touch bot/__init__.py bot/cogs/__init__.py bot/loaders/__init__.py bot/utils/__init__.py
touch utils/__init__.py
```

### Step 2: Move Embedding Files

**Files to move:**
```bash
# From services/embedding_service.py → Split into:
services/embedding_service.py → embedding/base.py         (EmbeddingProvider)
services/embedding_service.py → embedding/sentence_transformer.py  (SentenceTransformerEmbedder)
services/embedding_service.py → embedding/openai.py       (OpenAIEmbedder)
services/embedding_service.py → embedding/factory.py      (EmbeddingServiceFactory)
```

**New `embedding/__init__.py`:**
```python
"""
Embedding domain - Convert text to vector embeddings.

Exports:
    - EmbeddingProvider: Abstract base class
    - SentenceTransformerEmbedder: Local embeddings
    - OpenAIEmbedder: Cloud embeddings
    - EmbeddingFactory: Factory for creating providers
"""

from embedding.base import EmbeddingProvider
from embedding.sentence_transformer import SentenceTransformerEmbedder
from embedding.openai import OpenAIEmbedder
from embedding.factory import EmbeddingFactory

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "EmbeddingFactory",
]
```

### Step 3: Move Chunking Files

**Files to move:**
```bash
# From services/chunking_service.py → Split into:
services/chunking_service.py → chunking/base.py           (Chunk class)
services/chunking_service.py → chunking/service.py        (ChunkingService)
services/chunking_service.py → chunking/strategies/temporal.py
services/chunking_service.py → chunking/strategies/conversation.py
services/chunking_service.py → chunking/strategies/token_aware.py
```

**New `chunking/__init__.py`:**
```python
"""
Chunking domain - Split messages into meaningful chunks.

Exports:
    - Chunk: Chunk data structure
    - ChunkingService: Main chunking service
    - TemporalStrategy: Time-window chunking
    - ConversationStrategy: Conversation-gap chunking
"""

from chunking.base import Chunk
from chunking.service import ChunkingService

__all__ = [
    "Chunk",
    "ChunkingService",
]
```

### Step 4: Move Retrieval Files

**Files to move:**
```bash
# Vector store files
services/vector_store_base.py → retrieval/base.py
services/vector_store_chroma.py → retrieval/providers/chroma.py
services/vector_store_factory.py → retrieval/factory.py
```

**New `retrieval/__init__.py`:**
```python
"""
Retrieval domain - Vector storage and similarity search.

Exports:
    - VectorStore: Abstract base class
    - ChromaVectorStore: ChromaDB adapter
    - VectorStoreFactory: Factory for creating stores
"""

from retrieval.base import VectorStore
from retrieval.providers.chroma import ChromaVectorStore
from retrieval.factory import VectorStoreFactory

__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "VectorStoreFactory",
]
```

### Step 5: Move Storage Files

**Files to move:**
```bash
services/message_storage.py → storage/message_storage.py
```

**New `storage/__init__.py`:**
```python
"""
Storage domain - Data persistence layer.

Exports:
    - MessageStorage: SQLite message storage
"""

from storage.message_storage import MessageStorage

__all__ = ["MessageStorage"]
```

### Step 6: Move RAG Files

**Files to move:**
```bash
services/chunked_memory_service.py → rag/memory_service.py
services/memory_service.py → rag/pipeline.py  # If exists
```

**New `rag/__init__.py`:**
```python
"""
RAG domain - Retrieval-Augmented Generation orchestration.

Exports:
    - ChunkedMemoryService: RAG memory service
    - RAGPipeline: Complete RAG pipeline (Phase 10)
"""

from rag.memory_service import ChunkedMemoryService

__all__ = [
    "ChunkedMemoryService",
]
```

### Step 7: Move AI Service Files

**Files to move:**
```bash
services/ai_service.py → ai/service.py
```

**New `ai/__init__.py`:**
```python
"""
AI domain - Language model abstraction.

Exports:
    - AIService: Abstract AI service
"""

from ai.service import AIService

__all__ = ["AIService"]
```

### Step 8: Move Bot Files

**Files to move:**
```bash
cogs/* → bot/cogs/*
services/message_loader.py → bot/loaders/message_loader.py
utils/discord_utils.py → bot/utils/discord_utils.py
```

**New `bot/__init__.py`:**
```python
"""
Bot domain - Discord bot commands and integrations.
"""

# No exports needed - bot uses cogs
```

### Step 9: Move Security Files (Phase 3 & 18)

**Files to create/move:**
```bash
# From Phase 3 & 18 implementations
utils/input_validator.py → security/input_validator.py
utils/rate_limiter.py → security/rate_limiter.py
utils/prompt_injection.py → security/prompt_injection.py
```

**New `security/__init__.py`:**
```python
"""
Security domain - Input validation, rate limiting, and security.

Exports:
    - InputValidator: Input validation
    - RateLimiter: Rate limiting
    - PromptInjectionDetector: Prompt injection defense
"""

from security.input_validator import InputValidator, query_validator
from security.rate_limiter import RateLimiter, rate_limiter

__all__ = [
    "InputValidator",
    "query_validator",
    "RateLimiter",
    "rate_limiter",
]
```

### Step 10: Move General Utils

**Files to move:**
```bash
# Keep only non-domain specific utilities
utils/error_handler.py → utils/error_handler.py  (stays)
utils/secure_logger.py → utils/secure_logger.py  (stays)
utils/secrets_manager.py → utils/secrets_manager.py  (stays)
```

---

## Import Statement Changes

### Before (Old Imports)
```python
# Old imports
from services.embedding_service import EmbeddingServiceFactory
from services.chunking_service import ChunkingService
from services.vector_store_factory import VectorStoreFactory
from services.message_storage import MessageStorage
from services.chunked_memory_service import ChunkedMemoryService
from services.ai_service import AIService
from utils.discord_utils import format_discord_message
```

### After (New Imports)
```python
# New imports - cleaner and more explicit
from embedding import EmbeddingFactory
from chunking import ChunkingService
from retrieval import VectorStoreFactory
from storage import MessageStorage
from rag import ChunkedMemoryService
from ai import AIService
from bot.utils.discord_utils import format_discord_message
```

---

## Files That Need Import Updates

### Phase Documents to Update:
- [x] PHASE_01.md - Storage imports
- [x] PHASE_02.md - Message loader imports
- [x] PHASE_03.md - Embedding imports
- [x] PHASE_04.md - Chunking imports
- [x] PHASE_05.md - Vector store imports
- [x] PHASE_06.md - Memory service imports
- [x] PHASE_07.md - Bot integration imports
- [x] PHASE_09.md - All imports
- [x] PHASE_10.md - RAG pipeline imports

### Code Files to Update:
- [x] `bot.py` - All service imports
- [x] `bot/cogs/admin.py` - All imports
- [x] `bot/cogs/summary.py` - Storage, AI imports
- [x] `bot/cogs/mvp_chatbot.py` - AI imports
- [x] `bot/loaders/message_loader.py` - Storage imports
- [x] `rag/memory_service.py` - Embedding, retrieval imports
- [x] `rag/pipeline.py` - All RAG imports

### New Phase Files to Update:
- [x] PHASE_02_MVP.md - MVP chatbot imports
- [x] PHASE_03_SECURITY.md - Security imports

---

## Migration Script

Create `scripts/refactor_structure.py`:

```python
#!/usr/bin/env python3
"""
Refactoring migration script.

Usage:
    python scripts/refactor_structure.py --dry-run  # Preview changes
    python scripts/refactor_structure.py            # Execute migration
"""

import os
import shutil
from pathlib import Path

# Migration map: (source, destination)
FILE_MOVES = [
    # Embedding domain
    ("services/embedding_service.py", "embedding/"),

    # Chunking domain
    ("services/chunking_service.py", "chunking/"),

    # Retrieval domain
    ("services/vector_store_base.py", "retrieval/base.py"),
    ("services/vector_store_chroma.py", "retrieval/providers/chroma.py"),
    ("services/vector_store_factory.py", "retrieval/factory.py"),

    # Storage domain
    ("services/message_storage.py", "storage/message_storage.py"),

    # RAG domain
    ("services/chunked_memory_service.py", "rag/memory_service.py"),

    # AI domain
    ("services/ai_service.py", "ai/service.py"),

    # Bot domain
    ("services/message_loader.py", "bot/loaders/message_loader.py"),
    ("cogs/", "bot/cogs/"),
    ("utils/discord_utils.py", "bot/utils/discord_utils.py"),
]

def create_directory_structure():
    """Create new directory structure."""
    dirs = [
        "embedding",
        "chunking/strategies",
        "retrieval/providers",
        "storage",
        "rag",
        "ai",
        "security",
        "bot/cogs",
        "bot/loaders",
        "bot/utils",
        "utils",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py
        (Path(dir_path) / "__init__.py").touch(exist_ok=True)
        print(f"✅ Created {dir_path}/__init__.py")

def move_file(src, dst, dry_run=False):
    """Move a file or directory."""
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        print(f"⚠️  Source not found: {src}")
        return

    if dry_run:
        print(f"📋 Would move: {src} → {dst}")
    else:
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

        print(f"✅ Moved: {src} → {dst}")

def update_imports_in_file(file_path, import_map):
    """Update imports in a Python file."""
    if not file_path.exists():
        return

    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Replace imports
    for old_import, new_import in import_map.items():
        content = content.replace(old_import, new_import)

    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Updated imports in {file_path}")

def main(dry_run=False):
    """Run migration."""
    print("🚀 Starting refactoring migration...\n")

    # Step 1: Create directory structure
    print("📁 Creating directory structure...")
    create_directory_structure()
    print()

    # Step 2: Move files
    print("📦 Moving files...")
    for src, dst in FILE_MOVES:
        move_file(src, dst, dry_run=dry_run)
    print()

    # Step 3: Update imports (only if not dry run)
    if not dry_run:
        print("🔧 Updating imports...")
        import_map = {
            "from services.embedding_service import": "from embedding import",
            "from services.chunking_service import": "from chunking import",
            "from services.vector_store_factory import": "from retrieval import",
            "from services.vector_store_base import": "from retrieval.base import",
            "from services.message_storage import": "from storage import",
            "from services.chunked_memory_service import": "from rag import",
            "from services.ai_service import": "from ai import",
            "from utils.discord_utils import": "from bot.utils.discord_utils import",
        }

        # Update all Python files
        for py_file in Path(".").rglob("*.py"):
            if "venv" not in str(py_file) and ".git" not in str(py_file):
                update_imports_in_file(py_file, import_map)
        print()

    print("✅ Migration complete!")

    if dry_run:
        print("\n⚠️  This was a dry run. Run without --dry-run to execute.")

if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
```

---

## Testing After Migration

### 1. Verify Imports
```python
# Test all imports work
python -c "from embedding import EmbeddingFactory; print('✅ embedding')"
python -c "from chunking import ChunkingService; print('✅ chunking')"
python -c "from retrieval import VectorStoreFactory; print('✅ retrieval')"
python -c "from storage import MessageStorage; print('✅ storage')"
python -c "from rag import ChunkedMemoryService; print('✅ rag')"
```

### 2. Run Bot
```bash
python bot.py
# Should start without import errors
```

### 3. Test Commands
```
In Discord:
!ping          # Test basic command
!help          # Test help
!chunk_stats   # Test RAG imports
```

### 4. Run Tests (if any)
```bash
pytest tests/
```

---

## Benefits of New Structure

### ✅ Clear Architecture
```
User Query → bot/ → rag/ → retrieval/ → embedding/
                      ↓
                  storage/
```

### ✅ Domain Isolation
- Each domain has clear boundaries
- Easy to swap implementations
- Testing boundaries explicit

### ✅ Better Navigation
```
"Where's embedding code?" → embedding/
"How does chunking work?" → chunking/strategies/
"What retrieval providers exist?" → retrieval/providers/
```

### ✅ Scalability
```
# Adding new provider:
retrieval/providers/new_provider.py

# Adding new chunking strategy:
chunking/strategies/new_strategy.py
```

### ✅ Team-Friendly
- New contributors understand structure immediately
- Clear ownership per domain
- Parallel development easier

---

## Rollback Plan (If Needed)

If something breaks:

1. **Git revert:**
   ```bash
   git revert HEAD
   git push -f
   ```

2. **Manual rollback:**
   - Keep backup of `services/` folder
   - Restore old structure
   - Revert import changes

**Recommendation:** Test thoroughly on a branch before merging to main.

---

## Timeline

### Phase 1: Preparation (30 min)
- ✅ Review this plan
- ✅ Create feature branch
- ✅ Backup current code

### Phase 2: File Migration (60 min)
- Run migration script
- Manually split files (embedding, chunking)
- Create __init__.py files

### Phase 3: Import Updates (45 min)
- Update all imports
- Fix any missed references
- Update phase documents

### Phase 4: Testing (30 min)
- Test imports
- Run bot
- Test commands
- Verify functionality

### Phase 5: Documentation (15 min)
- Update IMPLEMENTATION_GUIDE.md
- Update README.md (if needed)
- Commit and push

**Total: 2-3 hours**

---

## Questions Before Starting?

1. Do you want to do this incrementally (one domain at a time) or all at once?
2. Should we create a backup branch first?
3. Any specific concerns about breaking existing code?

**Ready to start? Let me know and I'll begin the migration!** 🚀
