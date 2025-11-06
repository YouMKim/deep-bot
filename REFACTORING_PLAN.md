# Refactoring Plan: Domain-Based Architecture 🏗️

**Goal:** Reorganize codebase from flat `services/` structure to domain-based architecture, **while preserving the excellent `core/` AI abstraction that already exists**.

**Estimated Time:** 1-2 hours (less than originally estimated!)

**Risk Level:** Low (mostly file moves within `services/`, `core/` stays untouched)

---

## ⚠️ IMPORTANT: Keep `core/` Unchanged!

Your existing `core/` directory is **already well-architected** with:
- ✅ Clean provider abstraction (`BaseAIProvider`)
- ✅ Proper data models (`AIRequest`, `AIResponse`, `TokenUsage`, `CostDetails`)
- ✅ Multiple provider support (OpenAI, Anthropic)
- ✅ Cost tracking built-in

**DO NOT MOVE OR MODIFY `core/`!** This refactoring is about reorganizing `services/` and `cogs/` only.

---

## Current Structure (Actual - After Review)

```
deep-bot/
├── bot.py
├── config.py
├── core/                         # ✅ KEEP AS IS - Already well-structured!
│   ├── __init__.py
│   ├── base_provider.py          # Abstract AI provider
│   ├── ai_models.py              # Data models (AIRequest, AIResponse, etc.)
│   └── providers/
│       ├── openai_provider.py    # OpenAI implementation
│       └── anthropic_provider.py # Anthropic implementation
├── services/                      # ⚠️ NEEDS REORGANIZATION
│   ├── ai_service.py             # Uses core/ providers
│   ├── message_storage.py        # Storage layer
│   ├── memory_service.py         # RAG memory
│   ├── message_loader.py         # Data fetching
│   └── user_ai_tracker.py        # Usage tracking
├── cogs/                          # ✅ Discord commands (move to bot/)
│   ├── admin.py
│   ├── basic.py
│   └── summary.py
└── utils/
    └── ...
```

**What's good:**
- ✅ `core/` has excellent AI abstraction (provider pattern, cost tracking, proper data models)
- ✅ Clean separation between AI providers and application logic

**What needs improvement:**
- ⚠️ `services/` mixes storage, RAG, and bot concerns
- ⚠️ No clear domain boundaries in `services/`
- ⚠️ `cogs/` should be under `bot/` domain

---

## New Structure (Revised Plan)

```
deep-bot/
├── bot.py
├── config.py
│
├── core/                         # ✅ KEEP UNCHANGED - AI abstraction
│   ├── __init__.py
│   ├── base_provider.py
│   ├── ai_models.py
│   └── providers/
│       ├── openai_provider.py
│       └── anthropic_provider.py
│
├── ai/                           # 🆕 Application-level AI services
│   ├── __init__.py
│   ├── service.py                # AIService (from services/ai_service.py)
│   └── tracker.py                # UserAITracker (from services/user_ai_tracker.py)
│
├── embedding/                    # 🆕 Embedding domain (Phase 3)
│   ├── __init__.py
│   ├── base.py
│   ├── sentence_transformer.py
│   ├── openai.py
│   └── factory.py
│
├── chunking/                     # 🆕 Chunking domain (Phase 4)
│   ├── __init__.py
│   ├── base.py
│   ├── service.py
│   └── strategies/
│
├── retrieval/                    # 🆕 Vector retrieval domain (Phase 5)
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   └── providers/
│       └── chroma.py
│
├── storage/                      # 🆕 Data persistence domain
│   ├── __init__.py
│   └── message_storage.py        # From services/
│
├── rag/                          # 🆕 RAG orchestration domain
│   ├── __init__.py
│   └── memory_service.py         # From services/
│
├── security/                     # 🆕 Security domain
│   ├── __init__.py
│   ├── input_validator.py
│   ├── rate_limiter.py
│   └── prompt_injection.py
│
├── bot/                          # 🆕 Discord bot domain
│   ├── __init__.py
│   ├── cogs/                     # From cogs/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── basic.py
│   │   └── summary.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── message_loader.py     # From services/
│   └── utils/
│       └── discord_utils.py
│
└── utils/                        # General utilities
    └── ...
```

**Key Changes:**
1. **Keep `core/` untouched** - It's already excellent!
2. **Add `ai/`** - Application-level AI services (uses `core/`)
3. **Reorganize `services/`** - Split by domain (storage, rag, bot)
4. **Move `cogs/` → `bot/cogs/`** - Clear Discord bot boundary

---

## Revised Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layers                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  bot/cogs/                                                   │
│     └─> Uses ai/, rag/, storage/                            │
│                                                               │
│  ai/service.py                                               │
│     └─> Uses core/providers/                                 │
│                                                               │
│  rag/memory_service.py                                       │
│     └─> Uses embedding/, retrieval/, storage/                │
│                                                               │
│  core/providers/                                             │
│     └─> Base AI abstraction (OpenAI, Anthropic)             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Clean separation:**
- `core/` = Base AI provider abstraction (multi-provider support)
- `ai/` = Application-level AI services (summaries, generation)
- `rag/` = RAG-specific logic (memory, retrieval)
- `bot/` = Discord-specific code
- `embedding/`, `chunking/`, `retrieval/` = RAG components

---

## Migration Steps (In Order)

### Step 1: Create New Directory Structure

```bash
# Create new directories (core/ already exists - skip it!)
mkdir -p ai
mkdir -p embedding
mkdir -p chunking/strategies
mkdir -p retrieval/providers
mkdir -p storage
mkdir -p rag
mkdir -p security
mkdir -p bot/cogs bot/loaders bot/utils

# Create __init__.py files
touch ai/__init__.py
touch embedding/__init__.py
touch chunking/__init__.py chunking/strategies/__init__.py
touch retrieval/__init__.py retrieval/providers/__init__.py
touch storage/__init__.py
touch rag/__init__.py
touch security/__init__.py
touch bot/__init__.py bot/cogs/__init__.py bot/loaders/__init__.py bot/utils/__init__.py

# Note: Do NOT touch core/ - it already has __init__.py and is well-structured!
```

### Step 2: Move Existing Services

**Current existing files to move:**
```bash
# AI services
services/ai_service.py → ai/service.py
services/user_ai_tracker.py → ai/tracker.py

# Storage
services/message_storage.py → storage/message_storage.py

# RAG
services/memory_service.py → rag/memory_service.py

# Bot
services/message_loader.py → bot/loaders/message_loader.py
cogs/*.py → bot/cogs/*.py
```

**Future files (created in later phases):**
```bash
# These don't exist yet - will be created in Phase 3+
embedding/base.py         (Phase 3)
embedding/sentence_transformer.py  (Phase 3)
embedding/openai.py       (Phase 3)
embedding/factory.py      (Phase 3)

chunking/base.py          (Phase 4)
chunking/service.py       (Phase 4)

retrieval/base.py         (Phase 5)
retrieval/providers/chroma.py (Phase 5)
retrieval/factory.py      (Phase 5)
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
from services.ai_service import AIService
from services.message_storage import MessageStorage
from services.memory_service import MemoryService
from services.message_loader import MessageLoader
from services.user_ai_tracker import UserAITracker

from cogs.admin import AdminCog
from cogs.summary import SummaryCog

# Future RAG imports (from phases not yet implemented)
from services.embedding_service import EmbeddingServiceFactory
from services.chunking_service import ChunkingService
from services.vector_store_factory import VectorStoreFactory
from services.chunked_memory_service import ChunkedMemoryService
```

### After (New Imports)
```python
# New imports - cleaner and domain-based
from ai import AIService
from ai.tracker import UserAITracker
from storage import MessageStorage
from rag import MemoryService
from bot.loaders.message_loader import MessageLoader

from bot.cogs.admin import AdminCog
from bot.cogs.summary import SummaryCog

# Future RAG imports (from phases not yet implemented)
from embedding import EmbeddingFactory
from chunking import ChunkingService
from retrieval import VectorStoreFactory
from rag import ChunkedMemoryService

# Core AI providers (unchanged - already perfect!)
from core import create_provider, AIConfig, AIRequest, AIResponse
from core.providers import OpenAIProvider, AnthropicProvider
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
