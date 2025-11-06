# Refactoring Plan: Domain-Based Architecture 🏗️

**Goal:** Set up complete domain-based architecture with proper separation of concerns. Move existing code into the right places and create structure for future RAG components.

**Estimated Time:** 2-3 hours

**Risk Level:** Low (mostly directory creation + file moves, git history preserved)

---

## 🎯 Strategy: Full Structure Now, Implementation Over Time

**Why set up everything now?**
1. ✅ Clear architectural vision from day one
2. ✅ No future restructuring needed (Phases 3-18 just add files)
3. ✅ Easy to see where everything belongs
4. ✅ Prevents technical debt

**What we're doing:**
1. **Move `core/` → `ai/`** (consolidate AI into one domain)
2. **Create all domain folders** (embedding, chunking, retrieval, rag, security, bot, storage)
3. **Move existing files** into proper domains
4. **Future phases** add new files to already-organized structure

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

## New Structure (Complete Domain Architecture)

```
deep-bot/
├── bot.py                        # Main entry point
├── config.py                     # Global configuration
│
├── ai/                           # 🔀 AI domain (core/ + services/ai merged)
│   ├── __init__.py              # Exports: AIService, AIRequest, AIResponse, etc.
│   ├── models.py                # ← core/ai_models.py (AIRequest, AIResponse, TokenUsage, CostDetails)
│   ├── base.py                  # ← core/base_provider.py (BaseAIProvider)
│   ├── providers/               # ← core/providers/
│   │   ├── __init__.py
│   │   ├── openai.py           # ← core/providers/openai_provider.py
│   │   └── anthropic.py        # ← core/providers/anthropic_provider.py
│   ├── service.py               # ← services/ai_service.py (AIService)
│   └── tracker.py               # ← services/user_ai_tracker.py (UserAITracker)
│
├── embedding/                    # 🆕 Embedding domain (Phase 3)
│   ├── __init__.py
│   ├── base.py                  # EmbeddingProvider abstract class
│   ├── sentence_transformer.py  # Local embeddings
│   ├── openai.py               # OpenAI embeddings
│   └── factory.py              # Factory pattern
│
├── chunking/                     # 🆕 Chunking domain (Phase 4)
│   ├── __init__.py
│   ├── base.py                  # Chunk data structure
│   ├── service.py               # ChunkingService
│   └── strategies/
│       ├── __init__.py
│       ├── temporal.py          # Time-window chunking
│       ├── conversation.py      # Conversation-gap chunking
│       ├── token_aware.py       # Token-limit aware
│       └── sliding_window.py    # Sliding window
│
├── retrieval/                    # 🆕 Vector retrieval domain (Phase 5)
│   ├── __init__.py
│   ├── base.py                  # VectorStore abstract class
│   ├── factory.py               # Factory pattern
│   └── providers/
│       ├── __init__.py
│       ├── chroma.py           # ChromaDB adapter
│       ├── pinecone.py         # Pinecone adapter (future)
│       └── qdrant.py           # Qdrant adapter (future)
│
├── storage/                      # 📦 Data persistence domain
│   ├── __init__.py
│   └── message_storage.py       # ← services/message_storage.py
│
├── rag/                          # 🧠 RAG orchestration domain
│   ├── __init__.py
│   ├── memory_service.py        # ← services/memory_service.py
│   ├── pipeline.py              # RAG pipeline (Phase 10)
│   ├── reranking.py            # Reranking logic (Phase 15)
│   ├── query_optimization.py   # Query expansion (Phase 15)
│   └── strategies.py           # Advanced RAG (Phase 16)
│
├── security/                     # 🔒 Security domain (Phase 3 & 18)
│   ├── __init__.py
│   ├── input_validator.py      # Input validation
│   ├── rate_limiter.py         # Rate limiting
│   ├── prompt_injection.py     # Prompt injection defense
│   └── audit_log.py            # Security audit logging
│
├── bot/                          # 🤖 Discord bot domain
│   ├── __init__.py
│   ├── cogs/                    # ← cogs/
│   │   ├── __init__.py
│   │   ├── basic.py            # ← cogs/basic.py
│   │   ├── admin.py            # ← cogs/admin.py
│   │   ├── summary.py          # ← cogs/summary.py
│   │   └── mvp_chatbot.py      # Phase 2 MVP
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── message_loader.py   # ← services/message_loader.py
│   └── utils/
│       ├── __init__.py
│       └── discord_utils.py    # Discord formatting utilities
│
├── utils/                        # 🛠️ General utilities (non-domain)
│   ├── __init__.py
│   ├── error_handler.py
│   ├── secure_logger.py
│   └── secrets_manager.py
│
└── data/                         # Data directories (unchanged)
    ├── raw_messages/
    └── chroma/
```

**Legend:**
- 🔀 = Merge existing folders
- ← = Move from existing location
- 📦 = Simple move
- 🆕 = Create new (empty, filled in future phases)

**Key Changes:**
1. **Merge `core/` → `ai/`** - Single AI domain (not two!)
2. **Create ALL domain folders** - Even if empty initially
3. **Move existing files** - Into proper domains
4. **Future phases** - Just add files to existing structure

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

### Step 1: Create All Domain Directories

```bash
# Create all domain directories (complete structure)
mkdir -p ai/providers
mkdir -p embedding
mkdir -p chunking/strategies
mkdir -p retrieval/providers
mkdir -p storage
mkdir -p rag
mkdir -p security
mkdir -p bot/cogs bot/loaders bot/utils
mkdir -p utils

# Create __init__.py files for all domains
touch ai/__init__.py ai/providers/__init__.py
touch embedding/__init__.py
touch chunking/__init__.py chunking/strategies/__init__.py
touch retrieval/__init__.py retrieval/providers/__init__.py
touch storage/__init__.py
touch rag/__init__.py
touch security/__init__.py
touch bot/__init__.py bot/cogs/__init__.py bot/loaders/__init__.py bot/utils/__init__.py
touch utils/__init__.py
```

**Why create empty folders?**
- Clear architectural vision from day one
- Phases 3-18 know exactly where to add files
- No future restructuring needed

### Step 2: Move core/ → ai/ (Consolidate AI Domain)

```bash
# Move core/ AI abstraction into ai/
mv core/ai_models.py ai/models.py
mv core/base_provider.py ai/base.py
mv core/providers/openai_provider.py ai/providers/openai.py
mv core/providers/anthropic_provider.py ai/providers/anthropic.py

# Update core/__init__.py → ai/__init__.py
# (manual step - see Step 3)

# Remove empty core/ directory
rm -rf core/providers
rmdir core
```

### Step 3: Move services/ Files to Proper Domains

```bash
# AI services (from services/ → ai/)
mv services/ai_service.py ai/service.py
mv services/user_ai_tracker.py ai/tracker.py

# Storage
mv services/message_storage.py storage/message_storage.py

# RAG
mv services/memory_service.py rag/memory_service.py

# Bot loaders
mv services/message_loader.py bot/loaders/message_loader.py

# Remove empty services/ directory (if empty)
# Check first: ls services/
# If only empty files remain: rmdir services
```

### Step 4: Move cogs/ → bot/cogs/

```bash
# Move all cog files
mv cogs/admin.py bot/cogs/admin.py
mv cogs/basic.py bot/cogs/basic.py
mv cogs/summary.py bot/cogs/summary.py

# Remove old cogs directory
rmdir cogs
```

### Step 5: Future Phase Files

**These folders are empty now, filled in future phases:**

```bash
# Phase 3 (Security & Embedding)
security/input_validator.py
security/rate_limiter.py
security/prompt_injection.py
embedding/base.py
embedding/sentence_transformer.py
embedding/openai.py
embedding/factory.py

# Phase 4 (Chunking)
chunking/base.py
chunking/service.py
chunking/strategies/temporal.py
chunking/strategies/conversation.py
chunking/strategies/token_aware.py
chunking/strategies/sliding_window.py

# Phase 5 (Vector Retrieval)
retrieval/base.py
retrieval/factory.py
retrieval/providers/chroma.py

# Phase 10+ (RAG Pipeline)
rag/pipeline.py
rag/reranking.py (Phase 15)
rag/query_optimization.py (Phase 15)
rag/strategies.py (Phase 16)

# Phase 18 (Advanced Security)
security/audit_log.py
```

**Benefit:** When you implement Phase 3, you already know it goes in `embedding/` and `security/`!

### Step 6: Create __init__.py Exports

**ai/__init__.py** (Consolidates core + services/ai):
```python
"""
AI domain - Language model abstraction and providers.

Exports:
    Core Models:
    - AIRequest, AIResponse, TokenUsage, CostDetails, AIConfig

    Providers:
    - BaseAIProvider: Abstract base class
    - OpenAIProvider: OpenAI implementation
    - AnthropicProvider: Anthropic implementation
    - create_provider: Factory function

    Services:
    - AIService: Application-level AI service
    - UserAITracker: Usage tracking
"""

# Core models (from ai/models.py <- core/ai_models.py)
from ai.models import (
    AIProvider,
    AIConfig,
    AIRequest,
    AIResponse,
    TokenUsage,
    CostDetails
)

# Base provider (from ai/base.py <- core/base_provider.py)
from ai.base import BaseAIProvider

# Provider implementations (from ai/providers/)
from ai.providers.openai import OpenAIProvider
from ai.providers.anthropic import AnthropicProvider

# Factory function
def create_provider(config: AIConfig) -> BaseAIProvider:
    """Create an AI provider based on configuration."""
    if config.model_name == "openai" or "gpt" in config.model_name:
        return OpenAIProvider(config)
    elif config.model_name == "anthropic" or "claude" in config.model_name:
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unknown provider: {config.model_name}")

# Application services (from ai/service.py, ai/tracker.py)
from ai.service import AIService
from ai.tracker import UserAITracker

__all__ = [
    # Core models
    "AIProvider",
    "AIConfig",
    "AIRequest",
    "AIResponse",
    "TokenUsage",
    "CostDetails",
    # Providers
    "BaseAIProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_provider",
    # Services
    "AIService",
    "UserAITracker",
]
```

**storage/__init__.py:**
```python
"""Storage domain - Data persistence."""

from storage.message_storage import MessageStorage

__all__ = ["MessageStorage"]
```

**rag/__init__.py:**
```python
"""RAG domain - Retrieval-Augmented Generation orchestration."""

from rag.memory_service import MemoryService

__all__ = ["MemoryService"]
```

**bot/__init__.py:**
```python
"""Discord bot domain - Commands and integrations."""

# No exports needed - cogs are loaded by bot.py
```

**Future __init__.py files** (created in later phases):

**embedding/__init__.py** (Phase 3):
```python
"""Embedding domain - Text to vector embeddings."""

from embedding.base import EmbeddingProvider
from embedding.factory import EmbeddingFactory

__all__ = ["EmbeddingProvider", "EmbeddingFactory"]
```

**chunking/__init__.py** (Phase 4):
```python
"""Chunking domain - Message chunking strategies."""

from chunking.base import Chunk
from chunking.service import ChunkingService

__all__ = ["Chunk", "ChunkingService"]
```

**retrieval/__init__.py** (Phase 5):
```python
"""Retrieval domain - Vector storage and similarity search."""

from retrieval.base import VectorStore
from retrieval.factory import VectorStoreFactory

__all__ = ["VectorStore", "VectorStoreFactory"]
```

**security/__init__.py** (Phase 3 & 18):
```python
"""Security domain - Input validation and security."""

from security.input_validator import InputValidator
from security.rate_limiter import RateLimiter

__all__ = ["InputValidator", "RateLimiter"]
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
# Old imports - from core/ and services/
from core import create_provider, AIConfig, AIRequest, AIResponse
from core.providers import OpenAIProvider, AnthropicProvider

from services.ai_service import AIService
from services.message_storage import MessageStorage
from services.memory_service import MemoryService
from services.message_loader import MessageLoader
from services.user_ai_tracker import UserAITracker

from cogs.admin import AdminCog
from cogs.summary import SummaryCog

# Future RAG imports (not yet implemented)
from services.embedding_service import EmbeddingServiceFactory
from services.chunking_service import ChunkingService
from services.vector_store_factory import VectorStoreFactory
```

### After (New Imports)
```python
# New imports - clean, domain-based, hierarchical
from ai import (
    # Core models & providers (merged from core/)
    AIConfig,
    AIRequest,
    AIResponse,
    TokenUsage,
    CostDetails,
    BaseAIProvider,
    OpenAIProvider,
    AnthropicProvider,
    create_provider,
    # Application services (from services/)
    AIService,
    UserAITracker,
)

from storage import MessageStorage
from rag import MemoryService
from bot.loaders.message_loader import MessageLoader
from bot.cogs.admin import AdminCog
from bot.cogs.summary import SummaryCog

# Future RAG imports (implemented in Phases 3-18)
from embedding import EmbeddingFactory           # Phase 3
from chunking import ChunkingService              # Phase 4
from retrieval import VectorStoreFactory          # Phase 5
from security import InputValidator, RateLimiter  # Phase 3 & 18
```

**Benefits:**
- ✅ Single `ai` import for all AI-related code (not `core` + `services.ai`)
- ✅ Clear domain boundaries (`storage`, `rag`, `bot`, `security`)
- ✅ Future phases have predefined import paths

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
