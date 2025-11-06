#!/usr/bin/env python3
"""
Refactoring migration script.

This script migrates the codebase from flat services/ structure
to domain-based architecture.

Usage:
    python scripts/refactor_structure.py --dry-run  # Preview changes
    python scripts/refactor_structure.py            # Execute migration

See REFACTORING_PLAN.md for details.
"""

import os
import shutil
from pathlib import Path
import re


# Directories to create - FULL STRUCTURE (complete domain architecture)
DIRECTORIES = [
    "ai/providers",                # AI domain (core/ + services/ai merged)
    "storage/vectors/providers",   # Storage domain (unified messages + vectors)
    "embedding",                   # Embedding domain (Phase 3) - empty initially
    "chunking/strategies",         # Chunking domain (Phase 4) - empty initially
    "retrieval/advanced",          # Retrieval domain (Phase 5+) - empty initially
    "rag",                         # RAG orchestration domain (Phase 10+)
    "evaluation/datasets",         # Evaluation domain (Phase 6.5 & 17) - empty initially
    "security",                    # Security domain (Phase 3 & 18) - empty initially
    "bot/cogs",                    # Discord bot domain
    "bot/loaders",
    "bot/utils",
    "utils",                       # General utilities
]

# Import replacement map - handles core/ → ai/ merge and all domain moves
IMPORT_MAP = {
    # Core → AI (merge core/ into ai/)
    "from core import": "from ai import",
    "from core.providers import": "from ai.providers import",
    "from core.ai_models import": "from ai.models import",
    "from core.base_provider import": "from ai.base import",
    "from core.providers.openai_provider import": "from ai.providers.openai import",
    "from core.providers.anthropic_provider import": "from ai.providers.anthropic import",

    # Services → Domains
    "from services.ai_service import": "from ai.service import",
    "from services.user_ai_tracker import": "from ai.tracker import",
    "from services.message_storage import": "from storage import",
    "from services.memory_service import": "from rag import",
    "from services.message_loader import": "from bot.loaders.message_loader import",

    # Cogs → Bot
    "from cogs.": "from bot.cogs.",

    # Future RAG imports (for Phase documents)
    "from services.embedding_service import": "from embedding import",
    "from services.chunking_service import": "from chunking import",
    "from services.vector_store_factory import": "from retrieval import",
    "from services.vector_store_base import": "from retrieval.base import",
    "from services.vector_store_chroma import": "from retrieval.providers.chroma import",
    "from services.chunked_memory_service import": "from rag import",

    # Security (Phase 3 & 18)
    "from utils.input_validator": "from security.input_validator",
    "from utils.rate_limiter": "from security.rate_limiter",
    "from utils.prompt_injection": "from security.prompt_injection",
}


def create_directory_structure(dry_run=False):
    """Create new directory structure with __init__.py files."""
    print("📁 Creating directory structure...\n")

    for dir_path in DIRECTORIES:
        dir_full_path = Path(dir_path)

        if dry_run:
            print(f"📋 Would create: {dir_path}/__init__.py")
        else:
            dir_full_path.mkdir(parents=True, exist_ok=True)
            init_file = dir_full_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"✅ Created {dir_path}/__init__.py")
            else:
                print(f"⏭️  Skipped {dir_path}/__init__.py (already exists)")


def update_imports_in_file(file_path, dry_run=False):
    """Update imports in a Python file."""
    if not file_path.exists() or file_path.suffix != ".py":
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # Replace imports
    for old_import, new_import in IMPORT_MAP.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            changes.append((old_import, new_import))

    if content != original_content:
        if dry_run:
            print(f"📋 Would update imports in: {file_path}")
            for old, new in changes:
                print(f"    {old} → {new}")
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated imports in: {file_path}")
        return True

    return False


def update_all_imports(dry_run=False):
    """Update imports in all Python files."""
    print("\n🔧 Updating imports in all Python files...\n")

    updated_count = 0
    for py_file in Path(".").rglob("*.py"):
        # Skip virtual environments, .git, and other non-project files
        if any(part in str(py_file) for part in ["venv", ".git", "__pycache__", "site-packages"]):
            continue

        if update_imports_in_file(py_file, dry_run=dry_run):
            updated_count += 1

    print(f"\n{'📋' if dry_run else '✅'} {updated_count} files {'would be' if dry_run else 'were'} updated")


def show_next_steps():
    """Show manual steps that need to be done."""
    print("\n" + "=" * 70)
    print("📝 MANUAL STEPS REQUIRED")
    print("=" * 70)
    print("""
After running this script, you need to manually:

1. Move core/ → ai/ (consolidate AI domain):
   - mv core/ai_models.py ai/models.py
   - mv core/base_provider.py ai/base.py
   - mv core/providers/openai_provider.py ai/providers/openai.py
   - mv core/providers/anthropic_provider.py ai/providers/anthropic.py
   - rm -rf core/providers && rmdir core

2. Move services/ files to proper domains:
   - mv services/ai_service.py ai/service.py
   - mv services/user_ai_tracker.py ai/tracker.py
   - mv services/message_storage.py storage/message_storage.py
   - mv services/memory_service.py rag/memory_service.py
   - mv services/message_loader.py bot/loaders/message_loader.py

3. Move cogs/ → bot/cogs/:
   - mv cogs/admin.py bot/cogs/admin.py
   - mv cogs/basic.py bot/cogs/basic.py
   - mv cogs/summary.py bot/cogs/summary.py
   - rmdir cogs

4. Create __init__.py exports (see REFACTORING_PLAN.md for templates):
   - ai/__init__.py (merges core + services/ai exports)
   - storage/__init__.py
   - rag/__init__.py
   - bot/__init__.py

5. Empty folders created for future phases:
   - embedding/* (Phase 3)
   - chunking/* (Phase 4)
   - storage/vectors/* (Phase 5)
   - retrieval/* (Phase 5+)
   - rag/* (Phase 10+)
   - evaluation/* (Phase 6.5 & 17)
   - security/* (Phase 3 & 18)

6. Test the bot:
   - python bot.py (check for import errors)
   - Test commands in Discord (!ping, !summary, etc.)
   - Run pytest if you have tests

7. Update documentation:
   - Review IMPLEMENTATION_GUIDE.md
   - Update README.md if needed

See REFACTORING_PLAN.md for detailed instructions!
""")


def main(dry_run=False):
    """Run migration."""
    print("=" * 70)
    print("🚀 REFACTORING MIGRATION SCRIPT")
    print("=" * 70)
    print()

    if dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made\n")
    else:
        print("⚠️  LIVE MODE - Changes will be applied\n")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Migration cancelled")
            return

    # Step 1: Create directory structure
    create_directory_structure(dry_run=dry_run)

    # Step 2: Update imports (automatic)
    update_all_imports(dry_run=dry_run)

    # Step 3: Show manual steps
    show_next_steps()

    print("\n" + "=" * 70)
    if dry_run:
        print("✅ Dry run complete - no changes made")
        print("Run without --dry-run to apply changes")
    else:
        print("✅ Automatic migration steps complete!")
        print("⚠️  Complete manual steps above before testing")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    try:
        main(dry_run=dry_run)
    except KeyboardInterrupt:
        print("\n\n❌ Migration cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error during migration: {e}")
        raise
