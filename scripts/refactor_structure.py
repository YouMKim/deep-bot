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


# Directories to create
DIRECTORIES = [
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
]

# Import replacement map
IMPORT_MAP = {
    "from services.embedding_service import": "from embedding import",
    "from services.chunking_service import": "from chunking import",
    "from services.vector_store_factory import": "from retrieval import",
    "from services.vector_store_base import": "from retrieval.base import",
    "from services.vector_store_chroma import": "from retrieval.providers.chroma import",
    "from services.message_storage import": "from storage import",
    "from services.chunked_memory_service import": "from rag import",
    "from services.memory_service import": "from rag import",
    "from services.ai_service import": "from ai import",
    "from services.message_loader import": "from bot.loaders.message_loader import",
    "from utils.discord_utils": "from bot.utils.discord_utils",
    "from cogs.": "from bot.cogs.",
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

1. Split services/embedding_service.py into:
   - embedding/base.py (EmbeddingProvider)
   - embedding/sentence_transformer.py (SentenceTransformerEmbedder)
   - embedding/openai.py (OpenAIEmbedder)
   - embedding/factory.py (EmbeddingFactory)

2. Split services/chunking_service.py into:
   - chunking/base.py (Chunk class)
   - chunking/service.py (ChunkingService)
   - chunking/strategies/* (individual strategies)

3. Move these files:
   - services/vector_store_base.py → retrieval/base.py
   - services/vector_store_chroma.py → retrieval/providers/chroma.py
   - services/vector_store_factory.py → retrieval/factory.py
   - services/message_storage.py → storage/message_storage.py
   - services/chunked_memory_service.py → rag/memory_service.py
   - services/ai_service.py → ai/service.py
   - services/message_loader.py → bot/loaders/message_loader.py

4. Move all cogs:
   - cogs/* → bot/cogs/*

5. Move Discord utilities:
   - utils/discord_utils.py → bot/utils/discord_utils.py

6. Create __init__.py exports for each domain (see REFACTORING_PLAN.md)

7. Test the bot:
   - python bot.py (check for import errors)
   - Test commands in Discord
   - Run pytest if you have tests

8. Update documentation:
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
