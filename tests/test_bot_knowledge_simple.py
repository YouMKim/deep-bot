"""
Simple test for bot knowledge functionality.

Tests bot knowledge ingestion and search without requiring Discord dependencies.
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock discord import before importing config
import sys
from unittest.mock import MagicMock
sys.modules['discord'] = MagicMock()
sys.modules['discord'].Intents = MagicMock()

from storage.chunked_memory.bot_knowledge_service import BotKnowledgeService
from storage.chunked_memory.utils import get_collection_name


def test_bot_knowledge_file_exists():
    """Test that bot knowledge file exists."""
    print("=" * 60)
    print("Test 1: Bot Knowledge File Exists")
    print("=" * 60)
    
    bot_knowledge_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "docs",
        "bot_knowledge.md"
    )
    
    if os.path.exists(bot_knowledge_path):
        print(f"✓ Bot knowledge file found: {bot_knowledge_path}")
        
        # Check file size
        size = os.path.getsize(bot_knowledge_path)
        print(f"✓ File size: {size} bytes")
        
        # Read and check content
        with open(bot_knowledge_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✓ Content length: {len(content)} characters")
        
        # Check for key sections
        key_sections = [
            "Deep-Bot",
            "Available Commands",
            "RAG",
            "AI Providers",
            "Technical Stack"
        ]
        
        found_sections = []
        for section in key_sections:
            if section.lower() in content.lower():
                found_sections.append(section)
        
        print(f"✓ Found {len(found_sections)}/{len(key_sections)} key sections: {', '.join(found_sections)}")
        
        return True
    else:
        print(f"❌ Bot knowledge file not found: {bot_knowledge_path}")
        return False


def test_bot_knowledge_chunking():
    """Test bot knowledge chunking logic."""
    print("\n" + "=" * 60)
    print("Test 2: Bot Knowledge Chunking")
    print("=" * 60)
    
    try:
        from storage.vectors.factory import VectorStoreFactory
        from embedding.factory import EmbeddingFactory
        from config import Config
        
        vector_store = VectorStoreFactory.create()
        embedder = EmbeddingFactory.create_embedder(
            provider=Config.EMBEDDING_PROVIDER,
            model_name=Config.EMBEDDING_MODEL if Config.EMBEDDING_MODEL else ""
        )
        
        from storage.chunked_memory.embedding_service import EmbeddingService
        embedding_service = EmbeddingService(embedder=embedder, config=Config)
        
        bot_knowledge_service = BotKnowledgeService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            config=Config
        )
        
        # Load documentation
        content = bot_knowledge_service.load_bot_documentation()
        if not content:
            print("❌ Failed to load bot documentation")
            return False
        
        print(f"✓ Loaded {len(content)} characters")
        
        # Chunk documentation
        chunks = bot_knowledge_service.chunk_bot_docs(content)
        if not chunks:
            print("❌ No chunks created")
            return False
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Check chunk metadata
        sample_chunk = chunks[0]
        print(f"✓ Sample chunk:")
        print(f"    - Content length: {len(sample_chunk.content)} chars")
        print(f"    - Source: {sample_chunk.metadata.get('source')}")
        print(f"    - Section: {sample_chunk.metadata.get('section')}")
        print(f"    - Channel: {sample_chunk.metadata.get('channel_id')}")
        print(f"    - Author: {sample_chunk.metadata.get('author')}")
        
        # Check that all chunks have required metadata
        required_fields = ['source', 'chunk_strategy', 'channel_id', 'author']
        all_valid = True
        for i, chunk in enumerate(chunks):
            for field in required_fields:
                if field not in chunk.metadata:
                    print(f"❌ Chunk {i} missing required field: {field}")
                    all_valid = False
        
        if all_valid:
            print(f"✓ All chunks have required metadata")
        
        return all_valid
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collection_name():
    """Test collection name generation."""
    print("\n" + "=" * 60)
    print("Test 3: Collection Name")
    print("=" * 60)
    
    collection_name = get_collection_name('bot_docs')
    expected = 'discord_chunks_bot_docs'
    
    print(f"Strategy: bot_docs")
    print(f"Collection name: {collection_name}")
    print(f"Expected: {expected}")
    
    if collection_name == expected:
        print("✓ Collection name is correct")
        return True
    else:
        print("❌ Collection name mismatch")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Bot Knowledge Simple Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: File exists
    results.append(test_bot_knowledge_file_exists())
    
    # Test 2: Chunking
    results.append(test_bot_knowledge_chunking())
    
    # Test 3: Collection name
    results.append(test_collection_name())
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

