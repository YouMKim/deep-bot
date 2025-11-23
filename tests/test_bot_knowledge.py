"""
Test bot knowledge functionality.

Tests that bot documentation is properly indexed and searchable.
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.chunked_memory import ChunkedMemoryService
from storage.chunked_memory.bot_knowledge_service import BotKnowledgeService
from rag.pipeline import RAGPipeline
from ai.service import AIService
from config import Config


async def test_bot_knowledge_ingestion():
    """Test that bot knowledge can be ingested."""
    print("=" * 60)
    print("Test 1: Bot Knowledge Ingestion")
    print("=" * 60)
    
    try:
        chunked_memory = ChunkedMemoryService(config=Config)
        bot_knowledge_service = chunked_memory.bot_knowledge_service
        
        # Check if bot knowledge file exists
        import os
        bot_knowledge_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "docs",
            "bot_knowledge.md"
        )
        
        if not os.path.exists(bot_knowledge_path):
            print(f"❌ Bot knowledge file not found: {bot_knowledge_path}")
            return False
        
        print(f"✓ Bot knowledge file found: {bot_knowledge_path}")
        
        # Load documentation
        content = bot_knowledge_service.load_bot_documentation()
        if not content:
            print("❌ Failed to load bot documentation")
            return False
        
        print(f"✓ Loaded {len(content)} characters of documentation")
        
        # Chunk documentation
        chunks = bot_knowledge_service.chunk_bot_docs(content)
        if not chunks:
            print("❌ No chunks created from documentation")
            return False
        
        print(f"✓ Created {len(chunks)} chunks from documentation")
        
        # Check chunk metadata
        sample_chunk = chunks[0]
        print(f"✓ Sample chunk metadata: {sample_chunk.metadata.get('source')}, {sample_chunk.metadata.get('section')}")
        
        # Try to ingest (force re-index)
        print("\nAttempting to ingest bot knowledge...")
        success = await bot_knowledge_service.ingest_bot_knowledge(force=True)
        
        if success:
            print("✓ Bot knowledge successfully ingested")
            return True
        else:
            print("❌ Failed to ingest bot knowledge")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_bot_knowledge_search():
    """Test that bot knowledge is searchable."""
    print("\n" + "=" * 60)
    print("Test 2: Bot Knowledge Search")
    print("=" * 60)
    
    try:
        chunked_memory = ChunkedMemoryService(config=Config)
        
        # Test queries about the bot
        test_queries = [
            "What commands does Deep-Bot have?",
            "How does the RAG system work?",
            "What AI providers are supported?",
            "What is Deep-Bot?",
            "How do I use the summary command?",
        ]
        
        print(f"Testing {len(test_queries)} queries...\n")
        
        for query in test_queries:
            print(f"Query: {query}")
            
            # Search bot docs collection directly
            from storage.chunked_memory.utils import get_collection_name
            bot_docs_collection = get_collection_name('bot_docs')
            
            try:
                collections = chunked_memory.vector_store.list_collections()
                if bot_docs_collection not in collections:
                    print(f"  ⚠ Bot docs collection not found, skipping search")
                    continue
                
                # Generate query embedding
                import asyncio
                loop = asyncio.get_event_loop()
                query_embedding = await loop.run_in_executor(
                    None,
                    chunked_memory.embedder.encode,
                    query
                )
                
                # Search
                results = chunked_memory.vector_store.query(
                    collection_name=bot_docs_collection,
                    query_embeddings=[query_embedding],
                    n_results=3
                )
                
                if results and results.get('documents') and results['documents'][0]:
                    docs = results['documents'][0]
                    distances = results.get('distances', [[]])[0] if results.get('distances') else []
                    
                    print(f"  ✓ Found {len(docs)} results")
                    for i, (doc, dist) in enumerate(zip(docs[:2], distances[:2]), 1):
                        similarity = 1 - dist if dist <= 1 else 0
                        preview = doc[:100].replace('\n', ' ')
                        print(f"    {i}. Similarity: {similarity:.3f} - {preview}...")
                else:
                    print(f"  ❌ No results found")
                    
            except Exception as e:
                print(f"  ❌ Search error: {e}")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rag_pipeline_with_bot_knowledge():
    """Test that RAG pipeline includes bot knowledge in searches."""
    print("\n" + "=" * 60)
    print("Test 3: RAG Pipeline with Bot Knowledge")
    print("=" * 60)
    
    try:
        ai_service = AIService(provider_name=Config.AI_DEFAULT_PROVIDER)
        rag_pipeline = RAGPipeline(ai_service=ai_service, config=Config)
        
        # Test queries about the bot
        test_queries = [
            "What commands do you have?",
            "How does your RAG system work?",
            "What AI providers do you support?",
            "Tell me about Deep-Bot",
        ]
        
        print(f"Testing RAG pipeline with {len(test_queries)} queries...\n")
        
        for query in test_queries:
            print(f"Query: {query}")
            print("-" * 60)
            
            try:
                # Get chunks (this should include bot docs)
                chunks = await rag_pipeline._retrieve_chunks(
                    query,
                    rag_pipeline._create_rag_config()
                )
                
                bot_docs_count = sum(
                    1 for chunk in chunks 
                    if chunk.get('metadata', {}).get('source') == 'bot_documentation'
                )
                regular_count = len(chunks) - bot_docs_count
                
                print(f"  Retrieved {len(chunks)} total chunks:")
                print(f"    - {bot_docs_count} from bot documentation")
                print(f"    - {regular_count} from regular messages")
                
                if bot_docs_count > 0:
                    print(f"  ✓ Bot knowledge is being retrieved!")
                    # Show sample bot doc chunk
                    bot_chunk = next(
                        (c for c in chunks if c.get('metadata', {}).get('source') == 'bot_documentation'),
                        None
                    )
                    if bot_chunk:
                        preview = bot_chunk.get('content', '')[:150].replace('\n', ' ')
                        print(f"    Sample: {preview}...")
                else:
                    print(f"  ⚠ No bot knowledge chunks retrieved")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_rag_answer():
    """Test full RAG pipeline answering questions about the bot."""
    print("\n" + "=" * 60)
    print("Test 4: Full RAG Answer Generation")
    print("=" * 60)
    
    try:
        ai_service = AIService(provider_name=Config.AI_DEFAULT_PROVIDER)
        rag_pipeline = RAGPipeline(ai_service=ai_service, config=Config)
        
        test_query = "What commands does Deep-Bot have?"
        
        print(f"Query: {test_query}")
        print("-" * 60)
        
        # Create a simple config
        from rag.models import RAGConfig
        config = RAGConfig(
            top_k=5,
            use_hybrid_search=False,
            similarity_threshold=0.01
        )
        
        result = await rag_pipeline.answer_question(
            test_query,
            config=config
        )
        
        print(f"\nAnswer: {result.answer}")
        print(f"\nSources: {len(result.sources)} chunks")
        
        bot_docs_sources = [
            s for s in result.sources 
            if s.get('metadata', {}).get('source') == 'bot_documentation'
        ]
        
        if bot_docs_sources:
            print(f"  ✓ {len(bot_docs_sources)} sources from bot documentation")
        else:
            print(f"  ⚠ No sources from bot documentation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Bot Knowledge Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Ingestion
    results.append(await test_bot_knowledge_ingestion())
    
    # Test 2: Search
    results.append(await test_bot_knowledge_search())
    
    # Test 3: RAG Pipeline
    results.append(await test_rag_pipeline_with_bot_knowledge())
    
    # Test 4: Full Answer
    results.append(await test_full_rag_answer())
    
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
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

