# Phase 6: Multi-Strategy Chunk Storage

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Multi-Strategy Chunk Storage

### Learning Objectives
- Understand RAG vector store architecture
- Learn collection/namespace patterns
- Design for experimentation
- Compare strategies

### Implementation Steps

#### Step 6.1: Create ChunkedMemoryService

Create `storage/chunked_memory.py`:

```python
from storage.vectors.base import VectorStore
from embedding.base import EmbeddingProvider
from typing import List, Dict, Optional
import logging

class ChunkedMemoryService:
    """
    Manages chunk storage with multiple strategies.
    
    Learning: RAG architecture - store once, query multiple ways.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.logger = logging.getLogger(__name__)
        self.active_strategy = "temporal"  # Default
    
    def store_all_strategies(
        self,
        chunks_dict: Dict[str, List]  # {"temporal": [chunks], ...}
    ):
        """
        Store chunks for all strategies in separate collections.
        
        Learning: Separate collections enable strategy comparison.
        """
        for strategy, chunks in chunks_dict.items():
            if not chunks:
                self.logger.warning(f"No chunks for strategy {strategy}, skipping")
                continue
                
            collection_name = f"discord_chunks_{strategy}"
            
            # Create collection if needed
            self.vector_store.create_collection(collection_name)
            
            # Prepare batch data
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [f"{strategy}_{i}_{chunk.metadata.get('first_message_id', i)}" 
                   for i, chunk in enumerate(chunks)]
            
            # Generate embeddings in batch
            self.logger.info(f"Generating embeddings for {len(chunks)} {strategy} chunks...")
            try:
                embeddings = self.embedding_provider.encode_batch(documents)
                
                # Verify dimensions match
                if embeddings and len(embeddings[0]) != self.embedding_provider.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: "
                        f"expected {self.embedding_provider.dimension}, "
                        f"got {len(embeddings[0])}"
                    )
                
                # Store in vector DB
                self.vector_store.add_documents(
                    collection_name=collection_name,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                self.logger.info(f"Stored {len(chunks)} chunks for {strategy} strategy")
            except Exception as e:
                self.logger.error(f"Error storing {strategy} chunks: {e}")
                raise
    
    def search(
        self,
        query: str,
        strategy: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search using specified strategy.
        
        Learning: Strategy selection at query time enables comparison.
        """
        strategy = strategy or self.active_strategy
        collection_name = f"discord_chunks_{strategy}"
        
        # Generate query embedding
        query_embedding = self.embedding_provider.encode(query)
        
        # Query vector store
        results = self.vector_store.query(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results.get('documents') and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'similarity': 1 - results['distances'][0][i] if results.get('distances') else 0.0
                })
        
        return formatted_results
    
    def get_strategy_stats(self) -> Dict[str, int]:
        """Get document counts per strategy"""
        stats = {}
        for strategy in ["temporal", "conversation", "single"]:
            collection_name = f"discord_chunks_{strategy}"
            try:
                stats[strategy] = self.vector_store.get_collection_count(collection_name)
            except Exception:
                stats[strategy] = 0
        return stats
    
    def switch_active_strategy(self, strategy: str):
        """Switch the active strategy for queries"""
        if strategy not in ["temporal", "conversation", "single"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.active_strategy = strategy
        self.logger.info(f"Switched active strategy to {strategy}")
```

### Common Pitfalls - Phase 6

1. **Empty chunks**: Don't try to store empty chunk lists
2. **Dimension mismatch**: Verify embedding dimensions
3. **ID collisions**: Use unique IDs across strategies
4. **Missing collections**: Handle case where collection doesn't exist