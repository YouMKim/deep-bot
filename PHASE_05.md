# Phase 5: Vector Store Abstraction

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Vector Store Abstraction

### Learning Objectives
- Learn adapter pattern in practice
- Understand abstraction layers
- Compare cloud vs local storage
- Design for multi-provider support

### Design Principles
- **Adapter Pattern**: Wrap different APIs with same interface
- **Dependency Inversion**: Depend on abstractions
- **Provider Pattern**: Switch implementations easily

### Implementation Steps

#### Step 5.1: Create Abstract Base Class

Create `storage/vectors/base.py`:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class VectorStore(ABC):
    """
    Abstract interface for vector stores.
    
    Learning: This abstraction allows swapping ChromaDB, Pinecone, etc.
    without changing business logic.
    """
    
    @abstractmethod
    def create_collection(self, name: str, metadata: Dict = None):
        """Create a new collection"""
        pass
    
    @abstractmethod
    def get_collection(self, name: str):
        """Get existing collection"""
        pass
    
    @abstractmethod
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """
        Add documents to collection.
        
        Learning: Batch operations are more efficient.
        """
        pass
    
    @abstractmethod
    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Dict = None
    ) -> Dict:
        """
        Query collection by similarity.
        
        Returns:
            {
                'documents': [[text1, text2, ...]],
                'metadatas': [[meta1, meta2, ...]],
                'distances': [[dist1, dist2, ...]]
            }
        """
        pass
    
    @abstractmethod
    def get_collection_count(self, collection_name: str) -> int:
        """Get number of documents in collection"""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collection names"""
        pass
```

#### Step 5.2: Implement ChromaDB Adapter

Create `storage/vectors/providers/chroma.py`:

```python
from storage.vectors.base import VectorStore
from data.chroma_client import chroma_client
from typing import List, Dict
import logging

class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of vector store.
    
    Learning: Adapter pattern - wraps ChromaDB with our interface.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = chroma_client.client
    
    def create_collection(self, name: str, metadata: Dict = None):
        """Create collection (ChromaDB auto-creates on get)"""
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {}
        )
    
    def get_collection(self, name: str):
        """Get collection"""
        try:
            return self.client.get_collection(name)
        except Exception:
            # Collection doesn't exist, create it
            return self.create_collection(name)
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str]
    ):
        """Add documents in batch"""
        collection = self.get_collection(collection_name)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Dict = None
    ) -> Dict:
        """Query collection"""
        collection = self.get_collection(collection_name)
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )
        return results
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get document count"""
        try:
            collection = self.get_collection(collection_name)
            return collection.count()
        except Exception:
            return 0
    
    def list_collections(self) -> List[str]:
        """List collections"""
        try:
            return [col.name for col in self.client.list_collections()]
        except Exception:
            return []
```

#### Step 5.3: Create Factory

Create `storage/vectors/factory.py`:

```python
from typing import Optional
from config import Config
from storage.vectors.base import VectorStore
from storage.vectors.providers.chroma import ChromaVectorStore

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    @staticmethod
    def create(provider_name: Optional[str] = None) -> VectorStore:
        """
        Create vector store based on configuration.
        
        Learning: Factory pattern enables easy provider switching.
        """
        provider_name = provider_name or Config.VECTOR_STORE_PROVIDER
        
        if provider_name == "chroma":
            return ChromaVectorStore()
        elif provider_name == "pinecone":
            # Future: return PineconeVectorStore()
            raise NotImplementedError("Pinecone not yet implemented")
        else:
            raise ValueError(f"Unknown vector store: {provider_name}")
```

### Common Pitfalls - Phase 5

1. **Collection not found**: Handle missing collections gracefully
2. **Dimension mismatch**: Embeddings must match collection dimension
3. **Metadata types**: ChromaDB requires specific types
4. **Query format**: Results structure differs between providers

### Debugging Tips - Phase 5

- **Check collections**: List all collections to verify creation
- **Test queries**: Query with known documents first
- **Verify dimensions**: Ensure embedding dimensions match
- **Check metadata**: Ensure metadata is JSON-serializable