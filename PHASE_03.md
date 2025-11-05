# Phase 3: Embedding Service Abstraction

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Embedding Service Abstraction

### Learning Objectives
- Learn abstraction layer design
- Understand Strategy pattern
- Practice dependency injection
- Compare local vs cloud embeddings

### Design Principles
- **Strategy Pattern**: Different implementations, same interface
- **Dependency Inversion**: Depend on abstractions, not concretions
- **Factory Pattern**: Create objects from configuration

### Implementation Steps

#### Step 3.1: Create Abstract Base Class

Create `services/embedding_service.py`:

```python
from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    Learning: Abstract classes define contracts that implementations must follow.
    This enables swapping implementations without changing business logic.
    """
    
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """
        Encode a single text into an embedding vector.
        
        Returns:
            List of floats representing the embedding
        """
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts into embedding vectors.
        
        Learning: Batch operations are often more efficient.
        
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        pass
```

**Key Learning Points:**
- **ABC**: Python's Abstract Base Class enforces interface
- **Abstract Methods**: Must be implemented by subclasses
- **Interface Design**: Methods that all providers must support

#### Step 3.2: Implement SentenceTransformer Provider

```python
from sentence_transformers import SentenceTransformer
import logging

class SentenceTransformerEmbedder(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    
    Learning: Adapter pattern - wraps existing library with our interface.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.logger.info(f"Loading sentence transformer model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
    
    def encode(self, text: str) -> List[float]:
        """Encode single text"""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode batch of texts.
        
        Learning: Batch encoding is more efficient than individual calls.
        """
        embeddings = self._model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=32,  # Process in batches
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension
```

#### Step 3.3: Implement OpenAI Provider

```python
import openai
from typing import List
import logging
import asyncio
from config import Config

class OpenAIEmbedder(EmbeddingProvider):
    """
    Cloud embedding provider using OpenAI API.
    
    Learning: 
    - API clients need error handling and retries
    - Cost tracking is important for cloud services
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Dimension mapping for different models
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        self._dimension = self._dimensions.get(model_name, 1536)
    
    def encode(self, text: str) -> List[float]:
        """Encode single text with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                    raise
                self.logger.warning(f"Embedding failed, retrying ({attempt + 1}/{max_retries}): {e}")
                # Wait before retry (synchronous, not async)
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode batch of texts.
        
        Learning: OpenAI API supports batch encoding natively.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Batch embedding failed after {max_retries} attempts: {e}")
                    raise
                self.logger.warning(f"Batch embedding failed, retrying ({attempt + 1}/{max_retries}): {e}")
                import time
                time.sleep(2 ** attempt)
```

#### Step 3.4: Create Factory

```python
from typing import Optional
from config import Config

class EmbeddingServiceFactory:
    """
    Factory for creating embedding providers.
    
    Learning: Factory pattern centralizes object creation logic.
    Makes it easy to switch providers via configuration.
    """
    
    @staticmethod
    def create(provider_name: Optional[str] = None) -> EmbeddingProvider:
        """
        Create embedding provider based on configuration.
        
        Learning: Configuration-driven creation enables runtime switching.
        """
        provider_name = provider_name or Config.EMBEDDING_PROVIDER
        
        if provider_name == "sentence-transformers":
            return SentenceTransformerEmbedder()
        elif provider_name == "openai":
            return OpenAIEmbedder()
        else:
            raise ValueError(f"Unknown embedding provider: {provider_name}")
```

**Key Learning Points:**
- **Strategy Pattern**: Same interface, different implementations
- **Factory Pattern**: Centralized creation logic
- **Dependency Injection**: Pass provider to services, don't create internally

### Common Pitfalls - Phase 3

1. **Async/await confusion**: OpenAI client is synchronous, not async
2. **Missing API key**: Check config before creating client
3. **Dimension mismatch**: Different models have different dimensions
4. **Batch size too large**: Can hit API limits
5. **No retry logic**: API can be flaky, need retries

### Debugging Tips - Phase 3

- **Test dimensions**: Verify embedding dimensions match
- **Check API keys**: Test with simple encode first
- **Monitor costs**: OpenAI charges per token
- **Compare outputs**: Test both providers with same text

### Performance Considerations - Phase 3

- **Local vs Cloud**: Local is free but slower, cloud is faster but costs
- **Batch size**: 32-64 is good for sentence-transformers
- **Model size**: Smaller models are faster but less accurate