from typing import Optional, Dict
from embedding.base import EmbeddingBase
from embedding.openai import OpenAIEmbedder
from embedding.sentence_transformer import SentenceTransformerEmbedder

class EmbeddingFactory:
    """
    Factory for creating embedding instances with caching.
    
    Caches embedder instances to prevent loading the same model multiple times,
    reducing memory usage and startup time.
    """
    _instances: Dict[str, EmbeddingBase] = {}
    
    @staticmethod
    def create_embedder(provider: str = "sentence-transformers", model_name: str = "all-MiniLM-L6-v2") -> EmbeddingBase:
        """
        Create or retrieve a cached embedder instance.
        
        Args:
            provider: Embedding provider ("sentence-transformers" or "openai")
            model_name: Model name/identifier
            
        Returns:
            Cached or newly created EmbeddingBase instance
        """
        # Create a cache key from provider and model_name
        cache_key = f"{provider}:{model_name}"
        
        # Return cached instance if it exists
        if cache_key in EmbeddingFactory._instances:
            return EmbeddingFactory._instances[cache_key]
        
        # Create new instance
        if provider == "openai":
            embedder = OpenAIEmbedder(model_name=model_name)
        elif provider == "sentence-transformers":
            embedder = SentenceTransformerEmbedder(model_name=model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        # Cache and return
        EmbeddingFactory._instances[cache_key] = embedder
        return embedder
    
    @staticmethod
    def clear_cache():
        """Clear the embedder cache (useful for testing or memory management)."""
        EmbeddingFactory._instances.clear()
    
    @staticmethod
    def get_cache_size() -> int:
        """Get the number of cached embedder instances."""
        return len(EmbeddingFactory._instances)