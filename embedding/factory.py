from typing import Optional, Dict
from embedding.base import EmbeddingBase
from embedding.openai import OpenAIEmbedder
# Lazy import for SentenceTransformerEmbedder to avoid import errors when not needed

class EmbeddingFactory:
    """
    Factory for creating embedding instances with caching.
    
    Caches embedder instances to prevent loading the same model multiple times,
    reducing memory usage and startup time.
    """
    _instances: Dict[str, EmbeddingBase] = {}
    
    @staticmethod
    def create_embedder(provider: str = "sentence-transformers", model_name: str = "") -> EmbeddingBase:
        """
        Create or retrieve a cached embedder instance.
        
        Args:
            provider: Embedding provider ("sentence-transformers" or "openai")
            model_name: Model name/identifier (auto-selected if empty)
            
        Returns:
            Cached or newly created EmbeddingBase instance
        """
        # Auto-select model name if not provided
        if not model_name:
            if provider == "sentence-transformers":
                model_name = "all-MiniLM-L6-v2"
            elif provider == "openai":
                model_name = "text-embedding-3-small"
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        
        # Create a cache key from provider and model_name
        cache_key = f"{provider}:{model_name}"
        
        # Return cached instance if it exists
        if cache_key in EmbeddingFactory._instances:
            return EmbeddingFactory._instances[cache_key]
        
        # Create new instance
        if provider == "openai":
            embedder = OpenAIEmbedder(model_name=model_name)
        elif provider == "sentence-transformers":
            # Lazy import to avoid errors when sentence-transformers not installed
            try:
                from embedding.sentence_transformer import SentenceTransformerEmbedder
                embedder = SentenceTransformerEmbedder(model_name=model_name)
            except ImportError as e:
                raise ImportError(
                    f"sentence-transformers is not installed. "
                    f"Install it with: pip install sentence-transformers tokenizers. "
                    f"Original error: {e}"
                )
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