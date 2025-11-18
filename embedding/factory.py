from typing import Optional
from embedding.base import EmbeddingBase
from embedding.openai import OpenAIEmbedder
from embedding.sentence_transformer import SentenceTransformerEmbedder

class EmbeddingFactory:
    @staticmethod
    def create_embedder(provider: str = "sentence-transformers", model_name: str = "all-MiniLM-L6-v2") -> EmbeddingBase:
        if provider == "openai":
            return OpenAIEmbedder(model_name=model_name)
        elif provider == "sentence-transformers":
            return SentenceTransformerEmbedder(model_name=model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")