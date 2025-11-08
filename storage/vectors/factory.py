from typing import Optional
from config import Config
from storage.vectors.base import VectorStorage
from storage.vectors.providers.chroma import ChromaVectorStorage

class VectorStoreFactory:
    
    @staticmethod
    def create(provider_name: Optional[str] = None) -> VectorStorage:
        provider_name = provider_name or Config.VECTOR_STORE_PROVIDER
        
        if provider_name == "chroma":
            return ChromaVectorStorage()
        elif provider_name == "pinecone":
            # TODO: return PineconeVectorStore()
            raise NotImplementedError("Pinecone not yet implemented")
        else:
            raise ValueError(f"Unknown vector store: {provider_name}")