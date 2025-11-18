from typing import Optional, TYPE_CHECKING
from storage.vectors.base import VectorStorage
from storage.vectors.providers.chroma import ChromaVectorStorage

if TYPE_CHECKING:
    from config import Config

class VectorStoreFactory:
    
    @staticmethod
    def create(provider_name: Optional[str] = None, config: Optional['Config'] = None) -> VectorStorage:
        from config import Config as ConfigClass
        
        config_instance = config or ConfigClass
        provider_name = provider_name or config_instance.VECTOR_STORE_PROVIDER
        
        if provider_name == "chroma":
            return ChromaVectorStorage()
        elif provider_name == "pinecone":
            # TODO: return PineconeVectorStore()
            raise NotImplementedError("Pinecone not yet implemented")
        else:
            raise ValueError(f"Unknown vector store: {provider_name}")