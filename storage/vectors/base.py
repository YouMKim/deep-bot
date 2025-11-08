from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class VectorStorage(ABC):

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        pass 

    @abstractmethod
    def get_collection(self, collection_name: str) -> None:
        pass 
    
    @abstractmethod
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
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
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        pass
    
    @abstractmethod
    def delete_documents(
        self,
        collection_name: str,
        ids: List[str],
    ):
        pass
    