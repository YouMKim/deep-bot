from abc import ABC, abstractmethod
from typing import List

class EmbeddingBase(ABC):

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """
        Encode a text string into an embedding vector
        """
        pass
        
    @abstractmethod 
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple text into embedding vectors
        """
        pass 

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the dimension of the embedding vector
        """
        pass
    
    
