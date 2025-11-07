from typing import List, Optional, Dict 
from datetime import datetime
from config import Config 

class Chunk:

    def __init__(
        self,
        content: str,
        message_ids: List[str],
        metadata: Dict
    ):
        self.content = content
        self.message_ids = message_ids
        self.metadata = metadata
    
    def to_dict(self) -> Dict: 
        return {
            "content": self.content,
            "message_ids": self.message_ids,
            "metadata": self.metadata, 
        }
    
    def __repr__(self):
        return (
            f"Chunk(messages={len(self.message_ids)}, "
            f"tokens={self.metadata.get('token_count', 'unknown')}, "
            f"strategy={self.metadata.get('chunk_strategy', 'unknown')})"
        )