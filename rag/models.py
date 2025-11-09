from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass 
class RAGConfig:
    top_k: int = 10 
    similarity_threshold: float = 0.35  
    max_context_tokens: int = 4000
    temperature: float = 0.7
    strategy: str = "tokens"  # tokens gives better results than single for semantic search
    model: Optional[str] = None
    show_sources: bool = False
    filter_authors: Optional[List[str]] = None 
    
    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.max_context_tokens < 100:
            raise ValueError("max_context_tokens must be at least 100")

@dataclass
class RAGResult:
    answer: str
    sources: List[Dict] = field(default_factory=list)
    config_used: Optional[RAGConfig] = None
    tokens_used: int = 0
    cost: float = 0.0
    model: str = "unknown"
    
    def format_for_discord(self, include_sources: bool = False) -> str:
        output = f"**Answer:**\n{self.answer}\n\n"
        output += f"*Model: {self.model} | Tokens: {self.tokens_used} | Cost: ${self.cost:.4f}*"
        
        if include_sources and self.sources:
            output += "\n\n**Sources:**\n"
            for i, source in enumerate(self.sources, 1):
                similarity = source.get('similarity', 0)
                metadata = source.get('metadata', {})
                author = metadata.get('author', 'Unknown')
                timestamp = metadata.get('timestamp', 'Unknown')
                output += f"{i}. [{timestamp}] {author} (similarity: {similarity:.2f})\n"
        
        return output
