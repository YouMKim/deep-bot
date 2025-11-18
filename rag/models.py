from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class RAGConfig:
    """
    Configuration for RAG pipeline.

    Note on similarity_threshold:
    - Vector/BM25 search: Use 0.3-0.5 (default: 0.35)
    - Hybrid search (RRF): Use 0.005-0.015 (RRF scores are much smaller)
    - Reranking (cross-encoder): Use 0.5-2.0 (cross-encoder scores range from -10 to 10)
    """
    top_k: int = 10
    similarity_threshold: float = 0.35
    max_context_tokens: int = 4000
    temperature: float = 0.7
    strategy: str = "tokens"
    use_hybrid_search: bool = False
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    model: Optional[str] = None
    show_sources: bool = False
    filter_authors: Optional[List[str]] = None
    use_multi_query: bool = False
    num_query_variations: int = 3
    use_hyde: bool = False
    use_reranking: bool = False
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
