from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np
import logging


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    top_k: int = 10,
    k_constant: int = 60,
    weights: Optional[List[float]] = None
) -> List[Dict]:
    """
    Generic RRF implementation for merging multiple ranked lists with optional weights.

    Args:
        ranked_lists: Multiple ranked lists to fuse
        top_k: Number of results to return
        k_constant: RRF constant (default: 60)
        weights: Optional weights for each list (default: equal weights)
                 If provided, must match the length of ranked_lists

    Returns:
        Fused and re-ranked results with rrf_score and fusion_rank fields
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    
    if len(weights) != len(ranked_lists):
        raise ValueError(f"weights length ({len(weights)}) must match ranked_lists length ({len(ranked_lists)})")

    all_docs = {}
    rrf_scores = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked_list, start=1):
            # Use first_message_id as unique document identifier
            doc_id = doc.get('metadata', {}).get('first_message_id', f"doc_{id(doc)}")

            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                rrf_scores[doc_id] = 0

            # RRF formula with weight: weight / (k + rank)
            rrf_scores[doc_id] += weight / (k_constant + rank)

    # Sort by RRF score
    sorted_docs = sorted(
        [(doc_id, score) for doc_id, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Return top-k
    results = []
    for doc_id, score in sorted_docs[:top_k]:
        doc = all_docs[doc_id]
        doc['rrf_score'] = score
        doc['fusion_rank'] = len(results) + 1
        results.append(doc)

    return results


class HybridSearchService:
    """
    Combines BM25 (keyword) and vector (semantic) search using Reciprocal Rank Fusion.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def hybrid_search(
        self,
        query: str,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Merge BM25 and vector search results using Reciprocal Rank Fusion.

        This is now a thin wrapper around the generic reciprocal_rank_fusion function.
        """
        # Use the generic RRF function with weights
        results = reciprocal_rank_fusion(
            ranked_lists=[bm25_results, vector_results],
            top_k=top_k,
            k_constant=60,
            weights=[bm25_weight, vector_weight]
        )
        
        self.logger.info(
            f"Hybrid search: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"→ {len(results)} fused results"
        )
        return results
    