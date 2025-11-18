from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np
import logging


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    top_k: int = 10,
    k_constant: int = 60
) -> List[Dict]:
    """
    Generic RRF implementation for merging multiple ranked lists.
    
    Standalone function version for use in pipeline.
    """
    all_docs = {}
    rrf_scores = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            # Use first_message_id as unique document identifier
            doc_id = doc.get('metadata', {}).get('first_message_id', f"doc_{id(doc)}")

            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                rrf_scores[doc_id] = 0

            # RRF formula: 1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k_constant + rank)

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
        """
        # RRF constant (standard value)
        k = 60

        all_docs = {}
        rrf_scores = {}

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.get('metadata', {}).get('first_message_id', f"bm25_{rank}")
            all_docs[doc_id] = result
            rrf_scores[doc_id] = bm25_weight / (k + rank)

        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.get('metadata', {}).get('first_message_id', f"vector_{rank}")
            if doc_id not in all_docs:
                all_docs[doc_id] = result
                rrf_scores[doc_id] = 0

            rrf_scores[doc_id] += vector_weight / (k + rank)
            
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = all_docs[doc_id]
            doc['rrf_score'] = rrf_score
            doc['fusion_rank'] = len(results) + 1
            results.append(doc)
        
        self.logger.info(
            f"Hybrid search: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"→ {len(results)} fused results"
        )
        return results
    
    def reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Dict]],
        top_k: int = 10,
        k_constant: int = 60
    ) -> List[Dict]:
        """
        Generic RRF implementation for merging multiple ranked lists.
        """
        all_docs = {}
        rrf_scores = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                # Use first_message_id as unique document identifier
                doc_id = doc.get('metadata', {}).get('first_message_id', f"doc_{id(doc)}")

                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
                    rrf_scores[doc_id] = 0

                # RRF formula: 1 / (k + rank)
                rrf_scores[doc_id] += 1.0 / (k_constant + rank)

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
            results.append(doc)

        return results
