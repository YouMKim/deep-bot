from typing import List, Dict
from sentence_transformers import CrossEncoder
import logging

# #region agent log
import json as _dbg_json
def _dbg_log(location, message, data=None, hypothesis_id=None):
    try:
        log_entry = {"location": location, "message": message, "data": data or {}, "timestamp": __import__('time').time(), "hypothesisId": hypothesis_id, "sessionId": "debug-session"}
        with open("/Users/youmyeongkim/projects/deep-bot/.cursor/debug.log", "a") as f:
            f.write(_dbg_json.dumps(log_entry) + "\n")
    except: pass
# #endregion


class ReRankingService:
    """
    Re-ranks retrieval results using a cross-encoder model.

    Cross-encoders are more accurate than bi-encoders (used for initial retrieval)
    but slower, so we use them only for re-ranking top candidates.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        # #region agent log
        import psutil
        process = psutil.Process()
        _dbg_log("reranking.py:init:start", "Creating CrossEncoder (reranker) model", {"rss_mb_before": process.memory_info().rss / 1024 / 1024, "model_name": model_name}, "D")
        # #endregion
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        
        # #region agent log
        _dbg_log("reranking.py:init:complete", "CrossEncoder model loaded", {"rss_mb_after": process.memory_info().rss / 1024 / 1024}, "D")
        # #endregion

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Re-rank chunks using cross-encoder.

        Args:
            query: Search query
            chunks: Initial retrieval results
            top_k: Number of results to return

        Returns:
            Re-ranked chunks with cross-encoder scores
        """
        if not chunks:
            return []

        self.logger.info(f"Re-ranking {len(chunks)} chunks with cross-encoder")

        # Create query-document pairs
        pairs = [(query, chunk['content']) for chunk in chunks]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Add scores to chunks and sort
        for chunk, score in zip(chunks, scores):
            chunk['ce_score'] = float(score)
            chunk['original_rank'] = chunks.index(chunk) + 1

        # Sort by cross-encoder score
        reranked = sorted(chunks, key=lambda x: x['ce_score'], reverse=True)

        # Take top-k
        reranked = reranked[:top_k]

        # Log improvements
        for i, chunk in enumerate(reranked[:5], 1):
            original_rank = chunk.get('original_rank', '?')
            ce_score = chunk.get('ce_score', 0)
            original_score = chunk.get('similarity', chunk.get('rrf_score', 0))

            self.logger.info(
                f"  Rank {i} (was {original_rank}): "
                f"CE={ce_score:.3f}, Original={original_score:.3f}"
            )

        return reranked

