import logging
from typing import Any, Dict, List, Optional

from embedding.base import EmbeddingBase

# #region agent log
import json as _dbg_json
def _dbg_log(location, message, data=None, hypothesis_id=None):
    try:
        log_entry = {"location": location, "message": message, "data": data or {}, "timestamp": __import__('time').time(), "hypothesisId": hypothesis_id, "sessionId": "debug-session"}
        with open("/Users/youmyeongkim/projects/deep-bot/.cursor/debug.log", "a") as f:
            f.write(_dbg_json.dumps(log_entry) + "\n")
    except: pass
# #endregion


class SentenceTransformerEmbedder(EmbeddingBase):
    # #region agent log
    _instance_count = 0  # Track how many instances are created
    # #endregion
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: Optional[str] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # #region agent log
        import psutil
        SentenceTransformerEmbedder._instance_count += 1
        instance_num = SentenceTransformerEmbedder._instance_count
        process = psutil.Process()
        _dbg_log("sentence_transformer.py:init:start", f"Creating SentenceTransformer instance #{instance_num}", {"rss_mb_before": process.memory_info().rss / 1024 / 1024, "instance_num": instance_num}, "A")
        # #endregion
        
        # Lazy import to avoid errors if sentence-transformers/tokenizers not installed
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            error_msg = str(e).lower()
            if "tokenizers" in error_msg:
                raise ImportError(
                    "The tokenizers python package is not installed. "
                    "Please install it with `pip install tokenizers`. "
                    "sentence-transformers requires tokenizers to function."
                ) from e
            raise ImportError(
                f"sentence-transformers is not installed. "
                f"Install it with: pip install sentence-transformers tokenizers. "
                f"Original error: {e}"
            ) from e
        
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.logger.info(
            "Initializing SentenceTransformerEmbedding with model: %s",
            self.model_name,
        )

        model_init_kwargs: Dict[str, Any] = {}
        if device is not None:
            model_init_kwargs["device"] = device
        self._model = SentenceTransformer(self.model_name, **model_init_kwargs)
        
        # #region agent log
        _dbg_log("sentence_transformer.py:init:complete", f"SentenceTransformer instance #{instance_num} loaded", {"rss_mb_after": process.memory_info().rss / 1024 / 1024, "instance_num": instance_num, "model_name": model_name}, "A")
        # #endregion

        default_encode_kwargs: Dict[str, Any] = {
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        self._encode_kwargs = {**default_encode_kwargs, **(encode_kwargs or {})}
        self._dimension = self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> List[float]:
        if not text:
            raise ValueError("text must be a non-empty string")
        embedding = self._model.encode(text, **self._encode_kwargs)
        return self._to_list(embedding)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, **self._encode_kwargs)
        return self._to_list(embeddings)

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _to_list(data: Any):
        if hasattr(data, "tolist"):
            return data.tolist()
        return list(data)