import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from embedding.base import EmbeddingBase


class SentenceTransformerEmbedder(EmbeddingBase):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: Optional[str] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ):
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