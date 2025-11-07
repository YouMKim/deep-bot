import logging
import time
from typing import Any, Dict, List, Optional

import openai

from config import Config
from embedding.base import EmbeddingBase

class OpenAIEmbedder(EmbeddingBase):

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables")
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY) 
        known_dimensions: Dict[str, int] = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if model_name not in known_dimensions:
            self.logger.warning(
                "Embedding dimension for model '%s' is not predefined; it will be inferred after the first request.",
                model_name,
            )
        self._dimension: Optional[int] = known_dimensions.get(model_name)

    def encode(self, text: str) -> List[float]:
        if not text:
            raise ValueError("text must be a non-empty string")
        response = self._create_embeddings(text)
        embedding = response.data[0].embedding
        self._ensure_dimension_cached(len(embedding))
        return embedding

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._create_embeddings(texts)
        embeddings = [item.embedding for item in response.data]
        if embeddings:
            self._ensure_dimension_cached(len(embeddings[0]))
        return embeddings
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise ValueError(
                "Embedding dimension is not known yet; perform at least one encode call first."
            )
        return self._dimension

    def _create_embeddings(self, input_payload: Any):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=input_payload,
                    model=self.model_name,
                )
                return response
            except Exception as e:
                self.logger.error(f"Error encoding text: {e}")
                if attempt < max_retries - 1 and self._should_retry(e):
                    sleep_seconds = 2 ** attempt
                    self.logger.warning(
                        "Retrying embedding request (attempt %s/%s) after %.1f seconds...",
                        attempt + 1,
                        max_retries,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise

    def _should_retry(self, error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code is None and hasattr(error, "response"):
            status_code = getattr(getattr(error, "response"), "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True
        return False

    def _ensure_dimension_cached(self, dimension: int) -> None:
        if self._dimension is None:
            self._dimension = dimension