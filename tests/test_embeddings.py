from types import SimpleNamespace

import pytest

from config import Config
from embedding import openai as openai_module
from embedding import sentence_transformer as st_module
from embedding.openai import OpenAIEmbedder
from embedding.sentence_transformer import SentenceTransformerEmbedder


class _DummySentenceTransformer:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.init_kwargs = kwargs
        self.encode_calls = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, data, **kwargs):
        self.encode_calls.append((data, kwargs))
        if isinstance(data, str):
            return _DummyEmbedding([[1.0, 0.0, 0.0]][0])
        return _DummyEmbedding([[1.0, 0.0, 0.0] for _ in data])


class _DummyEmbedding:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload


class _DummyOpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = _DummyEmbeddings()


class _DummyEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, input, model):
        self.calls.append((input, model))
        if isinstance(input, list):
            data = [SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in input]
        else:
            data = [SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
        return SimpleNamespace(data=data)


def test_sentence_transformer_embedder_respects_encode_kwargs(monkeypatch):
    monkeypatch.setattr(st_module, "SentenceTransformer", _DummySentenceTransformer)
    embedder = SentenceTransformerEmbedder(
        model_name="dummy-model",
        device="cpu",
        encode_kwargs={"batch_size": 16, "show_progress_bar": True},
    )

    assert embedder.dimension == 3
    assert embedder.encode("hello") == [1.0, 0.0, 0.0]
    assert embedder.encode_batch(["a", "b"]) == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert embedder._model.encode_calls[-1][1]["batch_size"] == 16
    assert embedder.encode_batch([]) == []

    with pytest.raises(ValueError):
        embedder.encode("")


def test_openai_embedder_infers_dimension(monkeypatch):
    monkeypatch.setattr(Config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_module, "openai", SimpleNamespace(OpenAI=_DummyOpenAIClient))

    embedder = OpenAIEmbedder(model_name="custom-model")

    with pytest.raises(ValueError):
        _ = embedder.dimension

    assert embedder.encode("hello") == [0.1, 0.2, 0.3]
    assert embedder.dimension == 3
    assert embedder.encode_batch(["a", "b"]) == [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ]
    assert embedder.dimension == 3
    assert embedder.encode_batch([]) == []

