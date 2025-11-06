import chromadb
import logging
from pathlib import Path


class ChromaClient:

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._setup_client()

    def _setup_client(self):
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            self._client = chromadb.PersistentClient(path="data/chroma_db")
            self.logger = logging.getLogger(__name__)
            self.logger.info("ChromaDB client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise

    @property
    def client(self):
        return self._client

    def get_collection(self, name: str):
        try:
            return self._client.get_or_create_collection(name)
        except Exception as e:
            self.logger.error(f"Failed to get collection '{name}': {e}")
            raise

    def list_collections(self):
        try:
            return self._client.list_collections()
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []

    def delete_collection(self, name: str):
        try:
            self._client.delete_collection(name)
            self.logger.info(f"Deleted collection '{name}'")
        except Exception as e:
            self.logger.error(f"Failed to delete collection '{name}': {e}")
            raise


# Global instance
chroma_client = ChromaClient()

