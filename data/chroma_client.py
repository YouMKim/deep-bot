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
            # Initialize ChromaDB client with explicit settings to avoid compatibility issues
            self._client = chromadb.PersistentClient(
                path="data/chroma_db",
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("ChromaDB client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
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
            collections = self._client.list_collections()
            # Convert to list of names to avoid any internal ChromaDB issues
            return [col.name for col in collections] if collections else []
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}", exc_info=True)
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
