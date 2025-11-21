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
        import os
        import shutil
        
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        chroma_path = Path("data/chroma_db")
        
        # Check if we should reset ChromaDB (for compatibility issues)
        reset_chromadb = os.getenv("RESET_CHROMADB", "False").lower() == "true"
        
        try:
            # Initialize ChromaDB client with explicit settings to avoid compatibility issues
            self._client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
            self.logger = logging.getLogger(__name__)
            
            # Test if we can list collections (catches frozenset errors early)
            try:
                _ = self._client.list_collections()
                self.logger.info("ChromaDB client initialized successfully")
            except (KeyError, AttributeError) as e:
                if "frozenset" in str(e).lower():
                    self.logger.warning(
                        "ChromaDB metadata compatibility issue detected. "
                        "Clearing ChromaDB database..."
                    )
                    # Close client before deleting
                    del self._client
                    # Remove ChromaDB directory
                    if chroma_path.exists():
                        shutil.rmtree(chroma_path)
                        self.logger.info("ChromaDB database cleared")
                    # Recreate client
                    self._client = chromadb.PersistentClient(
                        path=str(chroma_path),
                        settings=chromadb.Settings(anonymized_telemetry=False)
                    )
                    self.logger.info("ChromaDB client reinitialized with clean database")
                else:
                    raise
                    
        except Exception as e:
            # If reset is enabled and we hit an error, try clearing and retrying
            if reset_chromadb and chroma_path.exists():
                self.logger.warning(f"ChromaDB error detected, resetting database: {e}")
                try:
                    shutil.rmtree(chroma_path)
                    self._client = chromadb.PersistentClient(
                        path=str(chroma_path),
                        settings=chromadb.Settings(anonymized_telemetry=False)
                    )
                    self.logger.info("ChromaDB reset successfully")
                except Exception as reset_error:
                    self.logger.error(f"Failed to reset ChromaDB: {reset_error}", exc_info=True)
                    raise
            else:
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
            result = []
            for col in collections:
                try:
                    result.append(col.name)
                except (KeyError, AttributeError) as e:
                    if "frozenset" in str(e).lower():
                        self.logger.warning(f"Skipping collection due to metadata issue: {e}")
                        continue
                    raise
            return result
        except KeyError as e:
            if "frozenset" in str(e).lower():
                self.logger.warning("ChromaDB frozenset error in list_collections, returning empty list")
                return []
            raise
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
