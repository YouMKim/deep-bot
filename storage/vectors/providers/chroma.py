from storage.vectors.base import VectorStorage
from data.chroma_client import chroma_client
from typing import List, Dict, Optional
import logging


class ChromaVectorStorage(VectorStorage):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = chroma_client.client

    def create_collection(self, collection_name: str) -> None:
        try:
            # Use get_or_create_collection with explicit metadata to avoid issues
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={"created_by": "deep-bot"}
            )
            self.logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to create collection '{collection_name}': {e}", exc_info=True)
            raise
    
    def get_collection(self, collection_name: str):
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            # If collection doesn't exist, create it
            self.logger.debug(f"Collection '{collection_name}' not found, creating it: {e}")
            return self.client.get_or_create_collection(collection_name)
    
    def get_all_documents(self, collection_name: str) -> List[Dict]:
        try:
            collection = self.get_collection(collection_name)
            results =  collection.get()
            documents = []
            for i, doc in enumerate(results['documents']):
                documents.append({
                    'document': doc,
                    'metadata': results['metadatas'][i] if i < len(results['metadatas']) else {},
                    'id': results['ids'][i] if i < len(results['ids']) else str(i)
                })

            return documents

        except Exception as e:
            self.logger.error(f"Failed to get all documents from {collection_name}: {e}")
            return []
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str]
    ):
        try:
            collection = self.get_collection(collection_name)
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            self.logger.error(
                f"Failed to add documents to collection '{collection_name}': {e}"
            )
            raise
    
    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Query collection by similarity.
        
        Args:
            collection_name: Name of the collection to query
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            where: Optional metadata filter dictionary
            
        Returns:
            Dictionary with 'documents', 'metadatas', 'distances', and 'ids'
        """
        try:
            collection = self.get_collection(collection_name)
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            self.logger.error(
                f"Failed to query collection '{collection_name}': {e}"
            )
            raise
    
    def get_collection_count(self, collection_name: str) -> int:
        try:
            collection = self.get_collection(collection_name)
            return collection.count()
        except Exception as e:
            self.logger.warning(
                f"Failed to get count for collection '{collection_name}': {e}"
            )
            return 0
    
    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            # Handle ChromaDB compatibility issues
            result = []
            for col in collections:
                try:
                    result.append(col.name)
                except (KeyError, AttributeError) as e:
                    # Skip collections with metadata issues
                    self.logger.warning(f"Skipping collection due to metadata issue: {e}")
                    continue
            return result
        except KeyError as e:
            if "frozenset" in str(e).lower():
                self.logger.warning(
                    "ChromaDB KeyError: frozenset() detected. "
                    "This may indicate corrupted metadata. Returning empty list."
                )
                return []
            raise
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}", exc_info=True)
            return []
    
    def delete_collection(self, collection_name: str) -> None:
        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            self.logger.error(
                f"Failed to delete collection '{collection_name}': {e}"
            )
            raise
    
    def delete_documents(
        self,
        collection_name: str,
        ids: List[str],
    ) -> None:

        if not ids:
            self.logger.warning("No IDs provided for deletion")
            return
        try:
            collection = self.get_collection(collection_name)
            collection.delete(ids=ids)
            self.logger.info(
                f"Deleted {len(ids)} document(s) from collection '{collection_name}'"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to delete documents from collection '{collection_name}': {e}"
            )
            raise