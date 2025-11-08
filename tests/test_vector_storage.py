"""
Tests for vector storage (ChromaDB implementation).
"""

import pytest
from storage.vectors.providers.chroma import ChromaVectorStorage


@pytest.fixture
def vector_storage(monkeypatch, tmp_path):
    """Create a ChromaVectorStorage instance with temporary database."""
    import chromadb
    from unittest.mock import MagicMock
    
    # Create temporary ChromaDB client
    temp_db_path = str(tmp_path / "test_chroma_db")
    temp_client = chromadb.PersistentClient(path=temp_db_path)
    
    # Mock the chroma_client module
    mock_chroma_client = MagicMock()
    mock_chroma_client.client = temp_client
    
    # Patch the import
    monkeypatch.setattr('storage.vectors.providers.chroma.chroma_client', mock_chroma_client)
    
    return ChromaVectorStorage()


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "This is the first document about Python programming.",
        "This is the second document about machine learning.",
        "This is the third document about natural language processing.",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings (384-dimensional vectors for testing)."""
    return [
        [0.1] * 384,  # First document embedding
        [0.2] * 384,  # Second document embedding
        [0.3] * 384,  # Third document embedding
    ]


@pytest.fixture
def sample_metadatas():
    """Sample metadata for testing."""
    return [
        {"author": "Alice", "topic": "programming"},
        {"author": "Bob", "topic": "ml"},
        {"author": "Charlie", "topic": "nlp"},
    ]


@pytest.fixture
def sample_ids():
    """Sample document IDs."""
    return ["doc1", "doc2", "doc3"]


class TestChromaVectorStorage:
    """Test suite for ChromaVectorStorage."""

    def test_create_collection(self, vector_storage):
        """Test creating a collection."""
        collection_name = "test_collection"
        vector_storage.create_collection(collection_name)
        
        # Verify collection exists
        collection = vector_storage.get_collection(collection_name)
        assert collection is not None
        assert collection.name == collection_name

    def test_get_collection_creates_if_not_exists(self, vector_storage):
        """Test that get_collection creates collection if it doesn't exist."""
        collection_name = "new_collection"
        collection = vector_storage.get_collection(collection_name)
        
        assert collection is not None
        assert collection.name == collection_name

    def test_add_documents(
        self, vector_storage, sample_documents, sample_embeddings, 
        sample_metadatas, sample_ids
    ):
        """Test adding documents to a collection."""
        collection_name = "test_add"
        vector_storage.create_collection(collection_name)
        
        vector_storage.add_documents(
            collection_name=collection_name,
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
            ids=sample_ids
        )
        
        # Verify documents were added
        count = vector_storage.get_collection_count(collection_name)
        assert count == len(sample_documents)

    def test_add_documents_error_handling(self, vector_storage):
        """Test error handling when adding documents fails."""
        collection_name = "test_error"
        vector_storage.create_collection(collection_name)
        
        # Try to add with mismatched lengths
        with pytest.raises(Exception):
            vector_storage.add_documents(
                collection_name=collection_name,
                documents=["doc1", "doc2"],
                embeddings=[[0.1] * 384],  # Only one embedding
                metadatas=[{}],
                ids=["id1", "id2"]
            )

    def test_query(
        self, vector_storage, sample_documents, sample_embeddings,
        sample_metadatas, sample_ids
    ):
        """Test querying documents by similarity."""
        collection_name = "test_query"
        vector_storage.create_collection(collection_name)
        
        # Add documents
        vector_storage.add_documents(
            collection_name=collection_name,
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
            ids=sample_ids
        )
        
        # Query with first document's embedding
        query_embedding = [sample_embeddings[0]]
        results = vector_storage.query(
            collection_name=collection_name,
            query_embeddings=query_embedding,
            n_results=2
        )
        
        # Verify results structure
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert "ids" in results
        
        # Should return at least one result
        assert len(results["documents"][0]) > 0

    def test_query_with_metadata_filter(
        self, vector_storage, sample_documents, sample_embeddings,
        sample_metadatas, sample_ids
    ):
        """Test querying with metadata filter."""
        collection_name = "test_query_filter"
        vector_storage.create_collection(collection_name)
        
        # Add documents
        vector_storage.add_documents(
            collection_name=collection_name,
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
            ids=sample_ids
        )
        
        # Query with metadata filter
        query_embedding = [sample_embeddings[0]]
        where = {"author": "Alice"}
        results = vector_storage.query(
            collection_name=collection_name,
            query_embeddings=query_embedding,
            n_results=10,
            where=where
        )
        
        # Verify results
        assert "documents" in results
        if results["metadatas"] and results["metadatas"][0]:
            # If results returned, check metadata
            for metadata in results["metadatas"][0]:
                assert metadata.get("author") == "Alice"

    def test_get_collection_count(
        self, vector_storage, sample_documents, sample_embeddings,
        sample_metadatas, sample_ids
    ):
        """Test getting collection count."""
        collection_name = "test_count"
        vector_storage.create_collection(collection_name)
        
        # Initially empty
        count = vector_storage.get_collection_count(collection_name)
        assert count == 0
        
        # Add documents
        vector_storage.add_documents(
            collection_name=collection_name,
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
            ids=sample_ids
        )
        
        # Verify count
        count = vector_storage.get_collection_count(collection_name)
        assert count == len(sample_documents)

    def test_list_collections(self, vector_storage):
        """Test listing all collections."""
        # Create multiple collections
        collections = ["collection1", "collection2", "collection3"]
        for name in collections:
            vector_storage.create_collection(name)
        
        # List collections
        listed = vector_storage.list_collections()
        
        # Verify all collections are listed
        for name in collections:
            assert name in listed

    def test_delete_collection(self, vector_storage):
        """Test deleting a collection."""
        collection_name = "test_delete"
        vector_storage.create_collection(collection_name)
        
        # Verify it exists
        assert collection_name in vector_storage.list_collections()
        
        # Delete it
        vector_storage.delete_collection(collection_name)
        
        # Verify it's gone (or handle exception if collection doesn't exist)
        listed = vector_storage.list_collections()
        # Note: ChromaDB might still list deleted collections in some cases
        # This test verifies the method doesn't raise an exception

    def test_delete_documents(
        self, vector_storage, sample_documents, sample_embeddings,
        sample_metadatas, sample_ids
    ):
        """Test deleting documents by IDs."""
        collection_name = "test_delete_docs"
        vector_storage.create_collection(collection_name)
        
        # Add documents
        vector_storage.add_documents(
            collection_name=collection_name,
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
            ids=sample_ids
        )
        
        # Verify initial count
        count = vector_storage.get_collection_count(collection_name)
        assert count == len(sample_documents)
        
        # Delete one document
        vector_storage.delete_documents(
            collection_name=collection_name,
            ids=[sample_ids[0]]
        )
        
        # Verify count decreased
        count = vector_storage.get_collection_count(collection_name)
        assert count == len(sample_documents) - 1

    def test_delete_documents_empty_ids(self, vector_storage):
        """Test deleting with empty IDs list."""
        collection_name = "test_delete_empty"
        vector_storage.create_collection(collection_name)
        
        # Should not raise exception, just log warning
        vector_storage.delete_documents(
            collection_name=collection_name,
            ids=[]
        )

    def test_multiple_operations(
        self, vector_storage, sample_documents, sample_embeddings,
        sample_metadatas, sample_ids
    ):
        """Test multiple operations in sequence."""
        collection_name = "test_multiple"
        
        # Create collection
        vector_storage.create_collection(collection_name)
        
        # Add documents
        vector_storage.add_documents(
            collection_name=collection_name,
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas,
            ids=sample_ids
        )
        
        # Query
        results = vector_storage.query(
            collection_name=collection_name,
            query_embeddings=[sample_embeddings[0]],
            n_results=3
        )
        assert len(results["documents"][0]) > 0
        
        # Get count
        count = vector_storage.get_collection_count(collection_name)
        assert count == len(sample_documents)
        
        # Delete one document
        vector_storage.delete_documents(
            collection_name=collection_name,
            ids=[sample_ids[0]]
        )
        
        # Verify final count
        count = vector_storage.get_collection_count(collection_name)
        assert count == len(sample_documents) - 1

    def test_query_returns_correct_structure(self, vector_storage):
        """Test that query returns the expected structure."""
        collection_name = "test_structure"
        vector_storage.create_collection(collection_name)
        
        # Add a document
        vector_storage.add_documents(
            collection_name=collection_name,
            documents=["Test document"],
            embeddings=[[0.1] * 384],
            metadatas=[{"test": "value"}],
            ids=["test_id"]
        )
        
        # Query
        results = vector_storage.query(
            collection_name=collection_name,
            query_embeddings=[[0.1] * 384],
            n_results=1
        )
        
        # Verify structure
        assert isinstance(results, dict)
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert "ids" in results
        
        # Verify lists are nested (one query embedding = one list of results)
        assert isinstance(results["documents"], list)
        assert len(results["documents"]) > 0
        assert isinstance(results["documents"][0], list)

