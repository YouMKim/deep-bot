"""
Tests for author filtering helper method and batching functionality.
"""
import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from storage.chunked_memory import ChunkedMemoryService
from storage.messages.messages import MessageStorage
from chunking.base import Chunk


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_messages.db")
    
    storage = MessageStorage(db_path)
    yield storage
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_vector_store():
    """Mock vector storage"""
    mock = Mock()
    mock.create_collection = Mock()
    mock.add_documents = Mock()
    mock.query = Mock(return_value={'documents': [], 'metadatas': [], 'distances': []})
    return mock


@pytest.fixture
def mock_chunking_service():
    """Mock chunking service"""
    mock = Mock()
    return mock


class TestShouldIncludeAuthor:
    """Test the _should_include_author helper method"""
    
    def test_no_filters(self):
        """Test that all authors pass when no filters applied."""
        mock_config = Mock()
        mock_config.BLACKLIST_IDS = []
        
        service = ChunkedMemoryService(config=mock_config)
        
        assert service._should_include_author("Alice", False, None) == True
        assert service._should_include_author("Bob", False, None) == True
    
    def test_blacklist_filtering(self):
        """Test that blacklisted authors are filtered."""
        mock_config = Mock()
        mock_config.BLACKLIST_IDS = [12345, 67890]
        
        service = ChunkedMemoryService(config=mock_config)
        
        # Test with blacklist enabled
        assert service._should_include_author(12345, True, None) == False
        assert service._should_include_author(67890, True, None) == False
        assert service._should_include_author(99999, True, None) == True
        
        # Test with string IDs
        assert service._should_include_author("12345", True, None) == False
        assert service._should_include_author("67890", True, None) == False
        
        # Test with blacklist disabled
        assert service._should_include_author(12345, False, None) == True
    
    def test_whitelist_filtering(self):
        """Test that only whitelisted authors pass."""
        mock_config = Mock()
        mock_config.BLACKLIST_IDS = []
        
        service = ChunkedMemoryService(config=mock_config)
        
        assert service._should_include_author("Alice", False, ["Alice", "Bob"]) == True
        assert service._should_include_author("Bob", False, ["Alice", "Bob"]) == True
        assert service._should_include_author("Charlie", False, ["Alice", "Bob"]) == False
    
    def test_case_insensitive_matching(self):
        """Test case-insensitive matching."""
        mock_config = Mock()
        mock_config.BLACKLIST_IDS = []
        
        service = ChunkedMemoryService(config=mock_config)
        
        assert service._should_include_author("alice", False, ["Alice"]) == True
        assert service._should_include_author("ALICE", False, ["alice"]) == True
        assert service._should_include_author("Alice", False, ["ALICE"]) == True
    
    def test_partial_matching(self):
        """Test partial matching for author names."""
        mock_config = Mock()
        mock_config.BLACKLIST_IDS = []
        
        service = ChunkedMemoryService(config=mock_config)
        
        # Should match if filter contains author or author contains filter
        assert service._should_include_author("Alice Smith", False, ["Alice"]) == True
        assert service._should_include_author("Alice", False, ["Alice Smith"]) == True
        assert service._should_include_author("Bob", False, ["Alice"]) == False
    
    def test_combined_filters(self):
        """Test combining blacklist and whitelist."""
        mock_config = Mock()
        mock_config.BLACKLIST_IDS = [12345]
        
        service = ChunkedMemoryService(config=mock_config)
        
        # Blacklisted author should be filtered even if in whitelist
        assert service._should_include_author(12345, True, ["Alice", "Bob"]) == False
        
        # Non-blacklisted author in whitelist should pass
        assert service._should_include_author("Alice", True, ["Alice", "Bob"]) == True
        
        # Non-blacklisted author not in whitelist should fail
        assert service._should_include_author("Charlie", True, ["Alice", "Bob"]) == False


class TestEmbeddingBatching:
    """Test embedding batching functionality"""
    
    @pytest.mark.asyncio
    async def test_batching_splits_large_documents(self, temp_db, mock_vector_store, mock_chunking_service):
        """Test that large document sets are split into batches."""
        from unittest.mock import Mock
        
        # Track batch calls
        batch_calls = []
        
        async def mock_sleep(delay):
            batch_calls.append(('sleep', delay))
        
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        
        # Simulate batch encoding - return embeddings for each batch
        def encode_batch_side_effect(docs):
            batch_calls.append(('encode_batch', len(docs)))
            return [[0.1] * 384 for _ in docs]
        
        mock_embedder.encode_batch = Mock(side_effect=encode_batch_side_effect)
        
        service = ChunkedMemoryService(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            message_storage=temp_db,
            chunking_service=mock_chunking_service
        )
        
        # Create 250 chunks (should be split into 3 batches with default batch_size=100)
        chunks = {
            'single': [
                Chunk(
                    content=f'Test message {i}',
                    metadata={'first_message_id': str(i), 'author': 'Alice'},
                    message_ids=[str(i)]
                )
                for i in range(250)
            ]
        }
        
        # Mock asyncio.sleep to track delays
        import asyncio
        original_sleep = asyncio.sleep
        asyncio.sleep = mock_sleep
        
        try:
            await service.store_all_strategies(chunks)
        finally:
            asyncio.sleep = original_sleep
        
        # Verify batching occurred
        assert len([c for c in batch_calls if c[0] == 'encode_batch']) == 3  # 3 batches
        
        # Verify delays between batches (should have 2 sleeps for 3 batches)
        sleep_calls = [c for c in batch_calls if c[0] == 'sleep']
        assert len(sleep_calls) == 2  # Delays between batches (not after last)
        
        # Verify all documents were stored
        assert mock_vector_store.add_documents.called
        call_args = mock_vector_store.add_documents.call_args
        assert len(call_args[1]['documents']) == 250
    
    @pytest.mark.asyncio
    async def test_small_batch_no_delay(self, temp_db, mock_vector_store, mock_chunking_service):
        """Test that small batches don't trigger delays."""
        from unittest.mock import Mock
        
        batch_calls = []
        
        async def mock_sleep(delay):
            batch_calls.append(('sleep', delay))
        
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        mock_embedder.encode_batch = Mock(return_value=[[0.1] * 384, [0.2] * 384])
        
        service = ChunkedMemoryService(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            message_storage=temp_db,
            chunking_service=mock_chunking_service
        )
        
        chunks = {
            'single': [
                Chunk(content='Test message 1', metadata={'first_message_id': '1', 'author': 'Alice'}, message_ids=['1']),
                Chunk(content='Test message 2', metadata={'first_message_id': '2', 'author': 'Bob'}, message_ids=['2']),
            ]
        }
        
        # Mock asyncio.sleep
        import asyncio
        original_sleep = asyncio.sleep
        asyncio.sleep = mock_sleep
        
        try:
            await service.store_all_strategies(chunks)
        finally:
            asyncio.sleep = original_sleep
        
        # Should have no sleep calls (only 1 batch, so no delays needed)
        sleep_calls = [c for c in batch_calls if c[0] == 'sleep']
        assert len(sleep_calls) == 0
    
    @pytest.mark.asyncio
    async def test_custom_batch_size(self, temp_db, mock_vector_store, mock_chunking_service):
        """Test that custom batch size works."""
        from unittest.mock import Mock
        
        batch_sizes = []
        
        mock_embedder = Mock()
        mock_embedder.dimension = 384
        
        def encode_batch_side_effect(docs):
            batch_sizes.append(len(docs))
            return [[0.1] * 384 for _ in docs]
        
        mock_embedder.encode_batch = Mock(side_effect=encode_batch_side_effect)
        
        service = ChunkedMemoryService(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            message_storage=temp_db,
            chunking_service=mock_chunking_service
        )
        
        # Create 50 chunks
        chunks = {
            'single': [
                Chunk(
                    content=f'Test message {i}',
                    metadata={'first_message_id': str(i), 'author': 'Alice'},
                    message_ids=[str(i)]
                )
                for i in range(50)
            ]
        }
        
        # Override batch size to 20
        await service._embed_in_batches(
            [chunk.content for chunk in chunks['single']],
            batch_size=20,
            delay=0  # No delay for faster test
        )
        
        # Should have 3 batches: 20, 20, 10
        assert len(batch_sizes) == 3
        assert batch_sizes[0] == 20
        assert batch_sizes[1] == 20
        assert batch_sizes[2] == 10

