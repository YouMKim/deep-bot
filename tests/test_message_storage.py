"""
Tests for MessageStorage service.

Tests cover:
- Message saving and retrieval
- Checkpoint management
- Batch operations
- Idempotency (duplicate handling)
- Channel statistics
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime
from storage.messages import MessageStorage


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_messages.db")
    
    storage = MessageStorage(db_path)
    yield storage
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_messages():
    """Sample messages for testing"""
    return [
        {
            'id': '123',
            'content': 'Hello world',
            'author_id': '456',
            'author': 'TestUser',
            'author_display_name': 'TestUser',
            'timestamp': '2024-01-01T00:00:00Z',
            'created_at': '2024-01-01T00:00:00Z',
            'channel_name': 'test-channel',
            'guild_name': 'Test Guild',
            'guild_id': '789',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        },
        {
            'id': '124',
            'content': 'Second message',
            'author_id': '456',
            'author': 'TestUser',
            'author_display_name': 'TestUser',
            'timestamp': '2024-01-01T00:01:00Z',
            'created_at': '2024-01-01T00:01:00Z',
            'channel_name': 'test-channel',
            'guild_name': 'Test Guild',
            'guild_id': '789',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        },
        {
            'id': '125',
            'content': 'Third message',
            'author_id': '789',
            'author': 'AnotherUser',
            'author_display_name': 'AnotherUser',
            'timestamp': '2024-01-01T00:02:00Z',
            'created_at': '2024-01-01T00:02:00Z',
            'channel_name': 'test-channel',
            'guild_name': 'Test Guild',
            'guild_id': '789',
            'is_bot': True,
            'has_attachments': True,
            'message_type': 'default',
            'metadata': {'reactions': 5}
        }
    ]


class TestMessageStorage:
    """Test suite for MessageStorage"""
    
    def test_save_and_load_messages(self, temp_db, sample_messages):
        """Test basic save and load functionality"""
        channel_id = 'test_channel'
        
        # Save messages
        result = temp_db.save_channel_messages(channel_id, sample_messages)
        assert result is True
        
        # Load all messages
        loaded = temp_db.load_channel_messages(channel_id)
        assert len(loaded) == 3
        
        # Verify first message
        assert loaded[0]['id'] == '123'
        assert loaded[0]['content'] == 'Hello world'
        assert loaded[0]['author'] == 'TestUser'
        assert loaded[0]['is_bot'] is False
        
        # Verify last message
        assert loaded[2]['id'] == '125'
        assert loaded[2]['is_bot'] is True
        assert loaded[2]['has_attachments'] is True
        assert loaded[2]['metadata']['reactions'] == 5
    
    def test_messages_ordered_by_timestamp(self, temp_db, sample_messages):
        """Test that messages are loaded in chronological order"""
        channel_id = 'test_channel'
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        loaded = temp_db.load_channel_messages(channel_id)
        
        # Should be ordered by timestamp ASC (oldest first)
        assert loaded[0]['timestamp'] == '2024-01-01T00:00:00Z'
        assert loaded[1]['timestamp'] == '2024-01-01T00:01:00Z'
        assert loaded[2]['timestamp'] == '2024-01-01T00:02:00Z'
    
    def test_idempotency_duplicate_messages(self, temp_db, sample_messages):
        """Test that duplicate messages are ignored (INSERT OR IGNORE)"""
        channel_id = 'test_channel'
        
        # Save messages twice
        temp_db.save_channel_messages(channel_id, sample_messages)
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        # Should still have only 3 messages, not 6
        loaded = temp_db.load_channel_messages(channel_id)
        assert len(loaded) == 3
    
    def test_checkpoint_creation(self, temp_db, sample_messages):
        """Test that checkpoints are created after saving"""
        channel_id = 'test_channel'
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        checkpoint = temp_db.get_checkpoint(channel_id)
        assert checkpoint is not None
        assert checkpoint['last_message_id'] == '125'  # Newest message
        assert checkpoint['total_messages'] == 3
        assert checkpoint['oldest_message_timestamp'] == '2024-01-01T00:00:00Z'
        assert checkpoint['newest_message_timestamp'] == '2024-01-01T00:02:00Z'
    
    def test_checkpoint_nonexistent_channel(self, temp_db):
        """Test getting checkpoint for non-existent channel"""
        checkpoint = temp_db.get_checkpoint('nonexistent_channel')
        assert checkpoint is None
    
    def test_empty_messages_list(self, temp_db):
        """Test saving empty messages list"""
        result = temp_db.save_channel_messages('test_channel', [])
        assert result is True
        
        loaded = temp_db.load_channel_messages('test_channel')
        assert len(loaded) == 0
    
    def test_channel_stats(self, temp_db, sample_messages):
        """Test getting channel statistics"""
        channel_id = 'test_channel'
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        stats = temp_db.get_channel_stats(channel_id)
        assert stats['channel_id'] == channel_id
        assert stats['message_count'] == 3
        assert stats['oldest_timestamp'] == '2024-01-01T00:00:00Z'
        assert stats['newest_timestamp'] == '2024-01-01T00:02:00Z'
        assert stats['checkpoint'] is not None
    
    def test_multiple_channels(self, temp_db, sample_messages):
        """Test storing messages for multiple channels"""
        channel1 = 'channel_1'
        channel2 = 'channel_2'
        
        # Save messages to different channels
        temp_db.save_channel_messages(channel1, sample_messages[:2])
        temp_db.save_channel_messages(channel2, sample_messages[2:])
        
        # Verify each channel has correct messages
        loaded1 = temp_db.load_channel_messages(channel1)
        loaded2 = temp_db.load_channel_messages(channel2)
        
        assert len(loaded1) == 2
        assert len(loaded2) == 1
        assert loaded1[0]['channel_id'] == channel1
        assert loaded2[0]['channel_id'] == channel2
    
    def test_metadata_json_handling(self, temp_db):
        """Test that metadata is properly serialized/deserialized"""
        channel_id = 'test_channel'
        message_with_metadata = {
            'id': '999',
            'content': 'Test',
            'author_id': '123',
            'author': 'Test',
            'author_display_name': 'Test',
            'timestamp': '2024-01-01T00:00:00Z',
            'created_at': '2024-01-01T00:00:00Z',
            'channel_name': 'test',
            'guild_name': 'test',
            'guild_id': '123',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {'key1': 'value1', 'key2': 42, 'nested': {'foo': 'bar'}}
        }
        
        temp_db.save_channel_messages(channel_id, [message_with_metadata])
        loaded = temp_db.load_channel_messages(channel_id)
        
        assert loaded[0]['metadata']['key1'] == 'value1'
        assert loaded[0]['metadata']['key2'] == 42
        assert loaded[0]['metadata']['nested']['foo'] == 'bar'
    
    def test_incremental_save(self, temp_db, sample_messages):
        """Test saving messages incrementally"""
        channel_id = 'test_channel'
        
        # Save first batch
        temp_db.save_channel_messages(channel_id, sample_messages[:2])
        assert len(temp_db.load_channel_messages(channel_id)) == 2
        
        # Save second batch
        temp_db.save_channel_messages(channel_id, sample_messages[2:])
        assert len(temp_db.load_channel_messages(channel_id)) == 3
        
        # Checkpoint should be updated
        checkpoint = temp_db.get_checkpoint(channel_id)
        assert checkpoint['total_messages'] == 3
        assert checkpoint['last_message_id'] == '125'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

