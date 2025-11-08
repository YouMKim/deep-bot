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


class TestNewCheckpointMethods:
    """Test suite for new checkpoint methods"""
    
    def test_get_recent_messages(self, temp_db, sample_messages):
        """Test getting N most recent messages in chronological order"""
        channel_id = 'test_channel'
        
        # Add more messages to test limit
        extra_messages = [
            {
                'id': '126',
                'content': 'Fourth message',
                'author_id': '456',
                'author': 'TestUser',
                'author_display_name': 'TestUser',
                'timestamp': '2024-01-01T00:03:00Z',
                'created_at': '2024-01-01T00:03:00Z',
                'channel_name': 'test-channel',
                'guild_name': 'Test Guild',
                'guild_id': '789',
                'is_bot': False,
                'has_attachments': False,
                'message_type': 'default',
                'metadata': {}
            },
            {
                'id': '127',
                'content': 'Fifth message',
                'author_id': '456',
                'author': 'TestUser',
                'author_display_name': 'TestUser',
                'timestamp': '2024-01-01T00:04:00Z',
                'created_at': '2024-01-01T00:04:00Z',
                'channel_name': 'test-channel',
                'guild_name': 'Test Guild',
                'guild_id': '789',
                'is_bot': False,
                'has_attachments': False,
                'message_type': 'default',
                'metadata': {}
            }
        ]
        
        temp_db.save_channel_messages(channel_id, sample_messages + extra_messages)
        
        # Get 3 most recent messages
        recent = temp_db.get_recent_messages(channel_id, 3)
        
        assert len(recent) == 3
        # Should be in chronological order (oldest to newest of the recent 3)
        assert recent[0]['message_id'] == '125'  # 3rd oldest overall
        assert recent[1]['message_id'] == '126'  # 2nd newest
        assert recent[2]['message_id'] == '127'  # newest
    
    def test_get_recent_messages_with_limit_greater_than_total(self, temp_db, sample_messages):
        """Test getting recent messages when limit exceeds total messages"""
        channel_id = 'test_channel'
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        # Request more messages than exist
        recent = temp_db.get_recent_messages(channel_id, 100)
        
        # Should return all 3 messages
        assert len(recent) == 3
        assert recent[0]['message_id'] == '123'
        assert recent[2]['message_id'] == '125'
    
    def test_get_messages_after(self, temp_db, sample_messages):
        """Test getting messages after a specific message ID"""
        channel_id = 'test_channel'
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        # Get messages after the first message
        after = temp_db.get_messages_after(channel_id, '123', limit=10)
        
        assert len(after) == 2
        assert after[0]['message_id'] == '124'
        assert after[1]['message_id'] == '125'
    
    def test_get_messages_after_with_limit(self, temp_db, sample_messages):
        """Test getting messages after with a limit"""
        channel_id = 'test_channel'
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        # Get only 1 message after the first
        after = temp_db.get_messages_after(channel_id, '123', limit=1)
        
        assert len(after) == 1
        assert after[0]['message_id'] == '124'
    
    def test_get_messages_after_nonexistent_id(self, temp_db, sample_messages):
        """Test getting messages after a non-existent message ID"""
        channel_id = 'test_channel'
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        # Try to get messages after non-existent ID
        after = temp_db.get_messages_after(channel_id, '999', limit=10)
        
        # Should return empty list
        assert len(after) == 0
    
    def test_get_messages_after_last_message(self, temp_db, sample_messages):
        """Test getting messages after the last message"""
        channel_id = 'test_channel'
        temp_db.save_channel_messages(channel_id, sample_messages)
        
        # Get messages after the last message
        after = temp_db.get_messages_after(channel_id, '125', limit=10)
        
        # Should return empty list
        assert len(after) == 0
    
    def test_chunking_checkpoint_crud(self, temp_db):
        """Test chunking checkpoint create, read, update operations"""
        channel_id = 'test_channel'
        strategy = 'temporal'
        
        # Initially no checkpoint
        checkpoint = temp_db.get_chunking_checkpoint(channel_id, strategy)
        assert checkpoint is None
        
        # Create checkpoint
        temp_db.update_chunking_checkpoint(
            channel_id=channel_id,
            strategy=strategy,
            last_chunk_id='chunk_123',
            last_message_id='msg_456',
            last_timestamp='2024-01-01T00:00:00Z'
        )
        
        # Retrieve checkpoint
        checkpoint = temp_db.get_chunking_checkpoint(channel_id, strategy)
        assert checkpoint is not None
        assert checkpoint['last_chunk_id'] == 'chunk_123'
        assert checkpoint['last_message_id'] == 'msg_456'
        assert checkpoint['last_message_timestamp'] == '2024-01-01T00:00:00Z'
        assert 'updated_at' in checkpoint
        
        # Update checkpoint
        temp_db.update_chunking_checkpoint(
            channel_id=channel_id,
            strategy=strategy,
            last_chunk_id='chunk_789',
            last_message_id='msg_999',
            last_timestamp='2024-01-01T00:05:00Z'
        )
        
        # Verify update
        checkpoint = temp_db.get_chunking_checkpoint(channel_id, strategy)
        assert checkpoint['last_chunk_id'] == 'chunk_789'
        assert checkpoint['last_message_id'] == 'msg_999'
        assert checkpoint['last_message_timestamp'] == '2024-01-01T00:05:00Z'
    
    def test_chunking_checkpoint_multiple_strategies(self, temp_db):
        """Test checkpoints for multiple strategies on same channel"""
        channel_id = 'test_channel'
        
        # Create checkpoints for different strategies
        temp_db.update_chunking_checkpoint(
            channel_id=channel_id,
            strategy='temporal',
            last_chunk_id='chunk_t_1',
            last_message_id='msg_100',
            last_timestamp='2024-01-01T00:00:00Z'
        )
        
        temp_db.update_chunking_checkpoint(
            channel_id=channel_id,
            strategy='author',
            last_chunk_id='chunk_a_1',
            last_message_id='msg_200',
            last_timestamp='2024-01-01T00:10:00Z'
        )
        
        # Retrieve each checkpoint
        temporal_cp = temp_db.get_chunking_checkpoint(channel_id, 'temporal')
        author_cp = temp_db.get_chunking_checkpoint(channel_id, 'author')
        
        # Verify they're independent
        assert temporal_cp['last_message_id'] == 'msg_100'
        assert author_cp['last_message_id'] == 'msg_200'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

