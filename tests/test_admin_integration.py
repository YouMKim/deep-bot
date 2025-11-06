"""
Integration tests for Admin cog with MessageStorage and MessageLoader.

Tests cover:
- Admin cog initialization with MessageStorage
- Database operations through admin commands
- Checkpoint/resume functionality
- SQLite database verification
"""

import pytest
import os
import tempfile
import shutil
import sqlite3
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import discord
from discord.ext import commands

from storage.messages import MessageStorage
from bot.loaders.message_loader import MessageLoader
from bot.cogs.admin import Admin


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_messages.db")
    
    storage = MessageStorage(db_path)
    yield storage, db_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_bot():
    """Create a mock Discord bot"""
    bot = MagicMock(spec=commands.Bot)
    bot.owner_id = 123456789
    return bot


@pytest.fixture
def admin_cog(mock_bot, temp_db):
    """Create Admin cog with temporary database"""
    storage, db_path = temp_db
    # Monkey patch the MessageStorage to use our temp db
    admin = Admin(mock_bot)
    admin.message_storage = storage
    admin.message_loader = MessageLoader(storage)
    return admin, db_path


@pytest.fixture
def mock_ctx(mock_bot):
    """Create a mock Discord context"""
    ctx = MagicMock(spec=commands.Context)
    ctx.bot = mock_bot
    ctx.author = MagicMock()
    ctx.author.id = 123456789  # Same as bot owner
    ctx.author.display_name = "TestUser"
    ctx.channel = MagicMock(spec=discord.TextChannel)
    ctx.channel.id = 987654321
    ctx.channel.name = "test-channel"
    ctx.send = AsyncMock()
    return ctx


@pytest.fixture
def sample_messages():
    """Sample messages for testing"""
    return [
        {
            'id': '111',
            'content': 'Message 1',
            'author_id': '456',
            'author': 'User1',
            'author_display_name': 'User1',
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
            'id': '222',
            'content': 'Message 2',
            'author_id': '456',
            'author': 'User1',
            'author_display_name': 'User1',
            'timestamp': '2024-01-01T01:00:00Z',
            'created_at': '2024-01-01T01:00:00Z',
            'channel_name': 'test-channel',
            'guild_name': 'Test Guild',
            'guild_id': '789',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        },
        {
            'id': '333',
            'content': 'Message 3',
            'author_id': '456',
            'author': 'User1',
            'author_display_name': 'User1',
            'timestamp': '2024-01-01T02:00:00Z',
            'created_at': '2024-01-01T02:00:00Z',
            'channel_name': 'test-channel',
            'guild_name': 'Test Guild',
            'guild_id': '789',
            'is_bot': False,
            'has_attachments': False,
            'message_type': 'default',
            'metadata': {}
        }
    ]


class TestAdminCogInitialization:
    """Test Admin cog initialization"""
    
    def test_admin_cog_initializes_with_message_storage(self, mock_bot, temp_db):
        """Test that Admin cog initializes with MessageStorage"""
        storage, db_path = temp_db
        admin = Admin(mock_bot)
        admin.message_storage = storage
        admin.message_loader = MessageLoader(storage)
        
        assert admin.message_storage is not None
        assert admin.message_loader is not None
        assert isinstance(admin.message_storage, MessageStorage)
        assert isinstance(admin.message_loader, MessageLoader)
    
    def test_message_loader_uses_message_storage(self, admin_cog):
        """Test that MessageLoader uses the correct MessageStorage instance"""
        admin, db_path = admin_cog
        assert admin.message_loader.message_storage is admin.message_storage


class TestAdminDatabaseOperations:
    """Test Admin cog database operations"""
    
    def test_admin_cog_has_message_storage(self, admin_cog):
        """Test that admin cog has message storage initialized"""
        admin, db_path = admin_cog
        assert admin.message_storage is not None
        assert admin.message_loader is not None
    
    def test_check_storage_data_with_no_messages(self, admin_cog, mock_ctx):
        """Test get_channel_stats when no messages are stored"""
        admin, db_path = admin_cog
        channel_id = str(mock_ctx.channel.id)
        
        stats = admin.message_storage.get_channel_stats(channel_id)
        
        assert stats['message_count'] == 0
        assert stats['channel_id'] == channel_id
        assert stats['oldest_timestamp'] is None
        assert stats['newest_timestamp'] is None
        assert stats['checkpoint'] is None
    
    def test_check_storage_data_with_messages(self, admin_cog, mock_ctx, sample_messages):
        """Test get_channel_stats when messages are stored"""
        admin, db_path = admin_cog
        channel_id = str(mock_ctx.channel.id)
        
        # Save some messages first
        admin.message_storage.save_channel_messages(channel_id, sample_messages)
        
        # Get stats
        stats = admin.message_storage.get_channel_stats(channel_id)
        
        assert stats['message_count'] == 3
        assert stats['channel_id'] == channel_id
        assert stats['oldest_timestamp'] == '2024-01-01T00:00:00Z'
        assert stats['newest_timestamp'] == '2024-01-01T02:00:00Z'
        assert stats['checkpoint'] is not None
    
    def test_checkpoint_info_with_no_checkpoint(self, admin_cog, mock_ctx):
        """Test get_checkpoint when no checkpoint exists"""
        admin, db_path = admin_cog
        channel_id = str(mock_ctx.channel.id)
        
        checkpoint = admin.message_storage.get_checkpoint(channel_id)
        
        assert checkpoint is None
    
    def test_checkpoint_info_with_checkpoint(self, admin_cog, mock_ctx, sample_messages):
        """Test get_checkpoint when checkpoint exists"""
        admin, db_path = admin_cog
        channel_id = str(mock_ctx.channel.id)
        
        # Save messages to create checkpoint
        admin.message_storage.save_channel_messages(channel_id, sample_messages)
        
        # Get checkpoint
        checkpoint = admin.message_storage.get_checkpoint(channel_id)
        
        assert checkpoint is not None
        assert checkpoint['total_messages'] == 3
        assert checkpoint['last_message_id'] == '333'  # Newest message
        assert checkpoint['oldest_message_id'] == '111'  # Oldest message


class TestSQLiteDatabaseVerification:
    """Test SQLite database structure and data integrity"""
    
    def test_database_tables_exist(self, temp_db):
        """Test that required tables exist in database"""
        storage, db_path = temp_db
        
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('messages', 'checkpoints')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'messages' in tables
            assert 'checkpoints' in tables
    
    def test_messages_table_schema(self, temp_db):
        """Test that messages table has correct schema"""
        storage, db_path = temp_db
        
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(messages)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            # Check required columns exist
            assert 'message_id' in columns
            assert 'channel_id' in columns
            assert 'content' in columns
            assert 'timestamp' in columns
            assert columns['message_id'] == 'TEXT'
            assert columns['channel_id'] == 'TEXT'
    
    def test_checkpoints_table_schema(self, temp_db):
        """Test that checkpoints table has correct schema"""
        storage, db_path = temp_db
        
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(checkpoints)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            # Check required columns exist
            assert 'channel_id' in columns
            assert 'last_message_id' in columns
            assert 'total_messages' in columns
            assert columns['channel_id'] == 'TEXT'
            assert columns['total_messages'] == 'INTEGER'
    
    def test_message_save_and_retrieve(self, temp_db, sample_messages):
        """Test saving and retrieving messages from database"""
        storage, db_path = temp_db
        
        channel_id = "123456789"
        storage.save_channel_messages(channel_id, sample_messages)
        
        # Retrieve messages
        messages = storage.load_channel_messages(channel_id)
        
        assert len(messages) == 3
        assert messages[0]['id'] == '111'
        assert messages[1]['id'] == '222'
        assert messages[2]['id'] == '333'
    
    def test_checkpoint_creation(self, temp_db, sample_messages):
        """Test that checkpoint is created after saving messages"""
        storage, db_path = temp_db
        
        channel_id = "123456789"
        storage.save_channel_messages(channel_id, sample_messages)
        
        checkpoint = storage.get_checkpoint(channel_id)
        
        assert checkpoint is not None
        assert checkpoint['total_messages'] == 3
        assert checkpoint['last_message_id'] == '333'  # Newest message
        assert checkpoint['oldest_message_id'] == '111'  # Oldest message
    
    def test_duplicate_message_handling(self, temp_db, sample_messages):
        """Test that duplicate messages are handled correctly (INSERT OR IGNORE)"""
        storage, db_path = temp_db
        
        channel_id = "123456789"
        
        # Save messages twice
        storage.save_channel_messages(channel_id, sample_messages)
        storage.save_channel_messages(channel_id, sample_messages)
        
        # Should only have 3 messages, not 6
        messages = storage.load_channel_messages(channel_id)
        assert len(messages) == 3
    
    def test_channel_stats(self, temp_db, sample_messages):
        """Test get_channel_stats returns correct information"""
        storage, db_path = temp_db
        
        channel_id = "123456789"
        storage.save_channel_messages(channel_id, sample_messages)
        
        stats = storage.get_channel_stats(channel_id)
        
        assert stats['message_count'] == 3
        assert stats['channel_id'] == channel_id
        assert stats['oldest_timestamp'] == '2024-01-01T00:00:00Z'
        assert stats['newest_timestamp'] == '2024-01-01T02:00:00Z'
        assert stats['checkpoint'] is not None


class TestAdminCommandIntegration:
    """Test full integration of admin commands with database"""
    
    @pytest.mark.asyncio
    async def test_load_channel_integration(self, admin_cog, mock_ctx, sample_messages):
        """Test that load_channel would work with mocked message loader"""
        admin, db_path = admin_cog
        
        # Mock the message loader's load_channel_messages
        async def mock_load(channel, limit=None, **kwargs):
            # Actually save messages to test real integration
            channel_id = str(channel.id)
            admin.message_storage.save_channel_messages(channel_id, sample_messages)
            return {
                'total_processed': 3,
                'successfully_loaded': 3,
                'skipped_bot_messages': 0,
                'skipped_empty_messages': 0,
                'skipped_commands': 0,
                'errors': 0,
                'rate_limit_errors': 0,
                'batches_saved': 1,
                'resumed_from_checkpoint': False,
                'start_time': datetime.now(),
                'end_time': datetime.now()
            }
        
        admin.message_loader.load_channel_messages = mock_load
        
        # Mock Config to pass owner check - patch it where it's used
        # Config is imported inside load_channel, so we need to patch it at runtime
        import config
        original_owner_id = config.Config.BOT_OWNER_ID
        config.Config.BOT_OWNER_ID = str(mock_ctx.author.id)
        
        try:
            # Mock ctx.send to capture the status message
            mock_ctx.send.return_value = AsyncMock()
            status_msg = mock_ctx.send.return_value
            status_msg.edit = AsyncMock()
            
            # Get the command callback and call it directly
            load_channel_cmd = type(admin).load_channel
            if hasattr(load_channel_cmd, 'callback'):
                await load_channel_cmd.callback(admin, mock_ctx, limit=None)
            else:
                await load_channel_cmd(admin, mock_ctx, limit=None)
        finally:
            # Restore original
            config.Config.BOT_OWNER_ID = original_owner_id
        
        # Verify messages were stored
        channel_id = str(mock_ctx.channel.id)
        messages = admin.message_storage.load_channel_messages(channel_id)
        assert len(messages) == 3
        
        # Verify checkpoint exists
        checkpoint = admin.message_storage.get_checkpoint(channel_id)
        assert checkpoint is not None
        assert checkpoint['total_messages'] == 3


class TestDatabaseIndexes:
    """Test that database indexes exist for performance"""
    
    def test_messages_table_indexes(self, temp_db):
        """Test that messages table has required indexes"""
        storage, db_path = temp_db
        
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='messages'
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Check for important indexes
            assert any('channel_timestamp' in idx for idx in indexes)
            assert any('idx_messages_id' in idx for idx in indexes)
    
    def test_checkpoints_table_indexes(self, temp_db):
        """Test that checkpoints table has required indexes"""
        storage, db_path = temp_db
        
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='checkpoints'
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Check for important indexes
            assert any('checkpoint' in idx.lower() for idx in indexes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

