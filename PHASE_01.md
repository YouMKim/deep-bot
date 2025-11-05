# Phase 1: Foundation - Message Storage Abstraction

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Foundation - Message Storage Abstraction

### Learning Objectives
- Understand when to use SQLite vs JSON files
- Learn checkpoint/resume patterns for long-running operations
- Practice database schema design
- Learn transaction management
- Understand connection pooling and context managers
- Learn inheritance patterns and code reuse with base classes
- Practice DRY (Don't Repeat Yourself) principles

### Design Principles
- **Separation of Concerns**: Storage logic separated from business logic
- **Database Abstraction**: Design interface that could swap databases later
- **Transaction Management**: Ensure data integrity with atomic operations
- **Resource Management**: Use context managers for proper cleanup
- **Code Reuse**: Shared base classes reduce duplication and make maintenance easier

### Implementation Steps

#### Step 1.1: Design Database Schema

Create `services/message_storage.py` and design the schema:

```python
# Schema design considerations:
# - Primary keys for fast lookups
# - Indexes on frequently queried columns (channel_id, timestamp)
# - JSON blob for flexible metadata storage
# - Timestamps for debugging and auditing

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    guild_id TEXT,
    content TEXT NOT NULL,
    author_id TEXT NOT NULL,
    author_name TEXT,
    author_display_name TEXT,
    channel_name TEXT,
    guild_name TEXT,
    timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_bot INTEGER DEFAULT 0,
    has_attachments INTEGER DEFAULT 0,
    message_type TEXT DEFAULT 'default',
    metadata TEXT  -- JSON blob for extra fields
);

CREATE TABLE IF NOT EXISTS checkpoints (
    channel_id TEXT PRIMARY KEY,
    last_message_id TEXT NOT NULL,
    last_fetch_timestamp TEXT NOT NULL,
    total_messages INTEGER DEFAULT 0,
    oldest_message_id TEXT,
    oldest_message_timestamp TEXT,
    newest_message_timestamp TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_channel_timestamp 
    ON messages(channel_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_id 
    ON messages(message_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_channel 
    ON checkpoints(channel_id);
"""
```

**Key Decisions:**
- Why TEXT for IDs? Discord uses snowflake IDs (strings, not integers)
- Why timestamp DESC? We query recent messages most often
- Why JSON metadata? Flexibility for future fields without schema changes

#### Step 1.1.5: Create Base SQLiteStorage Class

Before implementing `MessageStorage`, we'll create a shared base class that handles common SQLite operations. This follows the DRY principle and makes it easy to add more database services later.

**Note**: We place this in the `data/` directory (not `services/`) to keep storage infrastructure code together. This distinguishes it from vector database code (like `ChromaClient`) and business logic services.

Create `data/sqlite_storage.py`:

```python
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

class SQLiteStorage:
    """
    Base class for SQLite-based storage services.
    
    Learning: Inheritance allows us to share common database connection
    management code across multiple services. This reduces duplication
    and makes it easier to maintain and extend.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_db_directory()
    
    def _ensure_db_directory(self):
        """Create directory if it doesn't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        
        Learning: Context managers ensure connections are always closed,
        even if exceptions occur. This prevents resource leaks.
        All SQLite services inherit this method, so we write it once.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
```

**Why a Base Class?**
- **DRY Principle**: Write connection management code once, reuse everywhere
- **Consistency**: All database services use the same pattern
- **Maintainability**: Fix bugs or improve connection handling in one place
- **Future-Proofing**: Easy to swap SQLite for PostgreSQL later (just change base class)

**Why `data/` directory?**
- **Storage Infrastructure**: Similar to `chroma_client.py`, this is storage infrastructure code
- **Separation**: Keeps storage abstractions (`data/`) separate from business logic (`services/`)
- **Clarity**: Distinguishes SQL storage from vector database storage (ChromaDB)
- **Organization**: All database-related code lives together in `data/`

#### Step 1.2: Implement MessageStorage Class

```python
import json
from typing import List, Dict, Optional
from datetime import datetime
from data.sqlite_storage import SQLiteStorage

class MessageStorage(SQLiteStorage):
    """
    Storage service for Discord messages.
    
    Learning: By inheriting from SQLiteStorage, we get connection management
    and directory creation for free. We only need to implement message-specific logic.
    """
    
    def __init__(self, db_path: str = "data/raw_messages/messages.db"):
        super().__init__(db_path)
        self._init_database()
    
    def _init_database(self):
        """Create tables and indexes if they don't exist"""
        with self._get_connection() as conn:
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
            except Exception as e:
                self.logger.error(f"Failed to initialize database: {e}")
                raise
    
    def save_channel_messages(self, channel_id: str, messages: List[Dict]) -> bool:
        """
        Save messages to database using batch insert.
        
        Learning: Batch operations are much faster than individual inserts.
        Uses transaction for atomicity. Tracks oldest/newest during iteration.
        """
        if not messages:
            return True
        
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Prepare batch insert
                insert_sql = """
                    INSERT OR IGNORE INTO messages (
                        message_id, channel_id, guild_id, content,
                        author_id, author_name, author_display_name,
                        channel_name, guild_name, timestamp, created_at,
                        is_bot, has_attachments, message_type, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                batch_data = []
                oldest_message = None
                newest_message = None
                
                for msg in messages:
                    msg_timestamp = msg.get('timestamp', '')
                    batch_data.append((
                        str(msg.get('id', '')),
                        str(channel_id),
                        str(msg.get('guild_id', '')),
                        msg.get('content', ''),
                        str(msg.get('author_id', '')),
                        msg.get('author', ''),
                        msg.get('author_display_name', ''),
                        msg.get('channel_name', ''),
                        msg.get('guild_name', ''),
                        msg_timestamp,
                        msg.get('created_at', datetime.now().isoformat()),
                        1 if msg.get('is_bot', False) else 0,
                        1 if msg.get('has_attachments', False) else 0,
                        msg.get('message_type', 'default'),
                        json.dumps(msg.get('metadata', {}))
                    ))
                    
                    # Track oldest/newest during same loop (efficient!)
                    if oldest_message is None or msg_timestamp < oldest_message.get('timestamp', ''):
                        oldest_message = msg
                    if newest_message is None or msg_timestamp > newest_message.get('timestamp', ''):
                        newest_message = msg
                
                # Batch insert (much faster than individual inserts)
                cursor.executemany(insert_sql, batch_data)
                
                # Get actual count after insert (accounts for INSERT OR IGNORE)
                cursor.execute("SELECT COUNT(*) FROM messages WHERE channel_id = ?", (channel_id,))
                total_messages = cursor.fetchone()[0]
                
                # Update checkpoint
                if messages:
                    self._update_checkpoint(
                        conn, channel_id, 
                        newest_message.get('id', ''), 
                        newest_message.get('timestamp', ''),
                        total_messages,
                        oldest_message.get('id', ''),
                        oldest_message.get('timestamp', ''),
                        newest_message.get('timestamp', '')
                    )
                
                conn.commit()
                self.logger.info(f"Saved {len(messages)} messages for channel {channel_id}")
                return True
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to save channel messages: {e}")
                return False
    
    def load_channel_messages(self, channel_id: str) -> List[Dict]:
        """
        Load all messages for a channel in chronological order.
        
        Learning: Loads all messages (for chunking entire history).
        Ordered by timestamp ASC (oldest first).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_id, channel_id, guild_id, content,
                    author_id, author_name, author_display_name,
                    channel_name, guild_name, timestamp, created_at,
                    is_bot, has_attachments, message_type, metadata
                FROM messages
                WHERE channel_id = ?
                ORDER BY timestamp ASC
            """, (str(channel_id),))
            
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                messages.append({
                    'id': row[0],
                    'channel_id': row[1],
                    'guild_id': row[2],
                    'content': row[3],
                    'author_id': row[4],
                    'author': row[5],
                    'author_display_name': row[6],
                    'channel_name': row[7],
                    'guild_name': row[8],
                    'timestamp': row[9],
                    'created_at': row[10],
                    'is_bot': bool(row[11]),
                    'has_attachments': bool(row[12]),
                    'message_type': row[13],
                    'metadata': json.loads(row[14]) if row[14] else {}
                })
            
            return messages
    
    def get_checkpoint(self, channel_id: str) -> Dict:
        """Get checkpoint for a channel"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT last_message_id, last_fetch_timestamp, total_messages,
                    oldest_message_id, oldest_message_timestamp, newest_message_timestamp
                FROM checkpoints
                WHERE channel_id = ?
            """, (str(channel_id),))
            row = cursor.fetchone()
            if row:
                return {
                    'last_message_id': row[0],      
                    'last_fetch_timestamp': row[1],
                    'total_messages': row[2],
                    'oldest_message_id': row[3],
                    'oldest_message_timestamp': row[4],
                    'newest_message_timestamp': row[5]
                }
            return None
    
    def _update_checkpoint(self, conn, channel_id, last_message_id, 
                          timestamp, total_messages, oldest_message_id,
                          oldest_timestamp, newest_timestamp):
        """Internal method to update checkpoint"""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO checkpoints (
                channel_id, last_message_id, last_fetch_timestamp,
                total_messages, oldest_message_id, oldest_message_timestamp,
                newest_message_timestamp, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(channel_id), str(last_message_id), timestamp,
            total_messages, str(oldest_message_id), oldest_timestamp,
            newest_timestamp, datetime.now().isoformat()
        ))
    
    def get_channel_stats(self, channel_id: str) -> Dict:
        """Get statistics for a channel"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get message count and date range
            cursor.execute("""
                SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM messages
                WHERE channel_id = ?
            """, (str(channel_id),))
            
            row = cursor.fetchone()
            checkpoint = self.get_checkpoint(channel_id)
            
            return {
                'channel_id': channel_id,
                'message_count': row[0] if row else 0,
                'oldest_timestamp': row[1] if row and row[1] else None,
                'newest_timestamp': row[2] if row and row[2] else None,
                'checkpoint': checkpoint
            }
```

**Key Learning Points:**
- **Inheritance**: Base classes provide shared functionality, reducing code duplication
- **Context Managers**: `@contextmanager` ensures connections are always closed
- **Batch Operations**: `executemany()` is much faster than individual inserts
- **Single-Pass Efficiency**: Track oldest/newest during batch collection (no redundant loops)
- **Transactions**: `conn.commit()` ensures atomicity (all or nothing)
- **Indexes**: Queries on indexed columns are fast (channel_id + timestamp)
- **ON CONFLICT IGNORE**: Prevents duplicate inserts (idempotent)
- **Simplified API**: Only internal `_update_checkpoint` method (no public wrapper needed)

#### Step 1.3: Testing

Create a test script `test_message_storage.py`:

```python
import asyncio
from services.message_storage import MessageStorage
from utils.discord_utils import format_discord_message

def test_message_storage():
    storage = MessageStorage("test_messages.db")
    
    # Create test messages
    test_messages = [
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
        }
    ]
    
    # Test save
    result = storage.save_channel_messages('test_channel', test_messages)
    assert result == True
    
    # Test retrieve (all messages, chronological order)
    messages = storage.load_channel_messages('test_channel')
    assert len(messages) == 2
    assert messages[0]['id'] == '123'  # Oldest first (chronological)
    assert messages[1]['id'] == '124'  # Newest last
    
    # Test load all
    all_messages = storage.load_channel_messages('test_channel')
    assert len(all_messages) == 2
    
    # Test checkpoint
    checkpoint = storage.get_checkpoint('test_channel')
    assert checkpoint is not None
    assert checkpoint['last_message_id'] == '124'
    assert checkpoint['total_messages'] == 2
    
    # Test idempotency (duplicate messages)
    storage.save_channel_messages('test_channel', test_messages)
    all_messages_after = storage.load_channel_messages('test_channel')
    assert len(all_messages_after) == 2  # No duplicates
    
    # Test stats
    stats = storage.get_channel_stats('test_channel')
    assert stats['message_count'] == 2
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_message_storage()
```

**What to Test:**
- Messages save correctly
- Checkpoint updates after save
- Load all messages works (oldest first, chronological)
- Duplicate messages are ignored (idempotent)
- Channel stats are accurate
- Oldest/newest tracking during batch collection

#### Step 1.4: Refactor Existing Services

Now that we have a base class, we can refactor existing services to use it. For example, `UserAITracker` can be updated to inherit from `SQLiteStorage` and use context managers instead of manual connection management.

**Before (Manual Connection Management):**
```python
class UserAITracker:
    def log_ai_usage(self, user_display_name: str, cost: float = 0.0, tokens_total: int = 0):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # ... do work ...
        conn.commit()
        conn.close()  # Easy to forget!
```

**After (Using Base Class):**
```python
from data.sqlite_storage import SQLiteStorage

class UserAITracker(SQLiteStorage):
    def __init__(self, db_path: str = "data/ai_usage.db"):
        super().__init__(db_path)
        self._init_database()
    
    def _init_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_ai_stats (
                    user_display_name TEXT PRIMARY KEY,
                    lifetime_cost REAL DEFAULT 0,
                    lifetime_tokens INTEGER DEFAULT 0,
                    lifetime_credit REAL DEFAULT 0
                )
            """)
            conn.commit()
    
    def log_ai_usage(self, user_display_name: str, cost: float = 0.0, tokens_total: int = 0):
        with self._get_connection() as conn:  # Automatic cleanup!
            cursor = conn.cursor()
            # ... do work ...
            conn.commit()
            # Connection automatically closed when leaving 'with' block
```

**Benefits of Refactoring:**
- **No Resource Leaks**: Context manager always closes connections
- **Consistent Pattern**: Same code style across all database services
- **Less Code**: No need to manually manage connections in every method
- **Easier Maintenance**: Fix connection issues in one place (base class)

**When to Create a Base Class:**
- ✅ You have 2+ classes with similar code patterns
- ✅ The shared code is substantial (not just a single line)
- ✅ You want consistency across services
- ❌ Don't create base classes for trivial duplication (over-engineering)

### Common Pitfalls - Phase 1

1. **Forgetting to close connections**: Always use context managers
2. **Not calling super().__init__()**: When inheriting, must call parent constructor
3. **Not handling empty results**: Check if `row` is None before accessing
4. **String vs int IDs**: Discord IDs are strings, not integers
5. **Timestamp ordering**: DESC for recent, ASC for chronological
6. **Missing indexes**: Queries will be slow without proper indexes

### Debugging Tips - Phase 1

- **Check database file**: Use SQLite browser to inspect data
- **Log queries**: Add logging to see what SQL is executed
- **Test with small data**: Start with 10 messages before scaling
- **Verify indexes**: Use `EXPLAIN QUERY PLAN` to check index usage

### Performance Considerations - Phase 1

- **Batch inserts**: 100x faster than individual inserts
- **Indexes**: Critical for fast queries on large datasets
- **Connection pooling**: SQLite doesn't need it, but PostgreSQL would
- **Transaction size**: Don't commit after every message (batch!)