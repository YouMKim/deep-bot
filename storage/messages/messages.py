import sqlite3
import json
import logging

from pathlib import Path 
from typing import List, Dict, Optional
from datetime import datetime 
from storage.sqlite_storage import SQLiteStorage 

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

CREATE TABLE IF NOT EXISTS chunk_checkpoints (
         channel_id TEXT NOT NULL,
         strategy TEXT NOT NULL,
         last_chunk_id TEXT,
         last_message_id TEXT,
         last_message_timestamp TEXT,
         updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
         PRIMARY KEY (channel_id, strategy)
     );

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_channel_timestamp 
    ON messages(channel_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_id 
    ON messages(message_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_channel 
    ON checkpoints(channel_id);
"""


class MessageStorage(SQLiteStorage):
    def __init__(self, db_path: str = "data/raw_messages/messages.db"):
        super().__init__(db_path)
        self._init_database()

    def _init_database(self):
        with self._get_connection() as conn:
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit() 
            except Exception as e:
                self.logger.error(f"Failed to initialize database: {e}") 
                raise 
    
    def save_channel_messages(self, channel_id: str, messages: List[Dict]):
        if not messages:
            return True 
        
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor() 

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

                    if oldest_message is None or msg_timestamp < oldest_message.get('timestamp', ''):
                        oldest_message = msg
                    if newest_message is None or msg_timestamp > newest_message.get('timestamp', ''):
                        newest_message = msg

                cursor.executemany(insert_sql, batch_data)
                
                # Get actual count after insert (accounts for INSERT OR IGNORE)
                cursor.execute("SELECT COUNT(*) FROM messages WHERE channel_id = ?", (channel_id,))
                total_messages = cursor.fetchone()[0]
                
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
        with self._get_connection() as conn:
            cursor = conn.cursor()

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

    def get_recent_messages(self, channel_id: str, limit: int) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Fetch in DESC order, then reverse to return oldest-to-newest
            cursor.execute("""
                SELECT message_id, channel_id, guild_id, content,
                    author_id, author_name, author_display_name,
                    channel_name, guild_name, timestamp, created_at,
                    is_bot, has_attachments, message_type, metadata
                FROM messages
                WHERE channel_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (str(channel_id), limit))
            
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                messages.append({
                    'message_id': row[0],
                    'channel_id': row[1],
                    'guild_id': row[2],
                    'content': row[3],
                    'author_id': row[4],
                    'author_name': row[5],
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
            
            # Reverse to return oldest-to-newest
            return list(reversed(messages))

    def get_oldest_messages(self, channel_id: str, limit: int) -> List[Dict]:
        """
        Get oldest messages from a channel (for starting chunking from beginning).
        
        Args:
            channel_id: Channel ID to fetch from
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries, ordered oldest to newest
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
                LIMIT ?
            """, (str(channel_id), limit))
            
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                messages.append({
                    'message_id': row[0],
                    'channel_id': row[1],
                    'guild_id': row[2],
                    'content': row[3],
                    'author_id': row[4],
                    'author_name': row[5],
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

    def get_messages_after(self, channel_id: str, message_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get messages after a specific message ID (for incremental processing).
        
        Args:
            channel_id: Channel ID to fetch from
            message_id: Message ID to start after
            limit: Maximum number of messages to return (None for all)
            
        Returns:
            List of message dictionaries, ordered oldest to newest
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # First get the timestamp of the reference message
            cursor.execute("""
                SELECT timestamp FROM messages WHERE message_id = ?
            """, (str(message_id),))
            
            row = cursor.fetchone()
            if not row:
                self.logger.warning(f"Message {message_id} not found, returning empty list")
                return []
            
            reference_timestamp = row[0]
            
            # Get messages after that timestamp
            if limit:
                cursor.execute("""
                    SELECT message_id, channel_id, guild_id, content,
                        author_id, author_name, author_display_name,
                        channel_name, guild_name, timestamp, created_at,
                        is_bot, has_attachments, message_type, metadata
                    FROM messages
                    WHERE channel_id = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (str(channel_id), reference_timestamp, limit))
            else:
                cursor.execute("""
                    SELECT message_id, channel_id, guild_id, content,
                        author_id, author_name, author_display_name,
                        channel_name, guild_name, timestamp, created_at,
                        is_bot, has_attachments, message_type, metadata
                    FROM messages
                    WHERE channel_id = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                """, (str(channel_id), reference_timestamp))
            
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                messages.append({
                    'message_id': row[0],
                    'channel_id': row[1],
                    'guild_id': row[2],
                    'content': row[3],
                    'author_id': row[4],
                    'author_name': row[5],
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

    def get_chunking_checkpoint(self, channel_id: str, strategy: str) -> Optional[Dict]:
        """
        Get the last checkpoint for a specific chunking strategy.
        
        Args:
            channel_id: Channel ID
            strategy: Chunking strategy name
            
        Returns:
            Dictionary with checkpoint data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT last_chunk_id, last_message_id, last_message_timestamp, updated_at
                FROM chunk_checkpoints
                WHERE channel_id = ? AND strategy = ?
            """, (str(channel_id), str(strategy)))
            
            row = cursor.fetchone()
            if row:
                return {
                    'last_chunk_id': row[0],
                    'last_message_id': row[1],
                    'last_message_timestamp': row[2],
                    'updated_at': row[3]
                }
            return None

    def update_chunking_checkpoint(
        self,
        channel_id: str,
        strategy: str,
        last_chunk_id: str,
        last_message_id: str,
        last_timestamp: str,
    ) -> None:
        """
        Update checkpoint after successfully processing a batch.
        
        Args:
            channel_id: Channel ID
            strategy: Chunking strategy name
            last_chunk_id: ID of the last chunk created
            last_message_id: ID of the last message processed
            last_timestamp: Timestamp of the last message processed
        """
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO chunk_checkpoints (
                        channel_id, strategy, last_chunk_id, last_message_id,
                        last_message_timestamp, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(channel_id),
                    str(strategy),
                    str(last_chunk_id),
                    str(last_message_id),
                    str(last_timestamp),
                    datetime.now().isoformat()
                ))
                conn.commit()
                self.logger.info(
                    f"Updated chunking checkpoint for {channel_id} / {strategy}: "
                    f"last_message={last_message_id}"
                )
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to update chunking checkpoint: {e}")
                raise

    def delete_chunking_checkpoint(self, channel_id: str, strategy: Optional[str] = None) -> bool:
        """
        Delete a chunking checkpoint to force re-processing from the beginning.
        
        Args:
            channel_id: Channel ID
            strategy: Chunking strategy name (or None to delete all strategies for channel)
            
        Returns:
            True if checkpoint was deleted, False if it didn't exist
        """
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                if strategy:
                    cursor.execute("""
                        DELETE FROM chunk_checkpoints
                        WHERE channel_id = ? AND strategy = ?
                    """, (str(channel_id), str(strategy)))
                    deleted = cursor.rowcount > 0
                    self.logger.info(
                        f"Deleted chunking checkpoint for {channel_id} / {strategy}"
                    )
                else:
                    cursor.execute("""
                        DELETE FROM chunk_checkpoints
                        WHERE channel_id = ?
                    """, (str(channel_id),))
                    deleted = cursor.rowcount > 0
                    self.logger.info(
                        f"Deleted all chunking checkpoints for {channel_id}"
                    )
                conn.commit()
                return deleted
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to delete chunking checkpoint: {e}")
                raise