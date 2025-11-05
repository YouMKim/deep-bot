import sqlite3
import json
import logging

from pathlib import Path 
from typing import List, Dict, Optional
from datetime import datetime 
from data.sqlite_storage import SQLiteStorage 

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