"""
Social Credit Manager for behavior-based scoring system.

Manages user social credit scores, applies penalties/rewards, and provides
tone tier mapping for AI response modification.
"""

import sqlite3
import logging
import random
import asyncio
from typing import Optional, Dict, List
from contextlib import contextmanager
from storage.sqlite_storage import SQLiteStorage
from config import Config
from .modifiers import get_tone_tier

logger = logging.getLogger(__name__)


class SocialCreditManager(SQLiteStorage):
    """Manages social credit scores for users."""
    
    # Score bounds
    MIN_SCORE = -1000
    MAX_SCORE = 1000
    
    # Penalty types and their values
    PENALTY_TYPES = {
        "unauthorized_admin_command": Config.SOCIAL_CREDIT_PENALTY_ADMIN_COMMAND,
        "query_filter_violation": Config.SOCIAL_CREDIT_PENALTY_QUERY_FILTER,
    }
    
    def __init__(self, db_path: str = "data/ai_usage.db"):
        super().__init__(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables for social credit system."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create user_ai_stats table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_ai_stats (
                    user_id TEXT PRIMARY KEY,
                    user_display_name TEXT NOT NULL,
                    social_credit_score INTEGER DEFAULT 0,
                    last_interaction TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add columns if they don't exist (for migration)
            try:
                cursor.execute("ALTER TABLE user_ai_stats ADD COLUMN user_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE user_ai_stats ADD COLUMN social_credit_score INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE user_ai_stats ADD COLUMN last_interaction TIMESTAMP")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE user_ai_stats ADD COLUMN created_at TIMESTAMP")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE user_ai_stats ADD COLUMN updated_at TIMESTAMP")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Create social_credit_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS social_credit_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_display_name TEXT NOT NULL,
                    score_before INTEGER NOT NULL,
                    score_after INTEGER NOT NULL,
                    delta INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON user_ai_stats(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_user_id ON social_credit_history(user_id)
            """)
            
            conn.commit()
    
    async def get_or_initialize_score(self, user_id: str, display_name: str) -> int:
        """
        Get user's social credit score, initializing if needed.
        
        Args:
            user_id: Discord user ID
            display_name: User's display name
            
        Returns:
            User's social credit score
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_or_initialize_score_sync, user_id, display_name
        )
    
    def _get_or_initialize_score_sync(self, user_id: str, display_name: str) -> int:
        """Synchronous helper for get_or_initialize_score."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Try to get existing score by user_id
            cursor.execute(
                "SELECT social_credit_score, user_display_name FROM user_ai_stats WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                score, old_display_name = result
                # Update display name if it changed
                if old_display_name != display_name:
                    cursor.execute(
                        "UPDATE user_ai_stats SET user_display_name = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                        (display_name, user_id)
                    )
                    conn.commit()
                return score
            
            # Try to get by display_name (backwards compatibility)
            cursor.execute(
                "SELECT social_credit_score, user_id FROM user_ai_stats WHERE user_display_name = ?",
                (display_name,)
            )
            result_by_name = cursor.fetchone()
            
            if result_by_name:
                score_by_name, existing_user_id = result_by_name
                # Update user_id if missing
                if not existing_user_id:
                    cursor.execute(
                        "UPDATE user_ai_stats SET user_id = ?, updated_at = CURRENT_TIMESTAMP WHERE user_display_name = ?",
                        (user_id, display_name)
                    )
                    conn.commit()
                    logger.info(f"Updated user_id for existing user {display_name} to {user_id}")
                return score_by_name
            
            # Initialize new user with random score
            initial_score = int(random.gauss(Config.SOCIAL_CREDIT_INITIAL_MEAN, Config.SOCIAL_CREDIT_INITIAL_STD))
            initial_score = max(self.MIN_SCORE, min(self.MAX_SCORE, initial_score))
            
            try:
                cursor.execute(
                    "INSERT INTO user_ai_stats (user_id, user_display_name, social_credit_score, created_at, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                    (user_id, display_name, initial_score)
                )
                cursor.execute(
                    "INSERT INTO social_credit_history (user_id, user_display_name, score_before, score_after, delta, reason) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, display_name, 0, initial_score, initial_score, "initial_score")
                )
                conn.commit()
                logger.info(f"Initialized user {display_name} (id: {user_id}) with score: {initial_score}")
                return initial_score
            except sqlite3.IntegrityError as e:
                # Race condition: another thread inserted this user between our check and insert
                # Retry by fetching the existing record
                conn.rollback()
                cursor.execute(
                    "SELECT social_credit_score FROM user_ai_stats WHERE user_id = ? OR user_display_name = ?",
                    (user_id, display_name)
                )
                result = cursor.fetchone()
                if result:
                    logger.info(f"User {display_name} (id: {user_id}) was initialized by another thread, returning existing score: {result[0]}")
                    return result[0]
                # If still not found, re-raise the error
                raise
    
    async def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """
        Get user's social credit statistics.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            Dictionary with user stats or None if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_user_stats_sync, user_id)
    
    def _get_user_stats_sync(self, user_id: str) -> Optional[Dict]:
        """Synchronous helper for get_user_stats."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_display_name, social_credit_score, last_interaction, created_at, user_id
                FROM user_ai_stats
                WHERE user_id = ? OR user_display_name = ?
            """, (user_id, user_id))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            display_name, score, last_interaction, created_at, db_user_id = result
            
            # Update user_id if missing
            if not db_user_id and user_id:
                cursor.execute(
                    "UPDATE user_ai_stats SET user_id = ?, updated_at = CURRENT_TIMESTAMP WHERE user_display_name = ?",
                    (user_id, display_name)
                )
                conn.commit()
                logger.info(f"Updated user_id for existing user {display_name} to {user_id}")
            
            return {
                "user_id": user_id,
                "user_display_name": display_name,
                "social_credit_score": score,
                "last_interaction": last_interaction,
                "created_at": created_at,
                "tone_tier": get_tone_tier(score)
            }
    
    async def get_user_stats_by_display_name(self, display_name: str) -> Optional[Dict]:
        """
        Get user stats by display name.
        
        Args:
            display_name: User's display name
            
        Returns:
            Dictionary with user stats or None if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_user_stats_by_display_name_sync, display_name)
    
    def _get_user_stats_by_display_name_sync(self, display_name: str) -> Optional[Dict]:
        """Synchronous helper for get_user_stats_by_display_name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, user_display_name, social_credit_score, last_interaction, created_at
                FROM user_ai_stats
                WHERE user_display_name = ?
            """, (display_name,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            user_id, db_display_name, score, last_interaction, created_at = result
            return {
                "user_id": user_id,
                "user_display_name": db_display_name,
                "social_credit_score": score,
                "last_interaction": last_interaction,
                "created_at": created_at,
                "tone_tier": get_tone_tier(score)
            }
    
    def get_tone_tier(self, score: int) -> str:
        """
        Get tone tier name for a given score.
        
        Args:
            score: Social credit score
            
        Returns:
            Tier name string
        """
        return get_tone_tier(score)
    
    async def decay_score_on_usage(self, user_id: str) -> int:
        """
        Apply decay/growth based on usage.
        Negative scores decay further, positive scores grow.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            New score after decay/growth
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._decay_score_on_usage_sync, user_id)
    
    def _decay_score_on_usage_sync(self, user_id: str) -> int:
        """Synchronous helper for decay_score_on_usage."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT social_credit_score FROM user_ai_stats WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return 0
            
            current_score = result[0]
            
            # Apply decay for negative scores, growth for positive
            if current_score < 0:
                delta = Config.SOCIAL_CREDIT_DECAY_NEGATIVE
            else:
                delta = Config.SOCIAL_CREDIT_GROWTH_POSITIVE
            
            new_score = max(self.MIN_SCORE, min(self.MAX_SCORE, current_score + delta))
            
            cursor.execute(
                "UPDATE user_ai_stats SET social_credit_score = ?, last_interaction = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (new_score, user_id)
            )
            cursor.execute(
                "INSERT INTO social_credit_history (user_id, user_display_name, score_before, score_after, delta, reason) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, "System", current_score, new_score, delta, "usage_decay_growth")
            )
            conn.commit()
            
            return new_score
    
    async def apply_penalty(self, user_id: str, penalty_type: str, display_name: str = None) -> int:
        """
        Apply a penalty to user's score.
        
        Args:
            user_id: Discord user ID
            penalty_type: Type of penalty (must be in PENALTY_TYPES)
            display_name: User's display name (optional, for logging)
            
        Returns:
            New score after penalty
        """
        if penalty_type not in self.PENALTY_TYPES:
            logger.warning(f"Unknown penalty type: {penalty_type}")
            return await self.get_or_initialize_score(user_id, display_name or "Unknown")
        
        penalty_value = self.PENALTY_TYPES[penalty_type]
        return await self.update_score(user_id, penalty_value, f"penalty_{penalty_type}", display_name)
    
    async def update_score(self, user_id: str, delta: int, reason: str, display_name: str = None) -> int:
        """
        Update user's score by a delta amount.
        
        Args:
            user_id: Discord user ID
            delta: Amount to change score by (can be negative)
            reason: Reason for the change
            display_name: User's display name (optional)
            
        Returns:
            New score after update
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._update_score_sync, user_id, delta, reason, display_name
        )
    
    def _update_score_sync(self, user_id: str, delta: int, reason: str, display_name: str = None) -> int:
        """Synchronous helper for update_score."""
        # Ensure user exists
        try:
            if display_name:
                self._get_or_initialize_score_sync(user_id, display_name)
            else:
                # Get display name from database
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT user_display_name FROM user_ai_stats WHERE user_id = ?",
                        (user_id,)
                    )
                    result = cursor.fetchone()
                    if not result:
                        # Initialize with default name
                        display_name = "Unknown"
                        self._get_or_initialize_score_sync(user_id, display_name)
                    else:
                        display_name = result[0]
        except sqlite3.IntegrityError as e:
            # Race condition: user was initialized by another thread
            # Fetch the existing display_name
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_display_name FROM user_ai_stats WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                if result:
                    display_name = result[0] or display_name or "Unknown"
                else:
                    # If still not found, use provided display_name or default
                    display_name = display_name or "Unknown"
                    logger.warning(f"Failed to initialize user {user_id}, using display_name: {display_name}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT social_credit_score FROM user_ai_stats WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                # Should not happen after initialization, but handle it
                current_score = 0
            else:
                current_score = result[0]
            
            new_score = max(self.MIN_SCORE, min(self.MAX_SCORE, current_score + delta))
            
            # Update score
            # Check if display_name already exists for a different user before updating
            if display_name:
                cursor.execute(
                    "SELECT user_id FROM user_ai_stats WHERE user_display_name = ? AND user_id != ?",
                    (display_name, user_id)
                )
                existing_user = cursor.fetchone()
                if existing_user:
                    # Display name already exists for another user, don't update it
                    # Just update the score
                    cursor.execute(
                        "UPDATE user_ai_stats SET social_credit_score = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                        (new_score, user_id)
                    )
                else:
                    # Safe to update display_name
                    cursor.execute(
                        "UPDATE user_ai_stats SET social_credit_score = ?, user_display_name = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                        (new_score, display_name, user_id)
                    )
            else:
                # Just update score
                cursor.execute(
                    "UPDATE user_ai_stats SET social_credit_score = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (new_score, user_id)
                )
            
            # Log to history
            cursor.execute(
                "INSERT INTO social_credit_history (user_id, user_display_name, score_before, score_after, delta, reason) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, display_name, current_score, new_score, delta, reason)
            )
            conn.commit()
            
            logger.info(f"Updated score for {display_name} ({user_id}): {current_score} -> {new_score} (delta: {delta}, reason: {reason})")
            return new_score
    
    async def get_leaderboard(self, limit: int = 10, ascending: bool = False) -> List[Dict]:
        """
        Get leaderboard of users by social credit score.
        
        Args:
            limit: Number of users to return
            ascending: If True, return lowest scores first
            
        Returns:
            List of user dictionaries with stats
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_leaderboard_sync, limit, ascending)
    
    def _get_leaderboard_sync(self, limit: int, ascending: bool) -> List[Dict]:
        """Synchronous helper for get_leaderboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            order = "ASC" if ascending else "DESC"
            cursor.execute(f"""
                SELECT user_id, user_display_name, social_credit_score, last_interaction
                FROM user_ai_stats
                ORDER BY social_credit_score {order}
                LIMIT ?
            """, (limit,))
            
            results = cursor.fetchall()
            return [
                {
                    "user_id": row[0],
                    "user_display_name": row[1],
                    "social_credit_score": row[2],
                    "last_interaction": row[3],
                    "tone_tier": get_tone_tier(row[2])
                }
                for row in results
            ]
