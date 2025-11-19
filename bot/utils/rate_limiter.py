"""
Rate limiter utility for sliding window rate limiting.

Provides reusable rate limiting functionality for any feature that needs
to limit user actions per time window.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict
from collections import deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter with automatic cleanup.
    
    Tracks message timestamps per user and enforces rate limits
    using a sliding window approach.
    """
    
    def __init__(self, max_messages: int, window_seconds: int):
        """
        Initialize rate limiter.
        
        Args:
            max_messages: Maximum number of messages allowed in window
            window_seconds: Time window in seconds
        """
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.user_timestamps: Dict[int, deque] = {}
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, user_id: int) -> tuple[bool, float]:
        """
        Check if user has exceeded rate limit.
        
        Args:
            user_id: Discord user ID
            
        Returns:
            Tuple of (allowed: bool, retry_after: float)
            - allowed: True if request is allowed, False if rate limited
            - retry_after: Seconds until next request is allowed (0 if allowed)
        """
        async with self._lock:
            now = datetime.now()
            
            # Initialize deque for user if needed
            if user_id not in self.user_timestamps:
                self.user_timestamps[user_id] = deque()
            
            timestamps = self.user_timestamps[user_id]
            
            # Remove old timestamps outside window
            while timestamps and (now - timestamps[0]).total_seconds() > self.window_seconds:
                timestamps.popleft()
            
            # Check if limit exceeded
            if len(timestamps) >= self.max_messages:
                # Calculate retry after time
                oldest_timestamp = timestamps[0]
                retry_after = self.window_seconds - (now - oldest_timestamp).total_seconds()
                retry_after = max(0.0, retry_after)
                return False, retry_after
            
            # Add current timestamp
            timestamps.append(now)
            return True, 0.0
    
    async def cleanup_old_entries(self):
        """
        Remove users with no recent activity to free memory.
        
        Removes users who haven't sent messages in 2x the window duration.
        """
        async with self._lock:
            now = datetime.now()
            cutoff_time = timedelta(seconds=self.window_seconds * 2)
            to_remove = []
            
            for user_id, timestamps in self.user_timestamps.items():
                if not timestamps:
                    to_remove.append(user_id)
                elif now - timestamps[-1] > cutoff_time:
                    to_remove.append(user_id)
            
            for user_id in to_remove:
                del self.user_timestamps[user_id]
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} inactive rate limit entries")

