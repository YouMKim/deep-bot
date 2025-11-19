"""
Session manager for chatbot conversations.

Manages per-user conversation sessions with context history,
automatic trimming, and TTL-based cleanup.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter with automatic cleanup.
    
    Tracks message timestamps per user and enforces rate limits
    using a sliding window approach.
    """
    
    def __init__(self, max_messages: int, window_seconds: int):
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.user_timestamps: Dict[int, deque] = {}
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, user_id: int) -> tuple[bool, float]:
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


class SessionManager:

    def __init__(self, max_history: int, session_timeout: int, max_context_tokens: int = 2000):
        self.max_history = max_history
        self.session_timeout = session_timeout
        self.max_context_tokens = max_context_tokens
        self.sessions: Dict[int, Dict] = {}
        self._locks: Dict[int, asyncio.Lock] = {}
        self._lock = asyncio.Lock()
    
    async def _get_lock(self, user_id: int) -> asyncio.Lock:
        async with self._lock:
            if user_id not in self._locks:
                self._locks[user_id] = asyncio.Lock()
            return self._locks[user_id]
    
    async def get_session(self, user_id: int, channel_id: int) -> Dict:
        """
        Get or create session for user.
        """
        lock = await self._get_lock(user_id)
        async with lock:
            if user_id not in self.sessions:
                # Create new session
                self.sessions[user_id] = {
                    "user_id": user_id,
                    "channel_id": channel_id,
                    "messages": [],
                    "created_at": datetime.now(),
                    "last_activity": datetime.now()
                }
                logger.debug(f"Created new session for user {user_id}")
            
            session = self.sessions[user_id]
            session["last_activity"] = datetime.now()
            
            return session
    
    async def add_message(self, user_id: int, role: str, content: str):
        """
        Add message to user's session history.
        """

        lock = await self._get_lock(user_id)
        async with lock:
            if user_id not in self.sessions:
                logger.warning(f"Attempted to add message to non-existent session for user {user_id}")
                return
            
            session = self.sessions[user_id]
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            session["messages"].append(message)
            session["last_activity"] = datetime.now()
            
            if len(session["messages"]) > self.max_history:
                session["messages"] = session["messages"][-self.max_history:]
                logger.debug(f"Trimmed history for user {user_id} to {self.max_history} messages")
    
    async def get_history(self, user_id: int) -> List[Dict]:
        """
        Get formatted message history for user.
        """

        lock = await self._get_lock(user_id)
        async with lock:
            if user_id not in self.sessions:
                return []
            
            session = self.sessions[user_id]
            return session.get("messages", []).copy()
    
    def _trim_history_by_tokens(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        total_tokens = 0
        trimmed = []
        
        # Count backwards from most recent (keep newest messages)
        for msg in reversed(messages):
            # Rough token estimation: ~4 characters per token
            msg_tokens = len(msg.get('content', '')) // 4
            if total_tokens + msg_tokens <= max_tokens:
                trimmed.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        
        return trimmed
    
    async def format_for_ai(self, user_id: int, current_message: str, system_prompt: str, channel_context: str = "") -> str:
        """
        Format history + current message as AI prompt.
        """
        
        lock = await self._get_lock(user_id)
        async with lock:
            # Access history directly since we already have the lock
            # Don't call get_history() as it would try to acquire the same lock again
            if user_id not in self.sessions:
                history = []
            else:
                history = self.sessions[user_id].get("messages", []).copy()
            
            # Trim history to fit token budget
            # Reserve tokens for system prompt, channel context, current message, and response
            available_tokens = self.max_context_tokens - 200  # Reserve for other parts
            trimmed_history = self._trim_history_by_tokens(history, available_tokens)
            
            # Build prompt parts
            parts = [system_prompt]
            
            # Add channel context if provided
            if channel_context:
                parts.append(f"\n\n{channel_context}")
            
            # Add conversation history
            if trimmed_history:
                parts.append("\n\nConversation history:")
                for msg in trimmed_history:
                    role_label = "User" if msg['role'] == 'user' else "Assistant"
                    parts.append(f"{role_label}: {msg['content']}")
            
            # Add current message
            parts.append(f"\n\nUser: {current_message}\nAssistant:")
            
            return "\n".join(parts)
    
    async def reset_session(self, user_id: int):
        """
        Clear user's conversation history.
        
        Args:
            user_id: Discord user ID
        """
        lock = await self._get_lock(user_id)
        async with lock:
            if user_id in self.sessions:
                self.sessions[user_id]["messages"] = []
                self.sessions[user_id]["created_at"] = datetime.now()
                self.sessions[user_id]["last_activity"] = datetime.now()
                logger.debug(f"Reset session for user {user_id}")
    
    async def cleanup_expired_sessions(self):
        """
        Remove sessions older than timeout.
        
        Also cleans up locks for removed sessions.
        """
        async with self._lock:
            now = datetime.now()
            timeout_delta = timedelta(seconds=self.session_timeout)
            to_remove = []
            
            for user_id, session in self.sessions.items():
                last_activity = session.get("last_activity")
                if last_activity is None:
                    # No activity recorded, remove session
                    to_remove.append(user_id)
                    continue
                
                if isinstance(last_activity, str):
                    try:
                        last_activity = datetime.fromisoformat(last_activity)
                    except (ValueError, TypeError):
                        # Invalid format, remove session
                        to_remove.append(user_id)
                        continue
                
                if isinstance(last_activity, datetime):
                    if now - last_activity > timeout_delta:
                        to_remove.append(user_id)
                else:
                    # Invalid type, remove session
                    to_remove.append(user_id)
            
            for user_id in to_remove:
                del self.sessions[user_id]
                if user_id in self._locks:
                    del self._locks[user_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} expired chatbot sessions")

