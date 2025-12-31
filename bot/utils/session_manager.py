"""
Session manager for chatbot conversations.

Manages per-channel conversation sessions with context history,
automatic trimming, and TTL-based cleanup.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from bot.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions for chatbot channels.
    
    Features:
    - Per-channel message history (channel-level conversations)
    - Automatic context window trimming with accurate token counting
    - Session TTL management
    - Async-safe operations
    """
    
    def __init__(self, max_history: int, session_timeout: int, max_context_tokens: int = 2000):
        """
        Initialize session manager.
        
        Args:
            max_history: Maximum number of messages to keep per session
            session_timeout: Session expiry time in seconds
            max_context_tokens: Maximum tokens for context window
        """
        self.max_history = max_history
        self.session_timeout = session_timeout
        self.max_context_tokens = max_context_tokens
        self.sessions: Dict[int, Dict] = {}  # channel_id -> session
        self._locks: Dict[int, asyncio.Lock] = {}  # channel_id -> lock
        self._lock = asyncio.Lock()
    
    async def _get_lock(self, channel_id: int) -> asyncio.Lock:
        """Get or create lock for a channel."""
        async with self._lock:
            if channel_id not in self._locks:
                self._locks[channel_id] = asyncio.Lock()
            return self._locks[channel_id]
    
    async def get_session(self, channel_id: int) -> Dict:
        """
        Get or create session for channel.
        
        Args:
            channel_id: Discord channel ID
            
        Returns:
            Session dictionary
        """
        # Force cleanup if too many sessions (prevent memory issues)
        # Lowered threshold for proactive cleanup
        MAX_SESSIONS = 50  # Reduced from 500 to save memory
        async with self._lock:
            session_count = len(self.sessions)
            if session_count > MAX_SESSIONS:
                logger.warning(f"Session count ({session_count}) exceeds limit, forcing cleanup")
                # Note: We can't await here because we're holding _lock, so we'll do it after
                needs_cleanup = True
            else:
                needs_cleanup = False
        
        if needs_cleanup:
            await self.cleanup_expired_sessions()
        
        lock = await self._get_lock(channel_id)
        async with lock:
            if channel_id not in self.sessions:
                # Create new session
                self.sessions[channel_id] = {
                    "channel_id": channel_id,
                    "messages": [],
                    "created_at": datetime.now(),
                    "last_activity": datetime.now()
                }
                logger.debug(f"Created new session for channel {channel_id}")
            
            session = self.sessions[channel_id]
            session["last_activity"] = datetime.now()
            
            return session
    
    async def add_message(
        self,
        channel_id: int,
        role: str,
        content: str,
        author_id: Optional[int] = None,
        author_name: Optional[str] = None
    ):
        """
        Add message to channel's session history.
        
        Args:
            channel_id: Discord channel ID
            role: Message role ("user" or "assistant")
            content: Message content
            author_id: Optional author user ID (for user messages)
            author_name: Optional author display name (for user messages)
        """
        lock = await self._get_lock(channel_id)
        async with lock:
            if channel_id not in self.sessions:
                logger.warning(f"Attempted to add message to non-existent session for channel {channel_id}")
                return
            
            session = self.sessions[channel_id]
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add author information for user messages
            if role == "user" and author_id is not None:
                message["author_id"] = author_id
                message["author_name"] = author_name or f"User{author_id}"
            
            session["messages"].append(message)
            session["last_activity"] = datetime.now()
            
            # Trim history if exceeds max_history
            if len(session["messages"]) > self.max_history:
                session["messages"] = session["messages"][-self.max_history:]
                logger.debug(f"Trimmed history for channel {channel_id} to {self.max_history} messages")
    
    async def get_history(self, channel_id: int) -> List[Dict]:
        """
        Get formatted message history for channel.
        
        Args:
            channel_id: Discord channel ID
            
        Returns:
            List of message dictionaries
        """
        lock = await self._get_lock(channel_id)
        async with lock:
            if channel_id not in self.sessions:
                return []
            
            session = self.sessions[channel_id]
            return session.get("messages", []).copy()
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using shared tokenizer utility.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return count_tokens(text)
    
    def _trim_history_by_tokens(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Trim messages using FIFO, keeping most recent within token limit.
        
        Uses accurate token counting with tiktoken if available.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens allowed
            
        Returns:
            Trimmed list of messages
        """
        total_tokens = 0
        trimmed = []
        
        # Count backwards from most recent (keep newest messages)
        for msg in reversed(messages):
            # Format message for token counting (include author name if present)
            content = msg.get('content', '')
            author_name = msg.get('author_name')
            
            if author_name and msg.get('role') == 'user':
                # Include author name in token count
                formatted = f"{author_name}: {content}"
            else:
                formatted = content
            
            msg_tokens = self._count_tokens(formatted)
            
            if total_tokens + msg_tokens <= max_tokens:
                trimmed.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        
        return trimmed
    
    async def format_for_ai(
        self,
        channel_id: int,
        current_message: str,
        current_author_name: str,
        system_prompt: str,
        channel_context: str = ""
    ) -> str:
        """
        Format history + current message as AI prompt.
        
        Args:
            channel_id: Discord channel ID
            current_message: Current user message
            current_author_name: Display name of current message author
            system_prompt: System prompt for AI
            channel_context: Optional recent channel context
            
        Returns:
            Formatted prompt string
        """
        lock = await self._get_lock(channel_id)
        async with lock:
            # Access history directly since we already have the lock
            if channel_id not in self.sessions:
                history = []
            else:
                history = self.sessions[channel_id].get("messages", []).copy()
            
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
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    author_name = msg.get('author_name')
                    
                    if role == 'user' and author_name:
                        # Format with author name
                        parts.append(f"User ({author_name}): {content}")
                    elif role == 'assistant':
                        parts.append(f"Assistant: {content}")
                    else:
                        parts.append(f"User: {content}")
            
            # Add current message with author name
            parts.append(f"\n\nUser ({current_author_name}): {current_message}\nAssistant:")
            
            return "\n".join(parts)
    
    async def reset_session(self, channel_id: int):
        """
        Clear channel's conversation history.
        
        Args:
            channel_id: Discord channel ID
        """
        lock = await self._get_lock(channel_id)
        async with lock:
            if channel_id in self.sessions:
                self.sessions[channel_id]["messages"] = []
                self.sessions[channel_id]["created_at"] = datetime.now()
                self.sessions[channel_id]["last_activity"] = datetime.now()
                logger.debug(f"Reset session for channel {channel_id}")
    
    async def cleanup_expired_sessions(self):
        """
        Remove sessions older than timeout.
        
        Also cleans up locks for removed sessions.
        """
        async with self._lock:
            now = datetime.now()
            timeout_delta = timedelta(seconds=self.session_timeout)
            to_remove = []
            
            for channel_id, session in self.sessions.items():
                last_activity = session.get("last_activity")
                if last_activity is None:
                    # No activity recorded, remove session
                    to_remove.append(channel_id)
                    continue
                
                if isinstance(last_activity, str):
                    try:
                        last_activity = datetime.fromisoformat(last_activity)
                    except (ValueError, TypeError):
                        # Invalid format, remove session
                        to_remove.append(channel_id)
                        continue
                
                if isinstance(last_activity, datetime):
                    if now - last_activity > timeout_delta:
                        to_remove.append(channel_id)
                else:
                    # Invalid type, remove session
                    to_remove.append(channel_id)
            
            for channel_id in to_remove:
                del self.sessions[channel_id]
                if channel_id in self._locks:
                    del self._locks[channel_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} expired chatbot sessions")
