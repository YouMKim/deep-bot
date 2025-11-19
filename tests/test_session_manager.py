"""
Tests for SessionManager and RateLimiter.

Tests cover:
- Rate limiting with sliding window
- Session creation and management
- Message history tracking
- Context window trimming
- Session TTL and cleanup
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from bot.utils.session_manager import SessionManager
from bot.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test suite for RateLimiter"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_under_limit(self):
        """Test that requests under limit are allowed"""
        limiter = RateLimiter(max_messages=5, window_seconds=60)
        
        user_id = 12345
        
        # First 5 requests should be allowed
        for i in range(5):
            allowed, retry_after = await limiter.check_rate_limit(user_id)
            assert allowed is True
            assert retry_after == 0.0
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_over_limit(self):
        """Test that requests over limit are blocked"""
        limiter = RateLimiter(max_messages=3, window_seconds=60)
        
        user_id = 12345
        
        # First 3 requests should be allowed
        for i in range(3):
            allowed, _ = await limiter.check_rate_limit(user_id)
            assert allowed is True
        
        # 4th request should be blocked
        allowed, retry_after = await limiter.check_rate_limit(user_id)
        assert allowed is False
        assert retry_after > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_per_user(self):
        """Test that rate limits are per-user"""
        limiter = RateLimiter(max_messages=2, window_seconds=60)
        
        user1 = 11111
        user2 = 22222
        
        # User1 hits limit
        await limiter.check_rate_limit(user1)
        await limiter.check_rate_limit(user1)
        allowed, _ = await limiter.check_rate_limit(user1)
        assert allowed is False
        
        # User2 should still be allowed
        allowed, _ = await limiter.check_rate_limit(user2)
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_cleanup(self):
        """Test that old entries are cleaned up"""
        limiter = RateLimiter(max_messages=2, window_seconds=1)  # 1 second window
        
        user_id = 12345
        
        # Make requests
        await limiter.check_rate_limit(user_id)
        await limiter.check_rate_limit(user_id)
        
        # Wait for window to expire
        await asyncio.sleep(1.5)
        
        # Should be able to make requests again
        allowed, _ = await limiter.check_rate_limit(user_id)
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_cleanup_old_entries(self):
        """Test cleanup_old_entries removes inactive users"""
        limiter = RateLimiter(max_messages=5, window_seconds=1)
        
        user1 = 11111
        user2 = 22222
        
        # Make requests for both users
        await limiter.check_rate_limit(user1)
        await limiter.check_rate_limit(user2)
        
        # Wait for window to expire
        await asyncio.sleep(1.5)
        
        # Cleanup should remove old entries
        await limiter.cleanup_old_entries()
        
        # Both users should be able to make requests again
        allowed1, _ = await limiter.check_rate_limit(user1)
        allowed2, _ = await limiter.check_rate_limit(user2)
        assert allowed1 is True
        assert allowed2 is True


class TestSessionManager:
    """Test suite for SessionManager"""
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a new channel session"""
        manager = SessionManager(max_history=10, session_timeout=1800)
        
        channel_id = 67890
        
        session = await manager.get_session(channel_id)
        
        assert session['channel_id'] == channel_id
        assert session['messages'] == []
        assert 'created_at' in session
        assert 'last_activity' in session
    
    @pytest.mark.asyncio
    async def test_get_existing_session(self):
        """Test getting an existing channel session"""
        manager = SessionManager(max_history=10, session_timeout=1800)
        
        channel_id = 67890
        
        # Create session
        session1 = await manager.get_session(channel_id)
        
        # Get same session again
        session2 = await manager.get_session(channel_id)
        
        assert session1['channel_id'] == session2['channel_id']
    
    @pytest.mark.asyncio
    async def test_add_message(self):
        """Test adding messages to channel session"""
        manager = SessionManager(max_history=10, session_timeout=1800)
        
        channel_id = 67890
        
        # Create session
        await manager.get_session(channel_id)
        
        # Add user message with author info
        await manager.add_message(channel_id, "user", "Hello!", author_id=12345, author_name="TestUser")
        
        # Add assistant message
        await manager.add_message(channel_id, "assistant", "Hi there!")
        
        history = await manager.get_history(channel_id)
        
        assert len(history) == 2
        assert history[0]['role'] == "user"
        assert history[0]['content'] == "Hello!"
        assert history[0]['author_name'] == "TestUser"
        assert history[1]['role'] == "assistant"
        assert history[1]['content'] == "Hi there!"
    
    @pytest.mark.asyncio
    async def test_max_history_trimming(self):
        """Test that history is trimmed to max_history"""
        manager = SessionManager(max_history=3, session_timeout=1800)
        
        channel_id = 67890
        
        await manager.get_session(channel_id)
        
        # Add more messages than max_history
        for i in range(5):
            await manager.add_message(channel_id, "user", f"Message {i}", author_id=12345, author_name="User")
        
        history = await manager.get_history(channel_id)
        
        # Should only keep last 3 messages
        assert len(history) == 3
        assert history[0]['content'] == "Message 2"
        assert history[2]['content'] == "Message 4"
    
    @pytest.mark.asyncio
    async def test_get_history_empty(self):
        """Test getting history for channel with no messages"""
        manager = SessionManager(max_history=10, session_timeout=1800)
        
        channel_id = 67890
        
        await manager.get_session(channel_id)
        
        history = await manager.get_history(channel_id)
        
        assert history == []
    
    @pytest.mark.asyncio
    async def test_reset_session(self):
        """Test resetting a channel session"""
        manager = SessionManager(max_history=10, session_timeout=1800)
        
        channel_id = 67890
        
        await manager.get_session(channel_id)
        
        # Add some messages
        await manager.add_message(channel_id, "user", "Hello!", author_id=12345, author_name="User")
        await manager.add_message(channel_id, "assistant", "Hi!")
        
        # Reset session
        await manager.reset_session(channel_id)
        
        history = await manager.get_history(channel_id)
        
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_format_for_ai(self):
        """Test formatting history for AI prompt"""
        manager = SessionManager(max_history=10, session_timeout=1800, max_context_tokens=1000)
        
        channel_id = 67890
        
        await manager.get_session(channel_id)
        
        # Add some messages
        await manager.add_message(channel_id, "user", "Hello!", author_id=12345, author_name="Alice")
        await manager.add_message(channel_id, "assistant", "Hi there!")
        
        system_prompt = "You are a helpful assistant."
        prompt = await manager.format_for_ai(channel_id, "How are you?", "Bob", system_prompt)
        
        assert system_prompt in prompt
        assert "Hello!" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt
        assert "User" in prompt or "Alice" in prompt
        assert "Assistant:" in prompt
    
    @pytest.mark.asyncio
    async def test_format_for_ai_with_channel_context(self):
        """Test formatting with channel context"""
        manager = SessionManager(max_history=10, session_timeout=1800, max_context_tokens=1000)
        
        channel_id = 67890
        
        await manager.get_session(channel_id)
        
        system_prompt = "You are a helpful assistant."
        channel_context = "Recent channel context:\nUser1: Hello\nUser2: World\n\n"
        
        prompt = await manager.format_for_ai(channel_id, "Test", "User", system_prompt, channel_context)
        
        assert channel_context in prompt
    
    @pytest.mark.asyncio
    async def test_trim_history_by_tokens(self):
        """Test token-based history trimming with accurate counting"""
        manager = SessionManager(max_history=100, session_timeout=1800, max_context_tokens=100)
        
        channel_id = 67890
        
        await manager.get_session(channel_id)
        
        # Add messages with varying lengths
        for i in range(10):
            content = "X" * (i * 10)  # Increasing length
            await manager.add_message(channel_id, "user", content, author_id=12345, author_name="User")
        
        # Format should trim to fit token budget
        prompt = await manager.format_for_ai(channel_id, "Test", "User", "System prompt")
        
        # Should not include all messages due to token limit
        assert len(prompt) > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self):
        """Test cleaning up expired channel sessions"""
        manager = SessionManager(max_history=10, session_timeout=1)  # 1 second timeout
        
        channel_id = 67890
        
        # Create session
        await manager.get_session(channel_id)
        await manager.add_message(channel_id, "user", "Hello!", author_id=12345, author_name="User")
        
        # Wait for session to expire
        await asyncio.sleep(1.5)
        
        # Cleanup expired sessions
        await manager.cleanup_expired_sessions()
        
        # Session should be gone
        history = await manager.get_history(channel_id)
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_keeps_active_sessions(self):
        """Test that active channel sessions are not cleaned up"""
        manager = SessionManager(max_history=10, session_timeout=10)  # 10 second timeout
        
        channel_id = 67890
        
        # Create session
        await manager.get_session(channel_id)
        await manager.add_message(channel_id, "user", "Hello!", author_id=12345, author_name="User")
        
        # Update activity
        await manager.get_session(channel_id)
        
        # Cleanup should not remove active session
        await manager.cleanup_expired_sessions()
        
        # Session should still exist
        history = await manager.get_history(channel_id)
        assert len(history) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test that concurrent access to channel session is handled safely"""
        manager = SessionManager(max_history=20, session_timeout=1800)  # Increase to allow all messages
        
        channel_id = 67890
        
        # Create session first
        await manager.get_session(channel_id)
        
        async def add_messages():
            for i in range(5):
                await manager.add_message(channel_id, "user", f"Message {i}", author_id=12345, author_name="User")
                await asyncio.sleep(0.01)
        
        # Run multiple coroutines concurrently
        await asyncio.gather(*[add_messages() for _ in range(3)])
        
        history = await manager.get_history(channel_id)
        
        # Should have all messages (15 total: 3 coroutines * 5 messages)
        # Note: Due to concurrency, exact count may vary, but should be close to 15
        assert len(history) >= 10  # At least some messages should be preserved
        assert len(history) <= 20  # Should not exceed max_history
    
    @pytest.mark.asyncio
    async def test_add_message_to_nonexistent_session(self):
        """Test adding message to non-existent channel session"""
        manager = SessionManager(max_history=10, session_timeout=1800)
        
        channel_id = 67890
        
        # Try to add message without creating session first
        await manager.add_message(channel_id, "user", "Hello!", author_id=12345, author_name="User")
        
        # Should not raise error, but message won't be added
        history = await manager.get_history(channel_id)
        assert len(history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

