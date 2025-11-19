"""
Tests for Chatbot cog.

Tests cover:
- Question detection
- Message filtering
- RAG and chat response generation
- Admin commands
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime
import discord

from bot.cogs.chatbot import Chatbot
from config import Config


@pytest.fixture
def mock_bot():
    """Create a mock Discord bot"""
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 999999
    bot.wait_until_ready = AsyncMock()
    return bot


@pytest.fixture
def chatbot_cog(mock_bot):
    """Create a Chatbot cog instance with mocked dependencies"""
    with patch('bot.cogs.chatbot.SessionManager'), \
         patch('bot.cogs.chatbot.RateLimiter'), \
         patch('bot.cogs.chatbot.AIService'), \
         patch('bot.cogs.chatbot.RAGPipeline'), \
         patch('bot.cogs.chatbot.UserAITracker'):
        
        # Mock the task start methods to prevent starting tasks during init
        from discord.ext import tasks
        original_start = tasks.Loop.start
        
        def mock_start(self, *args, **kwargs):
            pass  # Don't actually start the task
        
        tasks.Loop.start = mock_start
        
        try:
            cog = Chatbot(mock_bot)
            
            # Mock the services
            cog.session_manager = MagicMock()
            cog.rate_limiter = MagicMock()
            cog.ai_service = MagicMock()
            cog.rag_pipeline = MagicMock()
            cog.ai_tracker = MagicMock()
            
            # Mock background tasks
            cog.cleanup_sessions = MagicMock()
            cog.cleanup_rate_limits = MagicMock()
            
            return cog
        finally:
            tasks.Loop.start = original_start


@pytest.fixture
def mock_message():
    """Create a mock Discord message"""
    message = MagicMock(spec=discord.Message)
    message.author = MagicMock()
    message.author.id = 12345
    message.author.display_name = "TestUser"
    message.author.bot = False
    message.author.mention = "<@12345>"
    message.channel = MagicMock()
    message.channel.id = Config.CHATBOT_CHANNEL_ID
    message.channel.send = AsyncMock()
    message.channel.typing = MagicMock()
    message.channel.typing.__aenter__ = AsyncMock()
    message.channel.typing.__aexit__ = AsyncMock(return_value=None)
    message.content = "Hello, bot!"
    message.add_reaction = AsyncMock()
    message.remove_reaction = AsyncMock()
    message.mentions = []
    return message


class TestQuestionDetection:
    """Test question detection logic"""
    
    @pytest.mark.asyncio
    async def test_question_with_question_mark(self, chatbot_cog):
        """Test detection of questions ending with ?"""
        assert await chatbot_cog._is_question("What is this?") is True
        assert await chatbot_cog._is_question("How are you?") is True
        assert await chatbot_cog._is_question("Is that correct?") is True
    
    @pytest.mark.asyncio
    async def test_question_starters(self, chatbot_cog):
        """Test detection of questions starting with question words"""
        assert await chatbot_cog._is_question("what is this") is True
        assert await chatbot_cog._is_question("when did that happen") is True
        assert await chatbot_cog._is_question("where are we going") is True
        assert await chatbot_cog._is_question("who said that") is True
        assert await chatbot_cog._is_question("why did you do that") is True
        assert await chatbot_cog._is_question("how does this work") is True
    
    @pytest.mark.asyncio
    async def test_question_phrases(self, chatbot_cog):
        """Test detection of question phrases"""
        assert await chatbot_cog._is_question("tell me about this") is True
        assert await chatbot_cog._is_question("explain how it works") is True
        assert await chatbot_cog._is_question("what about that") is True
        assert await chatbot_cog._is_question("do you know the answer") is True
    
    @pytest.mark.asyncio
    async def test_imperative_questions(self, chatbot_cog):
        """Test detection of short imperative questions"""
        assert await chatbot_cog._is_question("explain x") is True
        assert await chatbot_cog._is_question("tell me y") is True
        assert await chatbot_cog._is_question("describe z") is True
    
    @pytest.mark.asyncio
    async def test_non_questions(self, chatbot_cog):
        """Test that non-questions are not detected"""
        assert await chatbot_cog._is_question("Hello there") is False
        assert await chatbot_cog._is_question("I like this") is False
        assert await chatbot_cog._is_question("Thanks for the help") is False
        assert await chatbot_cog._is_question("This is great") is False


class TestRAGDetection:
    """Test RAG vs conversational question detection"""
    
    @pytest.mark.asyncio
    async def test_conversational_questions_no_rag(self, chatbot_cog):
        """Test that conversational questions don't need RAG"""
        assert await chatbot_cog._needs_rag("How are you?") is False
        assert await chatbot_cog._needs_rag("What's up?") is False
        assert await chatbot_cog._needs_rag("Hi there!") is False
        assert await chatbot_cog._needs_rag("Hello!") is False
        assert await chatbot_cog._needs_rag("What can you do?") is False
        assert await chatbot_cog._needs_rag("Who are you?") is False
    
    @pytest.mark.asyncio
    async def test_rag_needed_with_mentions(self, chatbot_cog):
        """Test that questions with mentions need RAG"""
        assert await chatbot_cog._needs_rag("What did Alice say?", ["Alice"]) is True
        assert await chatbot_cog._needs_rag("How are you?", ["Bob"]) is True  # Even conversational with mention
    
    @pytest.mark.asyncio
    async def test_rag_needed_temporal_references(self, chatbot_cog):
        """Test that questions with temporal references need RAG"""
        assert await chatbot_cog._needs_rag("What did we decide yesterday?") is True
        assert await chatbot_cog._needs_rag("What was discussed last week?") is True
        assert await chatbot_cog._needs_rag("Did we talk about this earlier?") is True
        assert await chatbot_cog._needs_rag("What happened before?") is True
    
    @pytest.mark.asyncio
    async def test_rag_needed_history_keywords(self, chatbot_cog):
        """Test that questions with history keywords need RAG"""
        assert await chatbot_cog._needs_rag("What did Alice say about the database?") is True
        assert await chatbot_cog._needs_rag("Who mentioned the meeting?") is True
        assert await chatbot_cog._needs_rag("What did we decide?") is True
        assert await chatbot_cog._needs_rag("What was the plan?") is True
        assert await chatbot_cog._needs_rag("Did anyone discuss this?") is True
    
    @pytest.mark.asyncio
    async def test_rag_needed_people_indicators(self, chatbot_cog):
        """Test that questions about people need RAG"""
        assert await chatbot_cog._needs_rag("What did they say?") is True
        assert await chatbot_cog._needs_rag("Who said that?") is True
        assert await chatbot_cog._needs_rag("What did someone mention?") is True
    
    @pytest.mark.asyncio
    async def test_rag_needed_decision_keywords(self, chatbot_cog):
        """Test that questions about decisions need RAG"""
        assert await chatbot_cog._needs_rag("What did we decide?") is True
        assert await chatbot_cog._needs_rag("What was the decision?") is True
        assert await chatbot_cog._needs_rag("What did we agree on?") is True
        assert await chatbot_cog._needs_rag("What was the plan?") is True
    
    @pytest.mark.asyncio
    async def test_conversational_with_history_keyword(self, chatbot_cog):
        """Test that conversational questions with history keywords still use RAG"""
        # "What did you say?" - conversational but has "say" keyword
        assert await chatbot_cog._needs_rag("What did you say?") is True
        # "How are you?" with "say" - should still be conversational
        assert await chatbot_cog._needs_rag("How are you?") is False
    
    @pytest.mark.asyncio
    async def test_general_questions_no_rag(self, chatbot_cog):
        """Test that general questions don't need RAG"""
        assert await chatbot_cog._needs_rag("What is Python?") is False
        assert await chatbot_cog._needs_rag("How does this work?") is False
        assert await chatbot_cog._needs_rag("Can you help me?") is False


class TestMessageFiltering:
    """Test message filtering logic"""
    
    @pytest.mark.asyncio
    async def test_filter_bot_messages(self, chatbot_cog, mock_message):
        """Test that bot messages are filtered"""
        mock_message.author.bot = True
        
        await chatbot_cog.on_message(mock_message)
        
        mock_message.channel.send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_filter_command_messages(self, chatbot_cog, mock_message):
        """Test that command messages are filtered"""
        mock_message.content = "!help"
        
        await chatbot_cog.on_message(mock_message)
        
        mock_message.channel.send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_filter_wrong_channel(self, chatbot_cog, mock_message):
        """Test that messages in wrong channel are filtered"""
        mock_message.channel.id = 999999  # Different channel
        
        await chatbot_cog.on_message(mock_message)
        
        mock_message.channel.send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_filter_blacklisted_users(self, chatbot_cog, mock_message):
        """Test that blacklisted users are filtered"""
        with patch.object(Config, 'is_blacklisted', return_value=True):
            await chatbot_cog.on_message(mock_message)
            mock_message.channel.send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_filter_long_messages(self, chatbot_cog, mock_message):
        """Test that messages over 2000 chars are filtered"""
        mock_message.content = "X" * 2001
        
        await chatbot_cog.on_message(mock_message)
        
        # Should send error message
        mock_message.channel.send.assert_called_once()
        call_args = mock_message.channel.send.call_args[0][0]
        assert "too long" in call_args.lower()


class TestRateLimiting:
    """Test rate limiting"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self, chatbot_cog, mock_message):
        """Test that allowed messages are processed"""
        chatbot_cog.rate_limiter.check_rate_limit = AsyncMock(return_value=(True, 0.0))
        chatbot_cog._generate_chat_response = AsyncMock(return_value={
            'content': 'Response',
            'cost': 0.01,
            'model': 'test',
            'tokens_total': 100,
            'mode': 'chat'
        })
        chatbot_cog.session_manager.get_session = AsyncMock(return_value={'messages': []})
        chatbot_cog.session_manager.add_message = AsyncMock()
        
        await chatbot_cog.on_message(mock_message)
        
        mock_message.channel.send.assert_called()
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, chatbot_cog, mock_message):
        """Test that rate limited messages are blocked"""
        chatbot_cog.rate_limiter.check_rate_limit = AsyncMock(return_value=(False, 30.0))
        
        await chatbot_cog.on_message(mock_message)
        
        # Should send rate limit message
        mock_message.channel.send.assert_called_once()
        call_args = mock_message.channel.send.call_args[0][0]
        assert "rate limit" in call_args.lower()


class TestResponseGeneration:
    """Test response generation"""
    
    @pytest.mark.asyncio
    async def test_chat_response_generation(self, chatbot_cog):
        """Test chat response generation"""
        chatbot_cog.session_manager.format_for_ai = AsyncMock(
            return_value="System prompt\n\nUser: Hello\nAssistant:"
        )
        chatbot_cog.ai_service.generate = AsyncMock(return_value={
            'content': 'Hi there!',
            'cost': 0.01,
            'model': 'test-model',
            'tokens_total': 50
        })
        chatbot_cog._get_recent_channel_context = AsyncMock(return_value="")
        
        response = await chatbot_cog._generate_chat_response(
            "Hello",
            12345,
            67890
        )
        
        assert response['content'] == 'Hi there!'
        assert response['cost'] == 0.01
        assert response['mode'] == 'chat'
    
    @pytest.mark.asyncio
    async def test_rag_response_generation(self, chatbot_cog):
        """Test RAG response generation"""
        from rag.models import RAGResult
        
        mock_result = RAGResult(
            answer="Based on the context, the answer is X",
            sources=[],
            tokens_used=100,
            cost=0.02,
            model="test-model"
        )
        
        chatbot_cog.rag_pipeline.answer_question = AsyncMock(return_value=mock_result)
        
        response = await chatbot_cog._generate_rag_response(
            "What is X?",
            12345,
            None,
            67890
        )
        
        assert response['content'] == "Based on the context, the answer is X"
        assert response['cost'] == 0.02
        assert response['mode'] == 'rag'
    
    @pytest.mark.asyncio
    async def test_rag_fallback_to_chat(self, chatbot_cog):
        """Test that RAG errors fallback to chat mode"""
        chatbot_cog.rag_pipeline.answer_question = AsyncMock(side_effect=Exception("RAG error"))
        chatbot_cog._generate_chat_response = AsyncMock(return_value={
            'content': 'Fallback response',
            'cost': 0.01,
            'model': 'test',
            'tokens_total': 50,
            'mode': 'chat'
        })
        
        response = await chatbot_cog._generate_rag_response(
            "Question?",
            12345,
            None,
            67890
        )
        
        assert response['mode'] == 'chat'
        assert response['content'] == 'Fallback response'


class TestMentionExtraction:
    """Test mention extraction"""
    
    def test_extract_mentions(self, chatbot_cog, mock_message):
        """Test extracting mentions from message"""
        # Create mock mentions
        user1 = MagicMock()
        user1.display_name = "Alice"
        user1.bot = False
        
        user2 = MagicMock()
        user2.display_name = "Bob"
        user2.bot = False
        
        bot_user = MagicMock()
        bot_user.display_name = "Bot"
        bot_user.bot = True
        
        mock_message.mentions = [user1, user2, bot_user]
        
        mentions = chatbot_cog._extract_mentions(mock_message)
        
        assert "Alice" in mentions
        assert "Bob" in mentions
        assert "Bot" not in mentions  # Bots should be filtered


class TestAdminCommands:
    """Test admin commands"""
    
    @pytest.mark.asyncio
    async def test_reset_conversation(self, chatbot_cog):
        """Test reset conversation command"""
        ctx = MagicMock()
        ctx.author.id = 12345
        ctx.author.mention = "<@12345>"
        ctx.send = AsyncMock()
        
        chatbot_cog.session_manager.reset_session = AsyncMock()
        
        await chatbot_cog.reset_conversation(ctx)
        
        chatbot_cog.session_manager.reset_session.assert_called_once_with(12345)
        ctx.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chatbot_stats(self, chatbot_cog):
        """Test chatbot stats command"""
        ctx = MagicMock()
        ctx.author.id = 12345
        ctx.author.display_name = "TestUser"
        ctx.channel.id = 67890
        ctx.send = AsyncMock()
        
        chatbot_cog.session_manager.get_session = AsyncMock(return_value={
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        })
        chatbot_cog.ai_tracker.get_user_stats = Mock(return_value={
            'lifetime_cost': 0.05,
            'lifetime_tokens': 500,
            'lifetime_credit': 150.0
        })
        
        await chatbot_cog.chatbot_stats(ctx)
        
        ctx.send.assert_called_once()
        # Check that embed was sent
        call_args = ctx.send.call_args
        assert 'embed' in call_args.kwargs or isinstance(call_args[0][0], discord.Embed)
    
    @pytest.mark.asyncio
    async def test_chatbot_mode_admin_only(self, chatbot_cog):
        """Test that chatbot_mode is admin-only"""
        ctx = MagicMock()
        ctx.author.id = 99999  # Not admin
        ctx.send = AsyncMock()
        
        with patch.object(Config, 'BOT_OWNER_ID', '12345'):
            await chatbot_cog.chatbot_mode(ctx)
            ctx.send.assert_called_once()
            assert "admin-only" in ctx.send.call_args[0][0].lower()
    
    @pytest.mark.asyncio
    async def test_chatbot_mode_admin_access(self, chatbot_cog):
        """Test that admin can access chatbot_mode"""
        ctx = MagicMock()
        ctx.author.id = 12345
        ctx.send = AsyncMock()
        
        with patch.object(Config, 'BOT_OWNER_ID', '12345'):
            await chatbot_cog.chatbot_mode(ctx)
            ctx.send.assert_called_once()
            # Should send embed with config
            call_args = ctx.send.call_args
            assert 'embed' in call_args.kwargs or isinstance(call_args[0][0], discord.Embed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

