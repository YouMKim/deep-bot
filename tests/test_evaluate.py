"""
Tests for the !evaluate command in Basic cog.

Tests cover:
- Helper methods (rating extraction, URL extraction, color coding)
- Command validation (reply requirement)
- Error handling (deleted message, empty message, API errors)
- Rate limiting
- Successful evaluation flow
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import discord
from discord.ext import commands

from bot.cogs.basic import Basic
from config import Config


@pytest.fixture
def mock_bot():
    """Create a mock Discord bot"""
    bot = MagicMock(spec=commands.Bot)
    bot.command_prefix = "!"
    return bot


@pytest.fixture
def basic_cog(mock_bot):
    """Create a Basic cog instance with mocked dependencies"""
    with patch('bot.cogs.basic.AIService'), \
         patch('bot.cogs.basic.UserAITracker'), \
         patch('bot.cogs.basic.Config'):
        
        cog = Basic(mock_bot)
        
        # Mock the AI services
        cog.ai_service = MagicMock()
        cog.evaluate_ai_service = MagicMock()
        cog.ai_tracker = MagicMock()
        cog.config = Config
        
        return cog


@pytest.fixture
def mock_ctx(mock_bot):
    """Create a mock Discord context"""
    ctx = MagicMock(spec=commands.Context)
    ctx.bot = mock_bot
    ctx.author = MagicMock()
    ctx.author.id = 12345
    ctx.author.display_name = "TestUser"
    ctx.channel = MagicMock(spec=discord.TextChannel)
    ctx.channel.id = 987654321
    ctx.channel.fetch_message = AsyncMock()
    ctx.send = AsyncMock()
    ctx.message = MagicMock()
    ctx.message.reference = None  # Default: no reply
    ctx.message.content = "!evaluate"
    return ctx


@pytest.fixture
def mock_replied_message():
    """Create a mock replied message"""
    message = MagicMock(spec=discord.Message)
    message.id = 11111
    message.content = "The Earth is flat."
    message.author = MagicMock()
    message.author.display_name = "ClaimAuthor"
    return message


class TestHelperMethods:
    """Test helper methods for evaluate command"""
    
    def test_extract_ratings_valid(self, basic_cog):
        """Test rating extraction with valid format"""
        text = """
        This is my analysis.
        TRUTHFULNESS: 75%
        EVIDENCE_ALIGNMENT: 80%
        """
        truthfulness, evidence = basic_cog._extract_ratings(text)
        assert truthfulness == 75
        assert evidence == 80
    
    def test_extract_ratings_case_insensitive(self, basic_cog):
        """Test rating extraction is case insensitive"""
        text = """
        truthfulness: 50%
        evidence_alignment: 60%
        """
        truthfulness, evidence = basic_cog._extract_ratings(text)
        assert truthfulness == 50
        assert evidence == 60
    
    def test_extract_ratings_missing(self, basic_cog):
        """Test rating extraction when ratings are missing"""
        text = "This is just analysis without ratings."
        truthfulness, evidence = basic_cog._extract_ratings(text)
        assert truthfulness is None
        assert evidence is None
    
    def test_extract_ratings_partial(self, basic_cog):
        """Test rating extraction with only one rating"""
        text = "TRUTHFULNESS: 90%"
        truthfulness, evidence = basic_cog._extract_ratings(text)
        assert truthfulness == 90
        assert evidence is None
    
    def test_extract_urls_plain(self, basic_cog):
        """Test URL extraction from plain URLs"""
        text = "Check out https://example.com and https://test.org for more info."
        urls = basic_cog._extract_urls(text)
        assert "https://example.com" in urls
        assert "https://test.org" in urls
    
    def test_extract_urls_markdown(self, basic_cog):
        """Test URL extraction from markdown links"""
        text = "See [this article](https://example.com/article) for details."
        urls = basic_cog._extract_urls(text)
        assert "https://example.com/article" in urls
    
    def test_extract_urls_mixed(self, basic_cog):
        """Test URL extraction with both plain and markdown URLs"""
        text = """
        Check https://plain.com and [this link](https://markdown.com).
        Also see https://another.com
        """
        urls = basic_cog._extract_urls(text)
        assert "https://plain.com" in urls
        assert "https://markdown.com" in urls
        assert "https://another.com" in urls
        # Should not duplicate markdown URLs
        assert urls.count("https://markdown.com") == 1
    
    def test_extract_urls_no_urls(self, basic_cog):
        """Test URL extraction when no URLs are present"""
        text = "This is just text without any URLs."
        urls = basic_cog._extract_urls(text)
        assert urls == []
    
    def test_extract_urls_limits_to_10(self, basic_cog):
        """Test URL extraction limits to 10 URLs"""
        text = " ".join([f"https://example{i}.com" for i in range(15)])
        urls = basic_cog._extract_urls(text)
        assert len(urls) == 10
    
    def test_get_rating_color_high(self, basic_cog):
        """Test color coding for high ratings"""
        color = basic_cog._get_rating_color(85)
        assert color == discord.Color.green()
    
    def test_get_rating_color_medium(self, basic_cog):
        """Test color coding for medium ratings"""
        color = basic_cog._get_rating_color(50)
        assert color == discord.Color.gold()
    
    def test_get_rating_color_low(self, basic_cog):
        """Test color coding for low ratings"""
        color = basic_cog._get_rating_color(30)
        assert color == discord.Color.red()
    
    def test_get_rating_color_boundary(self, basic_cog):
        """Test color coding at boundaries"""
        assert basic_cog._get_rating_color(70) == discord.Color.green()
        assert basic_cog._get_rating_color(69) == discord.Color.gold()
        assert basic_cog._get_rating_color(40) == discord.Color.gold()
        assert basic_cog._get_rating_color(39) == discord.Color.red()


class TestEvaluateCommand:
    """Test the evaluate command execution"""
    
    @pytest.mark.asyncio
    async def test_evaluate_no_reply(self, basic_cog, mock_ctx):
        """Test evaluate command fails when not used as reply"""
        mock_ctx.message.reference = None
        
        await basic_cog.evaluate.callback(basic_cog, mock_ctx)
        
        mock_ctx.send.assert_called_once()
        call_args = mock_ctx.send.call_args[0][0]
        assert "❌" in call_args
        assert "reply" in call_args.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_disabled(self, basic_cog, mock_ctx, mock_replied_message):
        """Test evaluate command when disabled in config"""
        mock_ctx.message.reference = MagicMock()
        mock_ctx.message.reference.message_id = 11111
        mock_ctx.channel.fetch_message.return_value = mock_replied_message
        
        with patch.object(basic_cog.config, 'EVALUATE_ENABLED', False):
            await basic_cog.evaluate.callback(basic_cog, mock_ctx)
            
            mock_ctx.send.assert_called_once()
            call_args = mock_ctx.send.call_args[0][0]
            assert "❌" in call_args
            assert "disabled" in call_args.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_message_not_found(self, basic_cog, mock_ctx):
        """Test evaluate command when replied message is deleted"""
        mock_ctx.message.reference = MagicMock()
        mock_ctx.message.reference.message_id = 11111
        mock_ctx.channel.fetch_message.side_effect = discord.NotFound(
            MagicMock(), "message"
        )
        
        await basic_cog.evaluate.callback(basic_cog, mock_ctx)
        
        mock_ctx.send.assert_called_once()
        call_args = mock_ctx.send.call_args[0][0]
        assert "❌" in call_args
        assert "not found" in call_args.lower() or "deleted" in call_args.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_forbidden(self, basic_cog, mock_ctx):
        """Test evaluate command when bot lacks permission"""
        mock_ctx.message.reference = MagicMock()
        mock_ctx.message.reference.message_id = 11111
        mock_ctx.channel.fetch_message.side_effect = discord.Forbidden(
            MagicMock(), "forbidden"
        )
        
        await basic_cog.evaluate.callback(basic_cog, mock_ctx)
        
        mock_ctx.send.assert_called_once()
        call_args = mock_ctx.send.call_args[0][0]
        assert "❌" in call_args
        assert "permission" in call_args.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_empty_message(self, basic_cog, mock_ctx, mock_replied_message):
        """Test evaluate command when replied message is empty"""
        mock_ctx.message.reference = MagicMock()
        mock_ctx.message.reference.message_id = 11111
        mock_replied_message.content = ""
        mock_ctx.channel.fetch_message.return_value = mock_replied_message
        
        await basic_cog.evaluate.callback(basic_cog, mock_ctx)
        
        mock_ctx.send.assert_called_once()
        call_args = mock_ctx.send.call_args[0][0]
        assert "❌" in call_args
        assert "empty" in call_args.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_success(self, basic_cog, mock_ctx, mock_replied_message):
        """Test successful evaluate command execution"""
        mock_ctx.message.reference = MagicMock()
        mock_ctx.message.reference.message_id = 11111
        mock_ctx.channel.fetch_message.return_value = mock_replied_message
        
        # Mock AI service response
        mock_result = {
            'content': """
            After searching the web, I found that this claim is false.
            TRUTHFULNESS: 20%
            EVIDENCE_ALIGNMENT: 15%
            
            Sources:
            https://example.com/fact-check
            https://test.org/evidence
            """,
            'model': 'gpt-4',
            'cost': 0.001,
            'tokens_total': 500
        }
        basic_cog.evaluate_ai_service.generate = AsyncMock(return_value=mock_result)
        
        # Mock status message edit
        status_msg = MagicMock()
        status_msg.edit = AsyncMock()
        mock_ctx.send.return_value = status_msg
        
        await basic_cog.evaluate.callback(basic_cog, mock_ctx)
        
        # Should send status message first
        assert mock_ctx.send.call_count >= 1
        
        # Should call AI service with correct parameters
        basic_cog.evaluate_ai_service.generate.assert_called_once()
        call_kwargs = basic_cog.evaluate_ai_service.generate.call_args[1]
        assert call_kwargs['user_id'] == str(mock_ctx.author.id)
        assert call_kwargs['user_display_name'] == mock_ctx.author.display_name
        assert 'system_prompt' in call_kwargs
        
        # Should edit status message with embed
        assert status_msg.edit.called
        edit_call = status_msg.edit.call_args
        assert edit_call[1]['embed'] is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_ai_error(self, basic_cog, mock_ctx, mock_replied_message):
        """Test evaluate command when AI service fails"""
        mock_ctx.message.reference = MagicMock()
        mock_ctx.message.reference.message_id = 11111
        mock_ctx.channel.fetch_message.return_value = mock_replied_message
        
        # Mock AI service to raise error
        basic_cog.evaluate_ai_service.generate = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        # Mock status message edit
        status_msg = MagicMock()
        status_msg.edit = AsyncMock()
        mock_ctx.send.return_value = status_msg
        
        await basic_cog.evaluate.callback(basic_cog, mock_ctx)
        
        # Should edit status message with error
        assert status_msg.edit.called
        edit_call = status_msg.edit.call_args
        # Check if content is passed as positional or keyword arg
        if edit_call[0] and len(edit_call[0]) > 0:
            content = edit_call[0][0]
        else:
            content = edit_call[1].get('content', '')
        assert "❌" in content or "error" in content.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_rate_limit(self, basic_cog, mock_ctx):
        """Test evaluate command rate limiting"""
        from discord.ext.commands import CommandOnCooldown, BucketType
        
        cooldown_mock = MagicMock()
        cooldown_mock.type = BucketType.user
        error = CommandOnCooldown(cooldown=cooldown_mock, retry_after=30.0, type=BucketType.user)
        
        # Call the error handler directly
        await basic_cog.evaluate_error(mock_ctx, error)
        
        # Should send rate limit message
        mock_ctx.send.assert_called_once()
        call_kwargs = mock_ctx.send.call_args[1]
        assert 'embed' in call_kwargs
        embed = call_kwargs['embed']
        assert "Rate Limit" in embed.title or "⏰" in embed.title


class TestFactCheckingPrompt:
    """Test the fact-checking prompt generation"""
    
    def test_create_fact_checking_prompt(self, basic_cog):
        """Test fact-checking prompt creation"""
        claim = "Test claim"
        prompt = basic_cog._create_fact_checking_prompt(claim)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "fact-checking" in prompt.lower() or "evaluate" in prompt.lower()
        assert "search" in prompt.lower() or "web" in prompt.lower()
        assert "cite" in prompt.lower() or "source" in prompt.lower()
        assert "truthfulness" in prompt.lower()
        assert "evidence" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

