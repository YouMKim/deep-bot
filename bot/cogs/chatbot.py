"""
Chatbot cog for conversational AI in Discord.

Provides natural conversation capabilities with RAG-enhanced question answering.
"""

import discord
from discord.ext import commands, tasks
import logging
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
from collections import OrderedDict

from config import Config
from ai.service import AIService
from ai.tracker import UserAITracker
from rag.pipeline import RAGPipeline
from bot.utils.session_manager import SessionManager
from bot.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class Chatbot(commands.Cog):
    """Chatbot cog for natural conversation with RAG support."""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config
        
        # Initialize services
        self.session_manager = SessionManager(
            max_history=self.config.CHATBOT_MAX_HISTORY,
            session_timeout=self.config.CHATBOT_SESSION_TIMEOUT,
            max_context_tokens=self.config.RAG_DEFAULT_MAX_CONTEXT_TOKENS
        )
        
        self.rate_limiter = RateLimiter(
            max_messages=self.config.CHATBOT_RATE_LIMIT_MESSAGES,
            window_seconds=self.config.CHATBOT_RATE_LIMIT_WINDOW
        )
        
        self.ai_service = AIService(provider_name=self.config.AI_DEFAULT_PROVIDER)
        self.rag_pipeline = RAGPipeline(config=self.config)
        self.ai_tracker = UserAITracker()
        
        # Channel context cache: {channel_id: (context, timestamp)} with LRU eviction
        self._channel_context_cache: OrderedDict[int, Tuple[str, datetime]] = OrderedDict()
        self._cache_ttl = 30  # seconds
        self._max_cache_size = 100  # Maximum number of channels to cache
        
        # Start background cleanup task
        self.cleanup_sessions.start()
        self.cleanup_rate_limits.start()
        
        logger.info("Chatbot cog initialized")
    
    async def _send_long_message(self, channel, content: str):
        """
        Send a message, splitting it into multiple messages if it exceeds Discord's 2000 char limit.
        
        Args:
            channel: Discord channel to send to
            content: Message content to send
        """
        DISCORD_MAX_LENGTH = 2000
        
        if len(content) <= DISCORD_MAX_LENGTH:
            # Message fits in one send
            await channel.send(content)
            return
        
        # Split into chunks, trying to break at sentence boundaries
        # Reserve space for prefix (e.g., "*(Part 2/3)*\n\n" ≈ 20 chars)
        # Use 1980 as max chunk size to leave room for prefix
        MAX_CHUNK_SIZE = 1980
        chunks = []
        current_chunk = ""
        
        # Split by sentences first (period, exclamation, question mark followed by space or newline)
        import re
        # Split on sentence endings, keeping punctuation
        sentences = re.split(r'([.!?][\s\n]+)', content)
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] if i < len(sentences) else ""
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + punctuation
            
            # Skip empty sentences
            if not full_sentence.strip():
                continue
            
            # If adding this sentence would exceed limit, save current chunk and start new one
            if len(current_chunk) + len(full_sentence) > MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = full_sentence
                else:
                    # Single sentence is too long, force split by character
                    # But try to avoid splitting in the middle of Unicode characters
                    if len(full_sentence) > MAX_CHUNK_SIZE:
                        # Split long sentence into character chunks, ensuring we don't break Unicode
                        remaining = full_sentence
                        while len(remaining) > MAX_CHUNK_SIZE:
                            # Try to find a safe break point (space or newline) near the limit
                            safe_break = MAX_CHUNK_SIZE
                            # Look backwards for a space or newline within last 100 chars
                            for j in range(MAX_CHUNK_SIZE, max(0, MAX_CHUNK_SIZE - 100), -1):
                                if j < len(remaining) and remaining[j] in [' ', '\n']:
                                    safe_break = j + 1
                                    break
                            
                            chunk = remaining[:safe_break]
                            if chunk.strip():  # Only add non-empty chunks
                                chunks.append(chunk.strip())
                            remaining = remaining[safe_break:]
                        
                        if remaining.strip():
                            current_chunk = remaining
                    else:
                        current_chunk = full_sentence
            else:
                current_chunk += full_sentence
        
        # Add remaining chunk
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Send all chunks
        for i, chunk in enumerate(chunks):
            if chunk and chunk.strip():  # Only send non-empty chunks
                # Add continuation indicator for multi-part messages
                if len(chunks) > 1:
                    prefix = f"*(Part {i + 1}/{len(chunks)})*\n\n"
                    # Ensure prefix + chunk doesn't exceed limit
                    if len(prefix) + len(chunk) > DISCORD_MAX_LENGTH:
                        # If prefix pushes us over, reduce chunk size
                        max_chunk_with_prefix = DISCORD_MAX_LENGTH - len(prefix)
                        chunk = chunk[:max_chunk_with_prefix]
                    await channel.send(prefix + chunk)
                else:
                    await channel.send(chunk)
    
    def cog_unload(self):
        """Clean up when cog is unloaded."""
        self.cleanup_sessions.cancel()
        self.cleanup_rate_limits.cancel()
        logger.info("Chatbot cog unloaded")
    
    @tasks.loop(minutes=10)
    async def cleanup_sessions(self):
        """Remove expired sessions to free memory."""
        await self.session_manager.cleanup_expired_sessions()
        logger.debug("Cleaned up expired chatbot sessions")
    
    @cleanup_sessions.before_loop
    async def before_cleanup_sessions(self):
        """Wait until bot is ready before starting cleanup task."""
        await self.bot.wait_until_ready()
    
    @tasks.loop(minutes=15)
    async def cleanup_rate_limits(self):
        """Clean up old rate limit entries."""
        await self.rate_limiter.cleanup_old_entries()
        logger.debug("Cleaned up old rate limit entries")
    
    @cleanup_rate_limits.before_loop
    async def before_cleanup_rate_limits(self):
        """Wait until bot is ready before starting cleanup task."""
        await self.bot.wait_until_ready()
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Main chatbot event listener."""
        # Skip if message is from bot
        if message.author.bot:
            return
        
        # Skip if message is a command
        if message.content.startswith(self.config.BOT_PREFIX):
            return
        
        # Check if message is in chatbot channel
        if message.channel.id != self.config.CHATBOT_CHANNEL_ID:
            return

        
        # Check rate limit
        allowed, retry_after = await self.rate_limiter.check_rate_limit(message.author.id)
        if not allowed:
            minutes, seconds = divmod(int(retry_after), 60)
            await message.channel.send(
                f"⏰ {message.author.mention} Rate limit exceeded. "
                f"Please wait {minutes}m {seconds}s before sending another message."
            )
            return
        
        # Sanitize input
        sanitized_content = self._sanitize_chat_input(message.content)
        if not sanitized_content:
            await message.channel.send(
                f"❌ {message.author.mention} Message is empty after sanitization."
            )
            return
        
        # Process message
        try:
            async with message.channel.typing():
                # React to show message received
                try:
                    await message.add_reaction("👀")
                except discord.HTTPException:
                    pass  # Ignore reaction errors
                
                # Get or create channel session
                channel_id = message.channel.id
                session = await self.session_manager.get_session(channel_id)
                
                # Detect if message is a question
                is_question = await self._is_question(sanitized_content)
                
                # Extract mentions for potential RAG filtering
                mentioned_users = self._extract_mentions(message)
                
                # Determine if RAG is needed (only for questions that need history search)
                needs_rag = False
                if is_question and self.config.CHATBOT_USE_RAG:
                    needs_rag = await self._needs_rag(sanitized_content, mentioned_users)
                
                # Generate response
                if needs_rag:
                    # Use RAG for questions about past conversations/events
                    response = await self._generate_rag_response(
                        sanitized_content,
                        message.author.id,
                        mentioned_users,
                        channel_id
                    )
                else:
                    # Use conversational chat for everything else
                    response = await self._generate_chat_response(
                        sanitized_content,
                        channel_id,
                        message.author.display_name
                    )
                
                # Update session history BEFORE sending (for consistency)
                await self.session_manager.add_message(
                    channel_id,
                    "user",
                    sanitized_content,
                    author_id=message.author.id,
                    author_name=message.author.display_name
                )
                await self.session_manager.add_message(
                    channel_id,
                    "assistant",
                    response['content']
                )
                
                # Invalidate channel context cache since we just added a message
                if channel_id in self._channel_context_cache:
                    del self._channel_context_cache[channel_id]
                
                # Remove "eyes" reaction, add "check" reaction
                try:
                    await message.remove_reaction("👀", self.bot.user)
                    await message.add_reaction("✅")
                except discord.HTTPException:
                    pass  # Ignore reaction errors
                
                # Send response (split if exceeds Discord's 2000 char limit)
                if response.get('content'):
                    # Sanitize content to remove any corrupted characters
                    content = self._sanitize_response_content(response['content'])
                    
                    if content:
                        await self._send_long_message(message.channel, content)
                    else:
                        logger.error(f"Content became empty after sanitization for channel {message.channel.id}")
                        await message.channel.send(
                            "❌ Sorry, I couldn't generate a valid response. Please try again."
                        )
                else:
                    logger.error(f"Response has no content for channel {message.channel.id}")
                    await message.channel.send(
                        "❌ Sorry, I couldn't generate a response. Please try again."
                    )
                
                # Track usage
                user_display_name = message.author.display_name
                self.ai_tracker.log_ai_usage(
                    user_display_name=user_display_name,
                    cost=response.get('cost', 0.0),
                    tokens_total=response.get('tokens_total', 0)
                )
                
        except discord.HTTPException as e:
            logger.error(f"Discord API error in chatbot: {e}", exc_info=True)
            await message.channel.send(
                f"⚠️ {message.author.mention} Discord API error. Please try again."
            )
        except Exception as e:
            logger.error(f"Chatbot error: {e}", exc_info=True)
            await message.channel.send(
                f"❌ Sorry {message.author.mention}, I encountered an error processing your message. "
                f"Please try again or use `{self.config.BOT_PREFIX}chatbot_reset` to clear your session."
            )
    
    async def _is_question(self, text: str) -> bool:
        """
        Detect if message is a question.
        
        Uses enhanced heuristics including:
        - Question marks
        - Question starters
        - Question phrases
        - Imperative questions
        """
        text_lower = text.lower().strip()
        
        # Direct question mark
        if text_lower.endswith('?'):
            return True
        
        # Question starters (comprehensive list)
        question_starters = [
            'what', 'when', 'where', 'who', 'why', 'how',
            'did', 'does', 'is', 'are', 'was', 'were',
            'can', 'could', 'would', 'should', 'will',
            'has', 'have', 'had', 'do', 'does'
        ]
        
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in question_starters:
            return True
        
        # Question phrases
        question_phrases = [
            'tell me', 'explain', 'what about', 'how about',
            'do you know', 'can you', 'could you', 'would you'
        ]
        for phrase in question_phrases:
            if phrase in text_lower:
                return True
        
        # Imperative questions (short commands like "Explain X", "Tell me about Y")
        if len(text_lower.split()) <= 5:
            imperative_words = ['explain', 'tell', 'describe', 'show']
            if any(word in text_lower for word in imperative_words):
                return True
        
        return False
    
    async def _needs_rag(self, text: str, mentioned_users: List[str] = None) -> bool:
        """
        Determine if a question requires RAG search vs. conversational response.
        
        RAG is needed for:
        - Questions about past events/conversations
        - Questions referencing specific people (with mentions or names)
        - Questions with temporal references (yesterday, last week, etc.)
        - Questions asking about decisions/plans made in Discord
        - Questions with keywords suggesting history search
        
        RAG is NOT needed for:
        - Greetings and small talk ("how are you", "what's up")
        - General conversational questions
        - Questions about the bot itself
        
        Args:
            text: The question text
            mentioned_users: List of mentioned users (if any)
            
        Returns:
            True if RAG search is needed, False for conversational response
        """
        text_lower = text.lower().strip()
        
        # If users are mentioned, likely asking about their messages
        if mentioned_users:
            return True
        
        # Temporal references suggest asking about past events
        temporal_keywords = [
            'yesterday', 'last week', 'last month', 'earlier', 'before',
            'previously', 'recently', 'earlier today', 'last time',
            'did we', 'did they', 'did you', 'was there', 'were there',
            'earlier?', 'earlier.', 'earlier!'
        ]
        
        # Conversational patterns that DON'T need RAG
        conversational_patterns = [
            'how are you', 'how\'s it going', 'what\'s up', 'whats up',
            'how do you do', 'how\'s everything', 'how are things',
            'what are you doing', 'what do you do', 'what can you do',
            'who are you', 'what are you', 'what is this', 'what is that',
            'can you help', 'can you do', 'will you', 'would you',
            'hi', 'hello', 'hey', 'greetings'
        ]
        
        # Check if it's a simple conversational question
        for pattern in conversational_patterns:
            if pattern in text_lower:
                # But allow if it has additional context suggesting RAG
                # Check for history keywords or temporal references
                history_context = any(keyword in text_lower for keyword in ['say', 'said', 'mention', 'discuss', 'decide', 'plan', 'talk'])
                temporal_context = any(keyword in text_lower for keyword in temporal_keywords)
                if not (history_context or temporal_context):
                    return False
        if any(keyword in text_lower for keyword in temporal_keywords):
            return True
        
        # Keywords suggesting asking about Discord history
        history_keywords = [
            'say', 'said', 'mention', 'discuss', 'decide', 'decided',
            'plan', 'planned', 'talk about', 'talked about',
            'decide on', 'decided on', 'agree', 'agreed',
            'conversation', 'discussion', 'meeting', 'call'
        ]
        if any(keyword in text_lower for keyword in history_keywords):
            return True
        
        # Questions about specific people (names or pronouns in context)
        # This is a simple heuristic - could be improved with NER
        people_indicators = [
            'what did', 'what did they', 'what did he', 'what did she',
            'what did we', 'what did you', 'what did someone',
            'who said', 'who mentioned', 'who discussed',
            'did anyone', 'did somebody', 'did someone'
        ]
        if any(indicator in text_lower for indicator in people_indicators):
            return True
        
        # Questions starting with "what" that are longer (likely specific questions)
        # vs. short "what's up" type questions
        if text_lower.startswith('what') and len(text_lower.split()) > 3:
            # Check if it's asking about something specific
            if any(word in text_lower for word in ['about', 'regarding', 'concerning', 'regarding']):
                return True
        
        # Questions about decisions, plans, or outcomes
        decision_keywords = [
            'decide', 'decision', 'choose', 'chose', 'pick', 'picked',
            'plan', 'planning', 'schedule', 'scheduled',
            'agree', 'agreement', 'consensus', 'vote', 'voted'
        ]
        if any(keyword in text_lower for keyword in decision_keywords):
            return True
        
        # Default: if it's a question but doesn't match conversational patterns,
        # use chat mode (safer default - avoids unnecessary RAG searches)
        return False
    
    def _extract_mentions(self, message: discord.Message) -> List[str]:
        """
        Extract mentioned users for RAG filtering.
        
        Returns list of display names for mentioned users.
        """
        return [user.display_name for user in message.mentions if not user.bot]
    
    def _sanitize_response_content(self, content: str) -> str:
        """
        Sanitize AI-generated response content to remove corrupted characters.
        
        Args:
            content: Raw content from AI response
            
        Returns:
            Sanitized content string
        """
        if not content:
            return ""
        
        # Remove any null bytes and other control characters (except newlines and tabs)
        import re
        # Remove null bytes and other problematic control chars
        content = content.replace('\x00', '')
        # Remove other control characters except newline, tab, carriage return
        content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)
        
        # Check if content looks corrupted (too many non-printable or weird characters)
        # If more than 10% of characters are non-ASCII and non-printable, it might be corrupted
        non_printable_count = sum(1 for c in content if ord(c) > 127 and not c.isprintable() and c not in '\n\t')
        if len(content) > 0 and non_printable_count / len(content) > 0.1:
            logger.warning(f"Detected potentially corrupted content: {non_printable_count}/{len(content)} non-printable chars")
            # Try to recover by keeping only printable ASCII + common Unicode
            content = ''.join(c for c in content if c.isprintable() or c in '\n\t')
        
        return content.strip()
    
    def _sanitize_chat_input(self, text: str) -> str:
        """
        Sanitize chat input by removing excessive whitespace and control characters.
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove control characters (except newlines and tabs)
        import re
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace (collapse multiple spaces, keep single newlines)
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def _generate_rag_response(
        self,
        question: str,
        user_id: int,
        mentioned_users: Optional[List[str]] = None,
        channel_id: Optional[int] = None
    ) -> dict:
        """
        Use RAG pipeline to answer questions with context from past messages.
        
        Args:
            question: User's question
            user_id: User ID for tracking
            mentioned_users: Optional list of @mentioned users to filter
            channel_id: Optional channel ID for fallback context
            
        Returns:
            dict with answer, sources, cost, etc.
        """
        try:
            # Build RAG config using shared utility method
            # Use RAG-specific token limit (can be longer for detailed answers)
            config = self.config.create_rag_config(
                similarity_threshold=self.config.CHATBOT_RAG_THRESHOLD,
                temperature=self.config.CHATBOT_TEMPERATURE,
                filter_authors=mentioned_users if mentioned_users else None,
                max_output_tokens=self.config.CHATBOT_RAG_MAX_TOKENS,
            )
            
            # Get answer from RAG pipeline
            result = await self.rag_pipeline.answer_question(question, config)
            
            # Summarize RAG response only if it exceeds Discord's 2000 char limit
            content = result.answer
            DISCORD_CHAR_LIMIT = 2000
            if len(content) > DISCORD_CHAR_LIMIT:
                logger.info(f"RAG response ({len(content)} chars) exceeds Discord limit ({DISCORD_CHAR_LIMIT}), summarizing...")
                content = await self._summarize_if_needed(content, question, is_rag=True)
            
            return {
                'content': content,
                'sources': result.sources,
                'cost': result.cost,
                'model': result.model,
                'tokens_total': result.tokens_used,
                'mode': 'rag'
            }
        except Exception as e:
            logger.error(f"RAG response generation error: {e}", exc_info=True)
            # Fallback to chat mode on RAG error with user notification
            fallback_response = await self._generate_chat_response(
                question,
                channel_id,
                "User"  # Default author name for fallback
            )
            # Prepend warning to inform user
            fallback_response['content'] = (
                "⚠️ *Note: Unable to search conversation history. "
                "Responding without historical context.*\n\n" +
                fallback_response['content']
            )
            return fallback_response
    
    async def _get_recent_channel_context(self, channel_id: int, limit: int = 5) -> str:
        """
        Fetch recent messages from channel for additional context.
        Uses caching to reduce Discord API calls.
        
        Args:
            channel_id: Discord channel ID
            limit: Number of recent messages to include
            
        Returns:
            Formatted string of recent messages
        """
        # Check cache first
        now = datetime.now()
        if channel_id in self._channel_context_cache:
            context, cache_time = self._channel_context_cache[channel_id]
            if (now - cache_time).total_seconds() < self._cache_ttl:
                # Move to end (most recently used) for LRU
                self._channel_context_cache.move_to_end(channel_id)
                return context
        
        # Cache miss or expired - fetch from Discord
        try:
            channel = self.bot.get_channel(channel_id)
            if not channel:
                return ""
            
            messages = []
            async for msg in channel.history(limit=limit):
                if not msg.author.bot and not msg.content.startswith(self.config.BOT_PREFIX):
                    messages.append(f"{msg.author.display_name}: {msg.content}")
            
            messages.reverse()  # Chronological order
            
            if messages:
                context = "Recent channel context:\n" + "\n".join(messages) + "\n\n"
                # Update cache with LRU eviction
                if channel_id in self._channel_context_cache:
                    # Update existing entry and move to end
                    self._channel_context_cache[channel_id] = (context, now)
                    self._channel_context_cache.move_to_end(channel_id)
                else:
                    # Add new entry
                    self._channel_context_cache[channel_id] = (context, now)
                    # Evict oldest if over limit
                    if len(self._channel_context_cache) > self._max_cache_size:
                        self._channel_context_cache.popitem(last=False)
                return context
            return ""
        except Exception as e:
            logger.warning(f"Error fetching channel context: {e}")
            return ""
    
    async def _generate_chat_response(
        self,
        message: str,
        channel_id: int,
        author_name: str
    ) -> dict:
        """
        Generate conversational response using chat history.
        
        Args:
            message: User's message
            channel_id: Channel ID for session tracking
            author_name: Display name of message author
            
        Returns:
            dict with content, cost, model, etc.
        """
        try:
            # Get channel context if available
            channel_context = await self._get_recent_channel_context(
                channel_id,
                self.config.CHATBOT_INCLUDE_CONTEXT_MESSAGES
            )
            
            # Format prompt with history (channel-level)
            prompt = await self.session_manager.format_for_ai(
                channel_id,
                message,
                author_name,
                self.config.CHATBOT_SYSTEM_PROMPT,
                channel_context
            )
            
            # Generate response with conversational token limit (shorter for chat)
            result = await self.ai_service.generate(
                prompt=prompt,
                max_tokens=self.config.CHATBOT_CHAT_MAX_TOKENS,
                temperature=self.config.CHATBOT_TEMPERATURE
            )
            
            # If response is long, try to summarize it for more conversational feel
            content = result['content']
            if not content:
                logger.warning(f"AI service returned empty content for channel {channel_id}")
            
            if len(content) > 800:  # If response is longer than ~800 chars (≈200 tokens)
                content = await self._summarize_if_needed(content, message, is_rag=False)
            
            result['content'] = content
            return {
                'content': result['content'],
                'cost': result['cost'],
                'model': result['model'],
                'tokens_total': result['tokens_total'],
                'mode': 'chat'
            }
        except Exception as e:
            logger.error(f"[CHAT] Chat response generation error: {e}", exc_info=True)
            return {
                'content': "Sorry, I encountered an error generating a response.",
                'cost': 0.0,
                'model': 'error',
                'tokens_total': 0,
                'mode': 'chat'
            }
    
    async def _summarize_if_needed(self, content: str, original_message: str, is_rag: bool = False) -> str:
        """
        Summarize long responses to keep them conversational.
        
        Args:
            content: Long response content
            original_message: Original user message for context
            is_rag: Whether this is a RAG response (allows slightly longer summaries)
            
        Returns:
            Summarized or original content
        """
        # Different thresholds for chat vs RAG
        # RAG: Only summarize if exceeds Discord's 2000 char limit
        # Chat: Summarize at 800 chars to keep it conversational
        DISCORD_CHAR_LIMIT = 2000
        threshold = DISCORD_CHAR_LIMIT if is_rag else 800
        if len(content) <= threshold:
            return content
        
        try:
            # Create a summarization prompt
            context_type = "RAG answer" if is_rag else "response"
            max_summary_tokens = 300 if is_rag else 200
            content_preview_length = 2000 if is_rag else 1500
            
            summarize_prompt = f"""The user asked: "{original_message[:200]}"

The AI generated this {context_type}, but it's too long for a conversational Discord chat. Please summarize it to be more concise and conversational (aim for 3-5 sentences for RAG, 2-4 for chat, keep the key points):

{content[:content_preview_length]}

Provide a concise, conversational summary:"""
            
            # Generate summary with appropriate token limit
            summary_result = await self.ai_service.generate(
                prompt=summarize_prompt,
                max_tokens=max_summary_tokens,
                temperature=0.7  # Lower temperature for more focused summary
            )
            
            summary = summary_result.get('content', '').strip()
            if summary and len(summary) < len(content):
                logger.info(f"Summarized {'RAG' if is_rag else 'chat'} response from {len(content)} to {len(summary)} chars")
                return summary
            else:
                # If summary failed or wasn't shorter, return original
                return content
        except Exception as e:
            logger.warning(f"Failed to summarize response: {e}")
            # Return original content if summarization fails
            return content
    
    @commands.command(name='chatbot_reset', help='Reset channel conversation history')
    async def reset_conversation(self, ctx):
        """Reset the channel's conversation history."""
        channel_id = ctx.channel.id
        await self.session_manager.reset_session(channel_id)
        
        # Invalidate cache
        if channel_id in self._channel_context_cache:
            del self._channel_context_cache[channel_id]
        
        await ctx.send(f"✅ {ctx.author.mention} Channel conversation history has been cleared!")
    
    @commands.command(name='chatbot_stats', help='View channel chatbot usage statistics')
    async def chatbot_stats(self, ctx):
        """Show channel's chatbot usage statistics."""
        channel_id = ctx.channel.id
        session = await self.session_manager.get_session(channel_id)
        
        embed = discord.Embed(
            title=f"📊 Chatbot Statistics for #{ctx.channel.name}",
            color=discord.Color.blue()
        )
        
        message_count = len(session.get('messages', []))
        created_at = session.get('created_at', 'N/A')
        last_activity = session.get('last_activity', 'N/A')
        
        # Format datetime if it's a datetime object
        if isinstance(created_at, datetime):
            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(created_at, str):
            try:
                dt = datetime.fromisoformat(created_at)
                created_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        if isinstance(last_activity, datetime):
            last_activity = last_activity.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(last_activity, str):
            try:
                dt = datetime.fromisoformat(last_activity)
                last_activity = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        embed.add_field(name="Messages in Session", value=str(message_count), inline=True)
        embed.add_field(name="Session Started", value=str(created_at), inline=True)
        embed.add_field(name="Last Activity", value=str(last_activity), inline=True)
        
        # Get user's chatbot usage stats (for the user who called the command)
        stats = self.ai_tracker.get_user_stats(ctx.author.display_name)
        
        if stats:
            embed.add_field(
                name=f"Your Total Cost (from {ctx.author.display_name})",
                value=f"${stats['lifetime_cost']:.4f}",
                inline=True
            )
            embed.add_field(
                name=f"Your Total Tokens (from {ctx.author.display_name})",
                value=f"{stats['lifetime_tokens']:,}",
                inline=True
            )
            # Note: Social credit removed - use !mystats for personal stats including social credit
        
        embed.set_footer(text="Use !mystats for your complete personal stats including social credit")
        await ctx.send(embed=embed)
    
    @commands.command(name='chatbot_mode', help='Check current chatbot settings (Admin only)')
    async def chatbot_mode(self, ctx):
        """Display current chatbot configuration."""
        if str(ctx.author.id) != str(self.config.BOT_OWNER_ID):
            await ctx.send("🚫 This command is admin-only!")
            return
        
        embed = discord.Embed(
            title="🤖 Chatbot Configuration",
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="Channel ID",
            value=str(self.config.CHATBOT_CHANNEL_ID),
            inline=True
        )
        embed.add_field(
            name="Max History",
            value=str(self.config.CHATBOT_MAX_HISTORY),
            inline=True
        )
        embed.add_field(
            name="Session Timeout",
            value=f"{self.config.CHATBOT_SESSION_TIMEOUT}s",
            inline=True
        )
        embed.add_field(
            name="Max Tokens",
            value=f"{self.config.CHATBOT_CHAT_MAX_TOKENS} (chat) / {self.config.CHATBOT_RAG_MAX_TOKENS} (RAG)",
            inline=True
        )
        embed.add_field(
            name="Temperature",
            value=str(self.config.CHATBOT_TEMPERATURE),
            inline=True
        )
        embed.add_field(
            name="Use RAG",
            value="✅" if self.config.CHATBOT_USE_RAG else "❌",
            inline=True
        )
        embed.add_field(
            name="RAG Threshold",
            value=str(self.config.CHATBOT_RAG_THRESHOLD),
            inline=True
        )
        embed.add_field(
            name="Rate Limit",
            value=f"{self.config.CHATBOT_RATE_LIMIT_MESSAGES}/min",
            inline=True
        )
        embed.add_field(
            name="AI Provider",
            value=self.config.AI_DEFAULT_PROVIDER,
            inline=True
        )
        
        await ctx.send(embed=embed)


async def setup(bot):
    """Load the chatbot cog."""
    # Validate chatbot config
    if not Config.validate_chatbot_config():
        logger.error("Chatbot configuration validation failed. Cog not loaded.")
        return
    
    await bot.add_cog(Chatbot(bot))
    logger.info("Chatbot cog loaded")

