import discord
from discord.ext import commands
from discord.ext.commands import cooldown, BucketType
import logging
from typing import Optional, List
from rag.pipeline import RAGPipeline
from rag.models import RAGConfig, RAGResult
from rag.validation import QueryValidator
from config import Config

class RAG(commands.Cog):
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config  # Store reference
        self.pipeline = RAGPipeline(config=self.config)
        self.logger = logging.getLogger(__name__)

    def _create_base_config(
        self,
        filter_authors: Optional[List[str]] = None,
        **overrides
    ) -> RAGConfig:
        """
        Create RAGConfig with defaults from Config.

        Args:
            filter_authors: Optional list of authors to filter to
            **overrides: Any config values to override defaults

        Returns:
            RAGConfig instance with defaults applied
        """
        config_dict = {
            'top_k': self.config.RAG_DEFAULT_TOP_K,
            'similarity_threshold': self.config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
            'max_context_tokens': self.config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
            'temperature': self.config.RAG_DEFAULT_TEMPERATURE,
            'strategy': self.config.RAG_DEFAULT_STRATEGY,
            'filter_authors': filter_authors,
            'use_hybrid_search': self.config.RAG_USE_HYBRID_SEARCH,
            'use_multi_query': self.config.RAG_USE_MULTI_QUERY,
            'use_hyde': self.config.RAG_USE_HYDE,
            'use_reranking': self.config.RAG_USE_RERANKING,
            'max_output_tokens': self.config.RAG_MAX_OUTPUT_TOKENS,
        }
        # Override with any custom settings
        config_dict.update(overrides)
        return RAGConfig(**config_dict)

    def _split_text_at_sentence_boundary(self, text: str, max_length: int) -> List[str]:
        """
        Split text into chunks at sentence boundaries, preserving paragraph structure.
        
        Preserves:
        - Paragraph breaks (double newlines \n\n)
        - Single newlines within paragraphs
        - Sentence punctuation and spacing
        
        Args:
            text: Text to split
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks with preserved formatting
        """
        import re
        chunks = []
        current_chunk = ""
        
        # Split on sentence endings (period, exclamation, question mark followed by space/newline)
        # This preserves the punctuation and whitespace including paragraph breaks
        sentences = re.split(r'([.!?][\s\n]+)', text)
        
        def preserve_formatting(chunk: str) -> str:
            """Preserve paragraph breaks while cleaning trailing spaces."""
            # Remove trailing spaces/tabs but preserve all newlines (including paragraph breaks \n\n)
            # This keeps paragraph structure intact while cleaning up trailing whitespace
            return chunk.rstrip(' \t')  # Only strip spaces/tabs, keep newlines
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] if i < len(sentences) else ""
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + punctuation
            
            if not full_sentence.strip():
                continue
            
            # If adding this sentence would exceed limit, save current chunk and start new one
            if current_chunk and len(current_chunk) + len(full_sentence) > max_length:
                # Preserve paragraph structure - keep newlines, remove trailing spaces
                if chunks:
                    chunks.append(preserve_formatting(current_chunk))
                else:
                    chunks.append(current_chunk.strip())
                current_chunk = full_sentence.lstrip()  # Remove leading whitespace from new chunk start
            else:
                current_chunk += full_sentence
        
        # Add remaining chunk - preserve all formatting
        if current_chunk.strip():
            if chunks:
                chunks.append(preserve_formatting(current_chunk))
            else:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _summarize_long_response(self, content: str, original_question: str, user_id: str) -> str:
        """
        Summarize a very long RAG response to fit within embed limits.
        
        Args:
            content: Long response content to summarize
            original_question: Original question for context
            user_id: User ID for social credit tone injection
            
        Returns:
            Summarized content
        """
        # Use the pipeline's AI service for summarization
        ai_service = self.pipeline.ai_service
        
        summarize_prompt = f"""The user asked: "{original_question[:200]}"

The AI generated this answer, but it's too long. Please create a concise summary that:
- Preserves the key information and main points
- Keeps only the most important details
- Maintains clarity
- Aim for 2-3 sentences (under 200 words)
- Focus on directly answering the question

Original answer:
{content[:3000]}

Provide a brief, concise summary:"""
        
        try:
            summary_result = await ai_service.generate(
                prompt=summarize_prompt,
                max_tokens=300,  # Reduced for more concise summaries
                temperature=0.3,  # Lower temperature for more focused summary
                user_id=user_id,
                user_display_name=None
            )
            
            summary = summary_result.get('content', '').strip()
            if summary and len(summary) < len(content):
                return summary
            else:
                # If summarization didn't help, return original (will be split)
                return content
        except Exception as e:
            self.logger.error(f"Error summarizing response: {e}", exc_info=True)
            return content  # Return original if summarization fails
    
    async def _send_split_embeds(self, ctx, title: str, content: str, footer_text: str, max_length: int):
        """
        Send content split across multiple embeds if it exceeds Discord's limits.
        
        Args:
            ctx: Command context
            title: Embed title
            content: Content to send
            footer_text: Footer text for embeds
            max_length: Maximum description length per embed
        """
        chunks = self._split_text_at_sentence_boundary(content, max_length)
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks, 1):
            chunk_title = title if i == 1 else f"{title} (Part {i}/{total_chunks})"
            embed = discord.Embed(
                title=chunk_title,
                description=chunk,
                color=discord.Color.blue()
            )
            # Only add footer to last chunk
            if i == total_chunks:
                embed.set_footer(text=footer_text)
            await ctx.send(embed=embed)
    
    def _parse_strategy_from_question(self, question: str) -> tuple[Optional[str], str]:
        """
        Parse optional strategy from question string.
        
        Supports formats:
        - "strategy:single What is..."
        - "single What is..."
        
        Returns:
            (strategy, cleaned_question) - strategy is None if not found
        """
        from chunking.constants import ChunkStrategy
        
        question = question.strip()
        
        # Check for "strategy:xxx" format
        if question.lower().startswith('strategy:'):
            parts = question.split(':', 1)
            if len(parts) == 2:
                strategy_candidate = parts[1].split()[0].lower()
                try:
                    ChunkStrategy(strategy_candidate)
                    remaining_question = parts[1][len(strategy_candidate):].strip()
                    return strategy_candidate, remaining_question
                except ValueError:
                    pass
        
        # Check if first word is a valid strategy
        words = question.split()
        if len(words) > 1:
            first_word = words[0].lower()
            try:
                ChunkStrategy(first_word)
                # Valid strategy found
                remaining_question = ' '.join(words[1:])
                return first_word, remaining_question
            except ValueError:
                pass
        
        # No strategy found, return question as-is
        return None, question

    @commands.command(
        name='ask',
        help='Ask a question about Discord conversations. Use "strategy:xxx" or "xxx" prefix to specify chunking strategy.'
    )
    @cooldown(rate=5, per=60, type=BucketType.user)  # 5 requests per minute per user
    async def ask(self, ctx, *, question: str):
        """
        Simple RAG command for all users.
        
        Usage: 
            !ask What was decided about the database?
            !ask @Alice what did you say about the schema?
            !ask strategy:single What was decided?
            !ask tokens What was decided?
        
        Rate limit: 5 questions per minute per user.
        """
        # Parse strategy from question if present
        strategy, cleaned_question = self._parse_strategy_from_question(question)
        
        # Validate cleaned question
        try:
            cleaned_question = await QueryValidator.validate(
                cleaned_question,
                user_id=str(ctx.author.id),
                user_display_name=ctx.author.display_name,
                social_credit_manager=getattr(self.pipeline.ai_service, 'social_credit_manager', None)
            )
        except ValueError as e:
            await ctx.send(f"❌ {str(e)}")
            return
        
        async with ctx.typing():
            self.logger.info(f"User {ctx.author} asked: {cleaned_question}" + (f" (strategy: {strategy})" if strategy else ""))
            
            # Parse mentioned users from the message
            mentioned_users = []
            if ctx.message.mentions:
                mentioned_users = [user.display_name for user in ctx.message.mentions]
                self.logger.info(f"Filtering to authors: {mentioned_users}")
            
            config_overrides = {
                'filter_authors': mentioned_users if mentioned_users else None,
                'show_sources': False,  # Simple mode = no sources
            }
            
            # Add strategy override if specified
            if strategy:
                config_overrides['strategy'] = strategy
            
            config = self._create_base_config(**config_overrides)
            
            result = await self.pipeline.answer_question(cleaned_question, config)
            
            # Build title with enabled features and filters
            title = "💡 Answer"
            title_parts = []
            
            # Show enabled features
            features = []
            if config.use_hybrid_search:
                features.append("Hybrid")
            if config.use_multi_query:
                features.append("Multi-Query")
            if config.use_hyde:
                features.append("HyDE")
            if config.use_reranking:
                features.append("Reranking")
            if features:
                title_parts.append(" | ".join(features))
            
            # Show strategy if custom
            if strategy:
                title_parts.append(f"strategy: {strategy}")
            
            # Show author filter if applicable
            if mentioned_users:
                authors_str = ", ".join(mentioned_users)
                title_parts.append(f"from {authors_str}")
            
            if title_parts:
                title = f"💡 Answer ({', '.join(title_parts)})"
            
            # Handle long answers: summarize if very long, otherwise split into multiple embeds
            answer_text = result.answer
            max_description_length = 4090  # Leave some buffer
            sources_count = len(result.sources) if result.sources else 0
            footer_text = f"Model: {result.model} | Cost: ${result.cost:.4f} | {sources_count} sources"
            
            # Summarization threshold: summarize if answer is longer than a reasonable Discord message
            SUMMARIZE_THRESHOLD = 3000  # Summarize if > 3000 chars (allows longer responses before summarizing)
            
            if len(answer_text) > SUMMARIZE_THRESHOLD:
                # Summarize long responses to keep them concise
                self.logger.info(f"Answer ({len(answer_text)} chars) exceeds summarize threshold ({SUMMARIZE_THRESHOLD}), summarizing...")
                try:
                    answer_text = await self._summarize_long_response(answer_text, cleaned_question, str(ctx.author.id))
                    self.logger.info(f"Summarized to {len(answer_text)} chars")
                except Exception as e:
                    self.logger.error(f"Failed to summarize response: {e}", exc_info=True)
                    # Fall back to splitting if summarization fails
                    pass
            
            if len(answer_text) <= max_description_length:
                # Single embed
                embed = discord.Embed(
                    title=title,
                    description=answer_text,
                    color=discord.Color.blue()
                )
                embed.set_footer(text=footer_text)
                message = await ctx.send(embed=embed)
                await message.add_reaction("📚")
            else:
                # Split into multiple embeds
                await self._send_split_embeds(ctx, title, answer_text, footer_text, max_description_length)
                # Get the last message sent by the bot for reaction
                message = None
                async for msg in ctx.channel.history(limit=10):
                    if msg.author == ctx.bot.user and msg.embeds:
                        message = msg
                        break
                if message:
                    await message.add_reaction("📚")
            
            self.bot._rag_cache = getattr(self.bot, '_rag_cache', {})
            self.bot._rag_cache[message.id] = result

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):

        if user.bot:
            return
        
        if str(reaction.emoji) != "📚":
            return
        
        rag_cache = getattr(self.bot, '_rag_cache', {})
        result = rag_cache.get(reaction.message.id)
        
        if not result or not result.sources:
            return
        
        embed = discord.Embed(
            title="📚 Sources Used",
            description=f"Top {len(result.sources)} relevant chunks (similarity scores shown):",
            color=discord.Color.green()
        )
        
        for i, source in enumerate(result.sources, 1):
            content = source.get('content', '')
            metadata = source.get('metadata', {})
            similarity = source.get('similarity', 0)
            
            preview = content[:200] + "..." if len(content) > 200 else content
            
            # Color code the similarity score
            if similarity >= 0.7:
                similarity_emoji = "🟢"  # High relevance
            elif similarity >= 0.5:
                similarity_emoji = "🟡"  # Medium relevance
            else:
                similarity_emoji = "🟠"  # Lower relevance
            
            # Extract author - try multiple field names for compatibility
            author = (
                metadata.get('author') or 
                metadata.get('primary_author_name') or 
                metadata.get('primary_author_id') or 
                'Unknown'
            )
            
            # Extract timestamp - prefer first_timestamp, fallback to timestamp
            timestamp = (
                metadata.get('first_timestamp') or 
                metadata.get('timestamp') or 
                metadata.get('last_timestamp') or 
                'Unknown'
            )
            
            # Format timestamp for display (extract date part if full ISO format)
            if timestamp != 'Unknown' and len(timestamp) > 10:
                timestamp = timestamp[:10]  # Show just YYYY-MM-DD
            
            embed.add_field(
                name=f"{i}. {similarity_emoji} Similarity: {similarity:.3f}",
                value=(
                    f"**Author:** {author}\n"
                    f"**Time:** {timestamp}\n"
                    f"**Content:** {preview}"
                ),
                inline=False
            )
        
        # Reply to the original message with sources
        await reaction.message.reply(embed=embed, mention_author=False)

    async def _handle_cooldown_error(
        self,
        ctx,
        error,
        rate_limit_msg: str,
        limit_text: str
    ):
        """
        Generic cooldown error handler.

        Args:
            ctx: Discord context
            error: Command error
            rate_limit_msg: Custom message explaining the rate limit
            limit_text: Text describing the limit (e.g., "5 questions per minute")
        """
        if isinstance(error, commands.CommandOnCooldown):
            minutes, seconds = divmod(int(error.retry_after), 60)

            embed = discord.Embed(
                title="⏰ Rate Limit Reached",
                description=(
                    f"{rate_limit_msg}\n\n"
                    f"Please wait **{minutes}m {seconds}s** before trying again."
                ),
                color=discord.Color.orange()
            )
            embed.set_footer(text=f"Limit: {limit_text}")

            await ctx.send(embed=embed)
        else:
            raise error

    @ask.error
    async def ask_error(self, ctx, error):
        """Handle errors for the ask command."""
        await self._handle_cooldown_error(
            ctx, error,
            rate_limit_msg=(
                "You're asking questions too quickly!\n\n"
                "This helps prevent API cost overruns and ensures fair usage."
            ),
            limit_text="5 questions per minute"
        )

async def setup(bot):
    await bot.add_cog(RAG(bot))