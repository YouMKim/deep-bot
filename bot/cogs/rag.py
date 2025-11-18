import discord
from discord.ext import commands
from discord.ext.commands import cooldown, BucketType
import logging
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

    @commands.command(
        name='ask',
        help='Ask a question about Discord conversations. Mention users to filter to their messages.'
    )
    @cooldown(rate=5, per=60, type=BucketType.user)  # 5 requests per minute per user
    async def ask(self, ctx, *, question: str):
        """
        Simple RAG command for all users.
        
        Usage: 
            !ask What was decided about the database?
            !ask @Alice what did you say about the schema?
        
        Rate limit: 5 questions per minute per user.
        """
        # Validate before processing
        try:
            question = QueryValidator.validate(question)
        except ValueError as e:
            await ctx.send(f"❌ {str(e)}")
            return
        
        async with ctx.typing():
            self.logger.info(f"User {ctx.author} asked: {question}")
            
            # Parse mentioned users from the message
            mentioned_users = []
            if ctx.message.mentions:
                mentioned_users = [user.display_name for user in ctx.message.mentions]
                self.logger.info(f"Filtering to authors: {mentioned_users}")
            
            config = RAGConfig(
                top_k=self.config.RAG_DEFAULT_TOP_K,
                similarity_threshold=self.config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
                max_context_tokens=self.config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
                temperature=self.config.RAG_DEFAULT_TEMPERATURE,
                strategy=self.config.RAG_DEFAULT_STRATEGY,
                show_sources=False,  # Simple mode = no sources
                filter_authors=mentioned_users if mentioned_users else None,
            )
            
            result = await self.pipeline.answer_question(question, config)
            
            # Add author filter info to title if applicable
            title = "💡 Answer"
            if mentioned_users:
                authors_str = ", ".join(mentioned_users)
                title = f"💡 Answer (from {authors_str})"
            
            embed = discord.Embed(
                title=title,
                description=result.answer,
                color=discord.Color.blue()
            )
            
            # Show how many sources were used
            sources_count = len(result.sources) if result.sources else 0
            embed.set_footer(
                text=f"Model: {result.model} | Cost: ${result.cost:.4f} | {sources_count} sources"
            )
            
            message = await ctx.send(embed=embed)
            
            await message.add_reaction("📚")
            
            self.bot._rag_cache = getattr(self.bot, '_rag_cache', {})
            self.bot._rag_cache[message.id] = result

    @commands.command(name='ask_hybrid')
    @cooldown(rate=3, per=60, type=BucketType.user)  # 3 requests per minute (more expensive)
    async def ask_hybrid(self, ctx, *, question: str):
        """
        Ask a question using hybrid search (BM25 + vector).

        Usage: !ask_hybrid What was decided about the database?
        
        Rate limit: 3 questions per minute per user (more expensive than regular ask).
        """
        # Validate before processing
        try:
            question = QueryValidator.validate(question)
        except ValueError as e:
            await ctx.send(f"❌ {str(e)}")
            return
        
        async with ctx.typing():
            self.logger.info(f"User {ctx.author} asked (hybrid): {question}")

            mentioned_users = []
            if ctx.message.mentions:
                mentioned_users = [user.display_name for user in ctx.message.mentions]

            config = RAGConfig(
                top_k=self.config.RAG_DEFAULT_TOP_K,
                similarity_threshold=self.config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
                max_context_tokens=self.config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
                temperature=self.config.RAG_DEFAULT_TEMPERATURE,
                strategy=self.config.RAG_DEFAULT_STRATEGY,
                use_hybrid_search=True,  
                bm25_weight=0.5,
                vector_weight=0.5,
                filter_authors=mentioned_users if mentioned_users else None,
            )

            result = await self.pipeline.answer_question(question, config)

            title = "💡 Answer (Hybrid Search)"
            if mentioned_users:
                authors_str = ", ".join(mentioned_users)
                title = f"💡 Answer (Hybrid - {authors_str})"

            embed = discord.Embed(
                title=title,
                description=result.answer,
                color=discord.Color.green()  
            )

            sources_count = len(result.sources) if result.sources else 0
            embed.set_footer(
                text=f"Model: {result.model} | Cost: ${result.cost:.4f} | {sources_count} sources"
            )

            message = await ctx.send(embed=embed)
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
            
            embed.add_field(
                name=f"{i}. {similarity_emoji} Similarity: {similarity:.3f}",
                value=(
                    f"**Author:** {metadata.get('author', 'Unknown')}\n"
                    f"**Time:** {metadata.get('timestamp', 'Unknown')}\n"
                    f"**Content:** {preview}"
                ),
                inline=False
            )
        
        # Reply to the original message with sources
        await reaction.message.reply(embed=embed, mention_author=False)

    @ask.error
    async def ask_error(self, ctx, error):
        """Handle errors for the ask command."""
        if isinstance(error, commands.CommandOnCooldown):
            # Custom cooldown message
            minutes, seconds = divmod(int(error.retry_after), 60)

            embed = discord.Embed(
                title="⏰ Rate Limit Reached",
                description=(
                    f"You're asking questions too quickly!\n\n"
                    f"Please wait **{minutes}m {seconds}s** before trying again.\n\n"
                    f"This helps prevent API cost overruns and ensures fair usage."
                ),
                color=discord.Color.orange()
            )
            embed.set_footer(text="Limit: 5 questions per minute")

            await ctx.send(embed=embed)
        else:
            # Re-raise other errors
            raise error

    @ask_hybrid.error
    async def ask_hybrid_error(self, ctx, error):
        """Handle errors for the ask_hybrid command."""
        if isinstance(error, commands.CommandOnCooldown):
            minutes, seconds = divmod(int(error.retry_after), 60)

            embed = discord.Embed(
                title="⏰ Rate Limit Reached",
                description=(
                    f"Hybrid search is expensive!\n\n"
                    f"Please wait **{minutes}m {seconds}s** before trying again.\n\n"
                    f"Consider using regular `!ask` for faster queries."
                ),
                color=discord.Color.orange()
            )
            embed.set_footer(text="Limit: 3 hybrid questions per minute")

            await ctx.send(embed=embed)
        else:
            raise error

async def setup(bot):
    await bot.add_cog(RAG(bot))