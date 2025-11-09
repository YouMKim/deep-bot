import discord
from discord.ext import commands
import logging
from rag.pipeline import RAGPipeline
from rag.models import RAGConfig, RAGResult
from config import Config

class RAG(commands.Cog):
    
    def __init__(self, bot):
        self.bot = bot
        self.pipeline = RAGPipeline()
        self.logger = logging.getLogger(__name__)

    @commands.command(
        name='ask',
        help='Ask a question about Discord conversations. Mention users to filter to their messages.'
    )
    async def ask(self, ctx, *, question: str):
        """
        Simple RAG command for all users.
        
        Usage: 
            !ask What was decided about the database?
            !ask @Alice what did you say about the schema?
        """
        async with ctx.typing():
            self.logger.info(f"User {ctx.author} asked: {question}")
            
            # Parse mentioned users from the message
            mentioned_users = []
            if ctx.message.mentions:
                mentioned_users = [user.display_name for user in ctx.message.mentions]
                self.logger.info(f"Filtering to authors: {mentioned_users}")
            
            config = RAGConfig(
                top_k=Config.RAG_DEFAULT_TOP_K,
                similarity_threshold=Config.RAG_DEFAULT_SIMILARITY_THRESHOLD,
                max_context_tokens=Config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
                temperature=Config.RAG_DEFAULT_TEMPERATURE,
                strategy=Config.RAG_DEFAULT_STRATEGY,
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

async def setup(bot):
    await bot.add_cog(RAG(bot))