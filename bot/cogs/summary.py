import discord
from discord.ext import commands
from discord.ext.commands import cooldown, BucketType
import asyncio
from ai.service import AIService
from storage.chunked_memory import ChunkedMemoryService
from storage.messages.messages import MessageStorage
from bot.loaders.message_loader import MessageLoader
from bot.utils.discord_utils import format_discord_message
from typing import List
import logging


class Summary(commands.Cog):

    def __init__(self, bot, ai_provider: str = "openai"):
        """
        Initialize the Summary cog.
        
        Args:
            bot: The Discord bot instance
            ai_provider: The AI provider to use ("openai" or "anthropic")
        """
        from config import Config
        
        self.bot = bot
        self.ai_service = AIService(provider_name=ai_provider)
        self.config = Config
        self.chunked_memory_service = ChunkedMemoryService(config=self.config)
        self.message_storage = MessageStorage()
        self.message_loader = MessageLoader(self.message_storage, config=self.config)
        self.logger = logging.getLogger(__name__)


    @commands.command(name="summary", help="Generate a summary of recent messages from local storage")
    @cooldown(rate=3, per=120, type=BucketType.user)  # 3 per 2 minutes
    async def summary(self, ctx, count: int = 50):
        """
        Generate a summary of recent messages.
        Uses SQLite cache first, fetches from Discord if needed.
        
        Rate limit: 3 summaries per 2 minutes per user.
        """
        status_msg = await ctx.send("🔍 Checking local storage...")
        
        channel_id = str(ctx.channel.id)
        
        # Step 1: Check how many messages exist in SQLite
        stored_messages = self.message_storage.get_recent_messages(channel_id, count)
        missing_count = 0
        
        # Step 2: If insufficient, fetch delta from Discord
        if len(stored_messages) < count:
            missing_count = count - len(stored_messages)
            await status_msg.edit(
                content=f"📥 Fetching {missing_count} missing messages from Discord..."
            )
            
            try:
                # Use MessageLoader to fetch and store missing messages
                await self.message_loader.load_channel_messages(
                    channel=ctx.channel,
                    limit=missing_count
                )
                
                # Re-fetch from SQLite to get complete set
                stored_messages = self.message_storage.get_recent_messages(channel_id, count)
                
                self.logger.info(
                    f"Fetched {missing_count} messages from Discord, "
                    f"now have {len(stored_messages)} in storage"
                )
            except Exception as e:
                self.logger.error(f"Error fetching missing messages: {e}", exc_info=True)
                await status_msg.edit(content=f"⚠️ Error fetching messages: {e}")
                return
        
        if not stored_messages:
            await status_msg.edit(content="❌ No messages found to summarize.")
            return
        
        await status_msg.edit(
            content=f"📊 Analyzing {len(stored_messages)} messages from local storage..."
        )
        
        # Step 3: Generate summary from stored messages
        formatted_messages = self._format_for_summary(stored_messages)
        
        try:
            results = await self.ai_service.compare_all_styles(formatted_messages)
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error generating summary: {e}")
            return
        
        await status_msg.delete()
        await self._send_summary_embeds(ctx, results, len(stored_messages))
        
        # Step 4: Optionally trigger background chunking for delta (if new messages were fetched)
        if len(stored_messages) == count and missing_count > 0:
            # Trigger background chunking (fire and forget)
            asyncio.create_task(self._background_chunk(channel_id))
    
    async def _background_chunk(self, channel_id: str):
        """Background task to chunk newly fetched messages."""
        try:
            await self.chunked_memory_service.ingest_channel(channel_id)
            self.logger.info(f"Background chunking complete for channel {channel_id}")
        except Exception as e:
            self.logger.error(f"Background chunking failed: {e}", exc_info=True)

    def _format_for_summary(self, messages: List[dict]) -> str:
        """
        Format messages for summary generation.
        Works with both Discord message format and SQLite storage format.
        """
        formatted = []
        for message in messages:
            # Handle both formats (timestamp vs created_at)
            timestamp = message.get("timestamp", message.get("created_at", ""))
            if timestamp:
                timestamp = timestamp.split("T")[0]
            
            # Handle both formats (author vs author_name vs author_display_name)
            author = (
                message.get("author") or 
                message.get("author_display_name") or 
                message.get("author_name") or 
                "Unknown"
            )
            
            content = message.get("content", "")
            formatted.append(f"{timestamp} - {author}: {content}")
        
        return "\n".join(formatted)

    async def _send_summary_embeds(self, ctx, results: dict, message_count: int):
        """Send beautiful embeds for each summary style."""
        # Main header embed
        header_embed = discord.Embed(
            title="📊 Summary Comparison Results",
            description=f"Analyzed **{message_count}** messages using three different AI prompt styles:",
            color=discord.Color.blue(),
            timestamp=discord.utils.utcnow(),
        )

        # Calculate total cost
        total_cost = sum(result["cost"] for result in results.values())
        header_embed.add_field(
            name="💰 Total Cost", value=f"${total_cost:.6f}", inline=True
        )

        header_embed.set_footer(text=f"Channel: #{ctx.channel.name}")
        await ctx.send(embed=header_embed)

        # Style colors and emojis
        style_config = {
            "generic": {"color": discord.Color.green(), "emoji": "📄"},
            "bullet_points": {"color": discord.Color.orange(), "emoji": "📋"},
            "headline": {"color": discord.Color.purple(), "emoji": "📰"},
        }

        # Send individual embeds for each style
        for style, result in results.items():
            config = style_config.get(
                style, {"color": discord.Color.dark_gray(), "emoji": "📝"}
            )

            embed = discord.Embed(
                title=f"{config['emoji']} {style.replace('_', ' ').title()} Summary",
                description=result["summary"],
                color=config["color"],
            )

            # Add statistics as fields
            embed.add_field(
                name="📊 Statistics",
                value=(
                    f"**Input Tokens:** {result['tokens_prompt']:,}\n"
                    f"**Output Tokens:** {result['tokens_completion']:,}\n"
                    f"**Total Tokens:** {result['tokens_total']:,}\n"
                    f"**Cost:** ${result['cost']:.6f}"
                ),
                inline=False,
            )

            embed.set_footer(text=f"Model: {result['model']}")
            await ctx.send(embed=embed)

    @commands.command(name='memory_search', help='Search through stored messages using vector search')
    async def memory_search(self, ctx, *, query: str):
        """
        Search through stored messages using vector similarity search.
        Uses the active strategy (default: tokens).
        Use !set_search_strategy to change which strategy to search.
        """
        try:
            active_strategy = self.chunked_memory_service.active_strategy
            status_msg = await ctx.send(
                f"🔍 Searching for: **{query}** (using `{active_strategy}` strategy)..."
            )
            
            # Use vector search with the active strategy
            results = self.chunked_memory_service.search(
                query=query,
                top_k=5
            )
            
            if not results:
                await status_msg.edit(content="No relevant messages found in this channel.")
                return
            
            # Create rich embed
            embed = discord.Embed(
                title=f"🔍 Search Results: {query}",
                description=f"Found {len(results)} relevant chunks",
                color=discord.Color.blue()
            )
            
            for i, result in enumerate(results, 1):
                content = result['content']
                metadata = result.get('metadata', {})
                similarity = result.get('similarity', 0)
                
                # Truncate content if too long
                display_content = content[:200] + "..." if len(content) > 200 else content
                
                embed.add_field(
                    name=f"{i}. Chunk from {metadata.get('channel_id', 'Unknown')}",
                    value=(
                        f"{display_content}\n"
                        f"*Strategy: {metadata.get('chunk_strategy', 'unknown')} | "
                        f"Similarity: {similarity:.2f}*"
                    ),
                    inline=False
                )
            
            await status_msg.edit(content="", embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in memory search: {e}", exc_info=True)
            await ctx.send(f"❌ Error searching messages: {e}")
    
    @commands.command(name='set_search_strategy', help='Set which chunking strategy to use for searches')
    async def set_search_strategy(self, ctx, strategy: str = None):
        """
        Set which chunking strategy to use for vector searches.
        
        Usage:
            !set_search_strategy - Show current strategy
            !set_search_strategy tokens - Use tokens strategy
            !set_search_strategy single - Use single strategy
        """
        if strategy is None:
            # Show current strategy
            current = self.chunked_memory_service.active_strategy
            await ctx.send(f"🔍 Current search strategy: **{current}**")
            return
        
        try:
            from chunking.constants import ChunkStrategy
            new_strategy = ChunkStrategy(strategy.lower())
            self.chunked_memory_service.set_active_strategy(new_strategy)
            
            await ctx.send(
                f"✅ Search strategy changed to: **{new_strategy.value}**\n"
                f"Future `!memory_search` commands will use this strategy."
            )
        except ValueError:
            from chunking.constants import ChunkStrategy
            valid_strategies = ", ".join([s.value for s in ChunkStrategy])
            await ctx.send(
                f"❌ Invalid strategy: `{strategy}`\n"
                f"Valid strategies: {valid_strategies}"
            )
    
    @summary.error
    async def summary_error(self, ctx, error):
        """Handle errors for the summary command."""
        if isinstance(error, commands.CommandOnCooldown):
            minutes, seconds = divmod(int(error.retry_after), 60)

            embed = discord.Embed(
                title="⏰ Rate Limit Reached",
                description=(
                    f"Summary generation is expensive!\n\n"
                    f"Please wait **{minutes}m {seconds}s** before trying again.\n\n"
                    f"This helps prevent API cost overruns."
                ),
                color=discord.Color.orange()
            )
            embed.set_footer(text="Limit: 3 summaries per 2 minutes")

            await ctx.send(embed=embed)
        else:
            raise error
    
async def setup(bot):
    await bot.add_cog(Summary(bot))
