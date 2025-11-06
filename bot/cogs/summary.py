import discord
from discord.ext import commands
import asyncio
from ai.service import AIService
from storage.memory import MemoryService
from bot.utils.discord_utils import format_discord_message
from typing import List


class Summary(commands.Cog):

    def __init__(self, bot, ai_provider: str = "openai"):
        """
        Initialize the Summary cog.
        
        Args:
            bot: The Discord bot instance
            ai_provider: The AI provider to use ("openai" or "anthropic")
        """
        self.bot = bot
        self.ai_service = AIService(provider_name=ai_provider)
        self.memory_service = MemoryService()


    @commands.command(name="summary", help="Generate a summary of previous 50 messages and store in memory")
    async def summary(self, ctx, count: int = 50):
        status_msg = await ctx.send("🔍 Fetching messages...")
        messages = await self._fetch_messages(ctx, count)

        if not messages:
            await status_msg.edit(content="❌ No messages found to summarize.")
            return

        # Store messages in memory
        stored_count = 0
        for message in messages:
            # messages is already a list of dicts from _fetch_messages
            # No need to call format_discord_message again
            success = await self.memory_service.store_message(message)
            if success:
                stored_count += 1

        await status_msg.edit(content=f"📊 Analyzing {len(messages)} messages (stored {stored_count})...")
        
        # Generate summary (existing logic)
        formatted_messages = self._format_for_summary(messages)
        results = await self.ai_service.compare_all_styles(formatted_messages)

        await status_msg.delete()
        await self._send_summary_embeds(ctx, results, len(messages))

    async def _fetch_messages(self, ctx, count=50) -> List[dict]:
        messages = []
        async for message in ctx.channel.history(limit=count):
            if message.author.bot or not message.content.strip():
                continue
            if message.content.startswith(ctx.prefix):
                continue

            messages.append(
                {
                    "content": message.content,
                    "author": message.author.name,
                    "timestamp": message.created_at.isoformat(),
                    "id": message.id,
                }
            )
        return list(reversed(messages))

    def _format_for_summary(self, messages: List[dict]) -> str:
        formatted = []
        for message in messages:
            timestamp = message["timestamp"].split("T")[0]
            formatted.append(f"{timestamp} - {message['author']}: {message['content']}")
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

    @commands.command(name='memory_search', help='Search through stored messages')
    async def memory_search(self, ctx, *, query: str):
        """Search through stored messages"""
        try:
            await ctx.send(f"🔍 Searching for: **{query}**...")
            
            # Search with channel filtering
            results = await self.memory_service.find_relevant_messages(
                query=query,
                limit=5,
                channel_id=str(ctx.channel.id)
            )
            
            if not results:
                await ctx.send("No relevant messages found in this channel.")
                return
            
            # Create rich embed
            embed = discord.Embed(
                title=f"🔍 Search Results: {query}",
                description=f"Found {len(results)} relevant messages",
                color=discord.Color.blue()
            )
            
            for i, result in enumerate(results, 1):
                embed.add_field(
                    name=f"{i}. {result['author_display_name']}",
                    value=f"{result['content'][:200]}...\n*Similarity: {result['similarity_score']:.2f}*",
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error searching messages: {e}")
    
async def setup(bot):
    await bot.add_cog(Summary(bot))
