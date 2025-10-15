import discord
from discord.ext import commands
import asyncio
from services.ai_service import AIService
from typing import List

class Summary(commands.Cog):

    def __init__(self, bot):
        self.bot = bot
        self.ai_service = AIService()

    @commands.command(name='summary', help= 'Generate a summary of previous 50 messages')
    async def summary(self, ctx):
        status_msg = await ctx.send("🔍 Fetching messages...")
        messages = await self._fetch_messages(ctx)
        
        if not messages:
            await status_msg.edit(content="❌ No messages found to summarize.")
            return
        
        await status_msg.edit(content=f" Analyzing {len(messages)} messages...")
        formatted_messages = self._format_for_summary(messages)
        results = await self.ai_service.compare_all_styles(formatted_messages) 
        
        await status_msg.delete()
        
        await self._send_summary_embeds(ctx, results, len(messages))


    async def _fetch_messages(self, ctx, count = 50) -> List[dict]:
        messages = [] 
        async for message in ctx.channel.history(limit=count):
            if message.author.bot or not message.content.strip():
                continue 
            if message.content.startswith(ctx.prefix):
                continue 
        
            messages.append({
                'content': message.content,
                'author': message.author.name,
                'timestamp': message.created_at.isoformat(),
                'id': message.id,
            })
        return list(reversed(messages))
    
    def _format_for_summary(self, messages: List[dict]) -> str:
        formatted = []
        for message in messages:
            timestamp = message['timestamp'].split('T')[0]
            formatted.append(f"{timestamp} - {message['author']}: {message['content']}")
        return "\n".join(formatted)
    
    async def _send_summary_embeds(self, ctx, results: dict, message_count: int):
        """Send beautiful embeds for each summary style."""
        # Main header embed
        header_embed = discord.Embed(
            title="📊 Summary Comparison Results",
            description=f"Analyzed **{message_count}** messages using three different AI prompt styles:",
            color=discord.Color.blue(),
            timestamp=discord.utils.utcnow()
        )
        
        # Calculate total cost
        total_cost = sum(result['cost'] for result in results.values())
        header_embed.add_field(
            name="💰 Total Cost", 
            value=f"${total_cost:.6f}", 
            inline=True
        )
        
        header_embed.set_footer(text=f"Channel: #{ctx.channel.name}")
        await ctx.send(embed=header_embed)
        
        # Style colors and emojis
        style_config = {
            'generic': {'color': discord.Color.green(), 'emoji': '📄'},
            'bullet_points': {'color': discord.Color.orange(), 'emoji': '📋'},
            'headline': {'color': discord.Color.purple(), 'emoji': '📰'}
        }
        
        # Send individual embeds for each style
        for style, result in results.items():
            config = style_config.get(style, {'color': discord.Color.dark_gray(), 'emoji': '📝'})
            
            embed = discord.Embed(
                title=f"{config['emoji']} {style.replace('_', ' ').title()} Summary",
                description=result['summary'],
                color=config['color']
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
                inline=False
            )
            
            embed.set_footer(text=f"Model: {result['model']}")
            await ctx.send(embed=embed)

async def setup(bot):
    await bot.add_cog(Summary(bot))