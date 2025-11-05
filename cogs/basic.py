"""
Basic commands cog for the Discord bot.
Contains simple utility and fun commands.
"""

import discord
from discord.ext import commands
import asyncio
from services.ai_service import AIService
from services.user_ai_tracker import UserAITracker


class Basic(commands.Cog):
    """Basic commands for the bot."""

    def __init__(self, bot):
        self.bot = bot
        self.ai_service = AIService(provider_name="openai")
        self.ai_tracker = UserAITracker()

    @commands.command(name="ping", help="Check bot latency")
    async def ping(self, ctx):
        """Check the bot's latency."""
        latency = round(self.bot.latency * 1000)
        await ctx.send(f"🏓 Pong! Latency: {latency}ms")

    @commands.command(name="hello", help="Say hello to the bot")
    async def hello(self, ctx):
        """Say hello to the bot."""
        user_name = ctx.author.display_name
        
        try:
            prompt = f"Say hello back to {user_name} in a casual way but make sure to make user feel bad for wasting electricity and resources for this simple greeting. Keep it short (1-2 sentences)."
            
            result = await self.ai_service.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=1.7,
            )
            
            embed = discord.Embed(
                title="Hello",
                description=result["content"],
                color=discord.Color.blue()
            )
            
            embed.set_footer(text=f"Cost: ${result['cost']:.6f} | Model: {result['model']}")
            
            await ctx.send(embed=embed)
            
            # Track AI usage (simple - just cost and tokens)
            self.ai_tracker.log_ai_usage(
                user_display_name=user_name,
                cost=result['cost'],
                tokens_total=result['tokens_total']
            )
            
        except Exception as e:
            await ctx.send(f"👋 Hello {ctx.author.mention}! (AI greeting failed: {e})")

    @commands.command(name="info", help="Get bot information")
    async def info(self, ctx):
        """Get bot information."""
        embed = discord.Embed(
            title="🤖 Bot Information",
            description="A Discord bot built with discord.py",
            color=discord.Color.blue(),
        )
        embed.add_field(name="Guilds", value=len(self.bot.guilds), inline=True)
        embed.add_field(name="Users", value=len(self.bot.users), inline=True)
        embed.add_field(
            name="Latency", value=f"{round(self.bot.latency * 1000)}ms", inline=True
        )
        embed.add_field(name="Prefix", value=self.bot.command_prefix, inline=True)
        embed.add_field(name="Debug Mode", value=self.bot.debug_mode, inline=True)

        await ctx.send(embed=embed)

    @commands.command(name="echo", help="Echo a message")
    async def echo(self, ctx, *, message):
        """Echo a message."""
        await ctx.send(f"📢 {message}")

    @commands.command(name="server", help="Get server information")
    @commands.guild_only()
    async def server(self, ctx):
        """Get information about the current server."""
        guild = ctx.guild

        embed = discord.Embed(title=f"📊 {guild.name}", color=discord.Color.green())
        embed.add_field(name="Members", value=guild.member_count, inline=True)
        embed.add_field(name="Channels", value=len(guild.channels), inline=True)
        embed.add_field(name="Roles", value=len(guild.roles), inline=True)
        embed.add_field(
            name="Created", value=guild.created_at.strftime("%Y-%m-%d"), inline=True
        )
        embed.add_field(
            name="Owner",
            value=guild.owner.mention if guild.owner else "Unknown",
            inline=True,
        )

        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)

        await ctx.send(embed=embed)
    
    @commands.command(name="mystats", help="View your AI usage stats and social credit")
    async def mystats(self, ctx):
        user_name = ctx.author.display_name
        stats = self.ai_tracker.get_user_stats(user_name)
        
        if not stats:
            await ctx.send("No Clanker usage recorded yet!")
            return
        
        embed = discord.Embed(
            title=f"📊 Clanker Stats for {user_name}",
            color=discord.Color.green()
        )
        
        # Social Credit
        embed.add_field(
            name="Social Credit Scores",
            value=f"{stats['lifetime_credit']:.1f} points",
            inline=True
        )
        
        # Lifetime Spending
        embed.add_field(
            name="Money Wasted So Far",
            value=f"${stats['lifetime_cost']:.6f}",
            inline=True
        )
        
        # Lifetime Tokens
        embed.add_field(
            name="Tokens Used So Far",
            value=f"{stats['lifetime_tokens']:,}",
            inline=True
        )
        
        await ctx.send(embed=embed)

async def setup(bot):
    """Load the cog."""
    await bot.add_cog(Basic(bot))
