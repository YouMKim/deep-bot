"""
Basic commands cog for the Discord bot.
Contains simple utility and fun commands.
"""

import discord
from discord.ext import commands
import asyncio


class Basic(commands.Cog):
    """Basic commands for the bot."""

    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="ping", help="Check bot latency")
    async def ping(self, ctx):
        """Check the bot's latency."""
        latency = round(self.bot.latency * 1000)
        await ctx.send(f"🏓 Pong! Latency: {latency}ms")

    @commands.command(name="hello", help="Say hello to the bot")
    async def hello(self, ctx):
        """Say hello to the bot."""
        try:
            audio_file = discord.File(
                "audio/Blitzcrank_Original_Taunt.ogg", filename=""
            )
            await ctx.send(f"👋 Hello {ctx.author.mention}!", file=audio_file)
        except FileNotFoundError:
            await ctx.send(f"👋 Hello {ctx.author.mention}! (Audio file not found)")
        except Exception as e:
            await ctx.send(f"👋 Hello {ctx.author.mention}! (Error loading audio: {e})")

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


async def setup(bot):
    """Load the cog."""
    await bot.add_cog(Basic(bot))
