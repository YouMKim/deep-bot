"""
Basic commands cog for the Discord bot.
Contains simple utility and fun commands.
"""

import discord
from discord.ext import commands
import asyncio
from ai.service import AIService
from ai.tracker import UserAITracker


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
    
    @commands.command(name='help', help='Show available commands for regular users')
    async def help_command(self, ctx):
        """Show available commands for regular users."""
        embed = discord.Embed(
            title="📚 Bot Commands",
            description="Available commands for regular users",
            color=discord.Color.blue()
        )
        
        # Get all commands and filter out admin-only ones
        admin_commands = {
            'whoami', 'check_blacklist', 'reload_blacklist', 'load_channel',
            'check_storage', 'checkpoint_info', 'chunk_status', 'reset_chunk_checkpoint',
            'rechunk', 'rag_settings', 'rag_set', 'rag_reset', 'rag_enable_all',
            'compare_rag', 'ai_provider', 'chatbot_mode'
        }
        
        # Group commands by cog
        commands_by_cog = {}
        for command in self.bot.commands:
            # Skip admin commands and the help_admin command
            if command.name in admin_commands or command.name == 'help_admin':
                continue
            
            # Skip if command is owner-only
            if command.checks:
                is_owner_only = any(
                    hasattr(check, '__name__') and 'is_owner' in str(check)
                    for check in command.checks
                )
                if is_owner_only:
                    continue
            
            cog_name = command.cog.qualified_name if command.cog else "Other"
            if cog_name not in commands_by_cog:
                commands_by_cog[cog_name] = []
            commands_by_cog[cog_name].append(command)
        
        # Add commands to embed
        for cog_name in sorted(commands_by_cog.keys()):
            commands_list = commands_by_cog[cog_name]
            if commands_list:
                value = "\n".join([
                    f"`{self.bot.command_prefix}{cmd.name}` - {cmd.help or 'No description'}"
                    for cmd in sorted(commands_list, key=lambda x: x.name)
                ])
                embed.add_field(
                    name=cog_name,
                    value=value,
                    inline=False
                )
        
        embed.set_footer(text=f"Use {self.bot.command_prefix}help_admin for admin commands")
        await ctx.send(embed=embed)
    
    @commands.command(name='help_admin', help='Show available admin commands (Admin only)')
    @commands.is_owner()
    async def help_admin(self, ctx):
        """Show available admin commands."""
        embed = discord.Embed(
            title="🔐 Admin Commands",
            description="Available commands for bot administrators",
            color=discord.Color.red()
        )
        
        # Admin commands grouped by category
        admin_categories = {
            "Bot Management": [
                'whoami',
                'ai_provider',
            ],
            "Blacklist": [
                'check_blacklist',
                'reload_blacklist',
            ],
            "Message Loading": [
                'load_channel',
                'check_storage',
            ],
            "Chunking": [
                'chunk_status',
                'checkpoint_info',
                'reset_chunk_checkpoint',
                'rechunk',
            ],
            "RAG Settings": [
                'rag_settings',
                'rag_set',
                'rag_reset',
                'rag_enable_all',
                'compare_rag',
            ],
            "Chatbot": [
                'chatbot_mode',
            ],
        }
        
        for category, command_names in admin_categories.items():
            commands_list = []
            for cmd_name in command_names:
                cmd = self.bot.get_command(cmd_name)
                if cmd:
                    commands_list.append(f"`{self.bot.command_prefix}{cmd.name}` - {cmd.help or 'No description'}")
            
            if commands_list:
                embed.add_field(
                    name=category,
                    value="\n".join(commands_list),
                    inline=False
                )
        
        embed.set_footer(text=f"Use {self.bot.command_prefix}help for regular user commands")
        await ctx.send(embed=embed)


async def setup(bot):
    """Load the cog."""
    await bot.add_cog(Basic(bot))
