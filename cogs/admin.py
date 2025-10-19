import discord
from discord.ext import commands
import asyncio
import logging
from services.memory_service import MemoryService
from services.message_loader import MessageLoader
from utils.discord_utils import format_discord_message
from typing import List


class Admin(commands.Cog):
    """Admin-only commands for managing the bot and loading messages."""

    def __init__(self, bot):
        self.bot = bot
        self.memory_service = MemoryService()
        self.message_loader = MessageLoader(self.memory_service)
        self.logger = logging.getLogger(__name__)

    async def cog_command_error(self, ctx, error):
        """Handle errors in admin commands"""
        if isinstance(error, commands.NotOwner):
            await ctx.send("🚫 **Access Denied!** You don't have permission to use admin commands. Only the bot admin can use these commands.")
        else:
            self.logger.error(f"Error in admin command: {error}")
            await ctx.send(f"❌ An error occurred: {error}")

    @commands.command(name='whoami', help='Check who the bot thinks is the owner')
    async def whoami(self, ctx):
        """Check who the bot thinks is the owner"""
        from config import Config
        
        embed = discord.Embed(
            title="🤖 Bot Owner Information",
            color=discord.Color.blue()
        )
        
        # Show configured owner ID
        embed.add_field(
            name="Configured Owner ID",
            value=f"`{Config.BOT_OWNER_ID}`",
            inline=False
        )
        
        # Show your user ID
        embed.add_field(
            name="Your User ID",
            value=f"`{ctx.author.id}`",
            inline=False
        )
        
        # Check if you're the owner
        is_owner = str(ctx.author.id) == str(Config.BOT_OWNER_ID)
        embed.add_field(
            name="Are you the owner?",
            value="✅ Yes" if is_owner else "❌ No",
            inline=False
        )
        
        # Show bot owner from Discord.py
        if self.bot.owner_id:
            embed.add_field(
                name="Bot Owner (Discord.py)",
                value=f"`{self.bot.owner_id}`",
                inline=False
            )
        else:
            embed.add_field(
                name="Bot Owner (Discord.py)",
                value="Not set",
                inline=False
            )
        
        await ctx.send(embed=embed)

    @commands.command(name='load_channel', help='Load all messages from current channel into memory (Admin only)')
    async def load_channel(self, ctx, limit: int = None):
        """Load all messages from the current channel into memory"""
        from config import Config
        
        # Manual owner check
        if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
            await ctx.send("🚫 **Access Denied!** You don't have permission to use admin commands. Only the bot admin can use these commands.")
            return
        
        try:
            if limit and limit > 10000:
                await ctx.send("❌ Limit cannot exceed 10,000 messages for safety.")
                return
            
            status_msg = await ctx.send(f"🔄 Loading messages from #{ctx.channel.name}...")

            stats = await self.message_loader.load_channel_messages(
                channel=ctx.channel,
                limit=limit
            )
            
            embed = discord.Embed(
                title="📥 Channel Loading Complete",
                description=f"Loaded messages from #{ctx.channel.name}",
                color=discord.Color.green()
            )
            
            embed.add_field(
                name="📊 Statistics",
                value=(
                    f"**Total Processed:** {stats['total_processed']}\n"
                    f"**Successfully Stored:** {stats['successfully_loaded']}\n"
                    f"**Bot Messages Skipped:** {stats['skipped_bot_messages']}\n"
                    f"**Blacklisted Users Skipped:** {stats['skipped_blacklisted']}\n"
                    f"**Empty Messages Skipped:** {stats['skipped_empty_messages']}\n"
                    f"**Commands Skipped:** {stats['skipped_commands']}\n"
                    f"**Errors:** {stats['errors']}"
                ),
                inline=False
            )
            
            if stats['end_time'] and stats['start_time']:
                duration = (stats['end_time'] - stats['start_time']).total_seconds()
                embed.add_field(
                    name="⏱️ Duration",
                    value=f"{duration:.1f} seconds",
                    inline=True
                )
            
            await status_msg.edit(content="", embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error loading channel messages: {e}")

async def setup(bot):
    await bot.add_cog(Admin(bot))
