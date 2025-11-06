import discord
from discord.ext import commands
import asyncio
import logging
from services.memory_service import MemoryService
from services.message_loader import MessageLoader
from services.ai_service import AIService
from utils.discord_utils import format_discord_message
from typing import List


class Admin(commands.Cog):
    """Admin-only commands for managing the bot and loading messages."""

    def __init__(self, bot):
        self.bot = bot
        self.memory_service = MemoryService()
        self.message_loader = MessageLoader(self.memory_service)
        self.logger = logging.getLogger(__name__)
        self.ai_service = None

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

    @commands.command(name='check_blacklist', help='Check the current blacklist configuration')
    async def check_blacklist(self, ctx):
        """Check the current blacklist configuration"""
        from config import Config
        import os
        
        embed = discord.Embed(
            title="🚫 Blacklist Configuration",
            color=discord.Color.orange()
        )
        
        # Show raw environment variable
        raw_env = os.getenv("BLACKLIST_IDS", "")
        embed.add_field(
            name="Raw ENV Variable (BLACKLIST_IDS)",
            value=f"`{raw_env if raw_env else '(not set)'}`",
            inline=False
        )
        
        if Config.BLACKLIST_IDS:
            blacklist_str = "\n".join([f"• `{user_id}`" for user_id in Config.BLACKLIST_IDS])
            embed.add_field(
                name=f"Blacklisted User IDs ({len(Config.BLACKLIST_IDS)})",
                value=blacklist_str,
                inline=False
            )
            
            # Check if current author is blacklisted
            is_blacklisted = ctx.author.id in Config.BLACKLIST_IDS
            embed.add_field(
                name="Are you blacklisted?",
                value="✅ Yes (you are blacklisted)" if is_blacklisted else "❌ No (you are not blacklisted)",
                inline=False
            )
        else:
            embed.add_field(
                name="Loaded Blacklist Status",
                value="❌ No blacklisted users loaded into Config.BLACKLIST_IDS",
                inline=False
            )
        
        embed.add_field(
            name="Your User ID",
            value=f"`{ctx.author.id}` (type: {type(ctx.author.id).__name__})",
            inline=False
        )
        
        if Config.BLACKLIST_IDS:
            embed.add_field(
                name="Blacklist Types",
                value=f"[{', '.join([type(x).__name__ for x in Config.BLACKLIST_IDS[:3]])}]",
                inline=False
            )
        
        await ctx.send(embed=embed)

    @commands.command(name='reload_blacklist', help='Reload the blacklist from environment variables (Admin only)')
    async def reload_blacklist(self, ctx):
        """Reload the blacklist from environment variables"""
        from config import Config
        import os
        
        # Manual owner check
        if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
            await ctx.send("🚫 **Access Denied!** Only the bot admin can reload the blacklist.")
            return
        
        try:
            # Show before state
            before_count = len(Config.BLACKLIST_IDS)
            raw_env = os.getenv("BLACKLIST_IDS", "")
            
            # Reload
            Config.load_blacklist()
            
            # Show after state
            after_count = len(Config.BLACKLIST_IDS)
            
            embed = discord.Embed(
                title="🔄 Blacklist Reload",
                color=discord.Color.green()
            )
            
            embed.add_field(
                name="Raw ENV Variable",
                value=f"`{raw_env if raw_env else '(not set)'}`",
                inline=False
            )
            
            embed.add_field(
                name="Before Reload",
                value=f"{before_count} user(s)",
                inline=True
            )
            
            embed.add_field(
                name="After Reload",
                value=f"{after_count} user(s)",
                inline=True
            )
            
            if Config.BLACKLIST_IDS:
                blacklist_str = "\n".join([f"• `{user_id}`" for user_id in Config.BLACKLIST_IDS])
                embed.add_field(
                    name="Loaded User IDs",
                    value=blacklist_str,
                    inline=False
                )
            else:
                embed.add_field(
                    name="⚠️ Warning",
                    value="No blacklist IDs loaded. Check your .env file.",
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"❌ Error reloading blacklist: {e}")

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
    
    @commands.command(name='ai_provider', help='Switch AI provider (Admin only)')
    async def ai_provider(self, ctx, provider: str = None):
        """
        Get or set the AI provider. (Admin only)
        
        Usage:
            !ai_provider - Show current provider
            !ai_provider openai - Switch to OpenAI
            !ai_provider anthropic - Switch to Anthropic
        """
        from config import Config
        
        # Manual owner check
        if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
            await ctx.send("🚫 **Access Denied!** Only the bot admin can change AI provider.")
            return
        
        # Get AI service from Summary cog
        summary_cog = self.bot.get_cog("Summary")
        if not summary_cog:
            await ctx.send("❌ Summary cog not loaded. Cannot access AI service.")
            return
        
        if provider is None:
            # Show current provider
            embed = discord.Embed(
                title="🤖 Current AI Provider",
                description=f"Currently using: **{summary_cog.ai_service.provider_name}**",
                color=discord.Color.blue()
            )
            
            # Add info about available providers
            embed.add_field(
                name="Available Providers",
                value="• `openai` - GPT models (fast, versatile)\n• `anthropic` - Claude models (advanced reasoning)",
                inline=False
            )
            
            # Show default model
            default_model = summary_cog.ai_service.provider.get_default_model()
            embed.add_field(
                name="Default Model",
                value=default_model,
                inline=True
            )
            
            await ctx.send(embed=embed)
            return
        
        # Validate provider
        if provider.lower() not in ["openai", "anthropic"]:
            await ctx.send("❌ Invalid provider. Use `openai` or `anthropic`")
            return
        
        # Switch provider
        try:
            summary_cog.ai_service = AIService(provider_name=provider.lower())
            
            embed = discord.Embed(
                title="✅ Provider Switched",
                description=f"Now using: **{provider}**",
                color=discord.Color.green()
            )
            
            # Get default model info
            default_model = summary_cog.ai_service.provider.get_default_model()
            embed.add_field(
                name="Default Model",
                value=default_model,
                inline=True
            )
            
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Error switching provider: {e}")

    @commands.command(name='ai_model', help='Switch AI model within current provider (Admin only)')
    async def ai_model(self, ctx, model: str = None):
        """
        Get or set the AI model. (Admin only)

        Usage:
            !ai_model - Show current model and available models
            !ai_model gpt-4o - Switch to GPT-4o
            !ai_model claude-3-5-sonnet - Switch to Claude 3.5 Sonnet
        """
        from config import Config

        # Manual owner check
        if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
            await ctx.send("🚫 **Access Denied!** Only the bot admin can change AI models.")
            return

        # Get AI service from Summary cog
        summary_cog = self.bot.get_cog("Summary")
        if not summary_cog:
            await ctx.send("❌ Summary cog not loaded. Cannot access AI service.")
            return

        if model is None:
            # Show current model and available models
            current_model = summary_cog.ai_service.get_current_model()
            available_models = summary_cog.ai_service.get_available_models()
            provider_name = summary_cog.ai_service.provider_name

            embed = discord.Embed(
                title=f"🤖 AI Models for {provider_name.title()}",
                description=f"**Current Model:** `{current_model}`",
                color=discord.Color.blue()
            )

            # Group models by category for better display
            if provider_name == "openai":
                model_groups = {
                    "o1 Series (Reasoning)": [m for m in available_models if m.startswith("o1")],
                    "GPT-4o Series": [m for m in available_models if m.startswith("gpt-4o") and "mini" not in m],
                    "GPT-4o Mini": [m for m in available_models if "4o-mini" in m],
                    "GPT-4 Turbo": [m for m in available_models if "4-turbo" in m],
                    "GPT-4": [m for m in available_models if m.startswith("gpt-4") and "turbo" not in m and "4o" not in m],
                    "GPT-3.5": [m for m in available_models if "3.5" in m],
                }
            else:  # anthropic
                model_groups = {
                    "Claude 3.5 Sonnet": [m for m in available_models if "3-5-sonnet" in m],
                    "Claude 3.5 Haiku": [m for m in available_models if "3-5-haiku" in m],
                    "Claude 3 Opus": [m for m in available_models if "3-opus" in m],
                    "Claude 3 Sonnet": [m for m in available_models if "3-sonnet" in m and "3-5" not in m],
                    "Claude 3 Haiku": [m for m in available_models if "3-haiku" in m and "3-5" not in m],
                }

            for category, models in model_groups.items():
                if models:
                    # Get pricing for the first model in category
                    pricing = summary_cog.ai_service.provider.PRICING_TABLE.get(models[0], {})
                    price_text = f"${pricing.get('prompt', 0):.2f}/${pricing.get('completion', 0):.2f} per 1M tokens"

                    model_list = "\n".join([f"• `{m}`" for m in models[:3]])  # Show max 3 models per category
                    if len(models) > 3:
                        model_list += f"\n• ... and {len(models) - 3} more"

                    embed.add_field(
                        name=f"{category} ({price_text})",
                        value=model_list,
                        inline=False
                    )

            embed.set_footer(text="Use !ai_model <model_name> to switch models")
            await ctx.send(embed=embed)
            return

        # Switch model
        try:
            summary_cog.ai_service.switch_model(model)

            # Get pricing info
            pricing = summary_cog.ai_service.provider.PRICING_TABLE.get(model, {})
            price_text = f"${pricing.get('prompt', 0):.2f}/${pricing.get('completion', 0):.2f} per 1M tokens"

            embed = discord.Embed(
                title="✅ Model Switched",
                description=f"Now using: **{model}**",
                color=discord.Color.green()
            )

            embed.add_field(
                name="Provider",
                value=summary_cog.ai_service.provider_name.title(),
                inline=True
            )

            embed.add_field(
                name="Pricing",
                value=price_text,
                inline=True
            )

            await ctx.send(embed=embed)
        except ValueError as e:
            await ctx.send(f"❌ {str(e)}")
        except Exception as e:
            await ctx.send(f"❌ Error switching model: {e}")

async def setup(bot):
    await bot.add_cog(Admin(bot))
