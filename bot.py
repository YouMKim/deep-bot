"""
Main Discord bot file.
This is the entry point for your Discord bot.
"""

import asyncio
import logging
import sys
import discord
from discord.ext import commands
from discord.ext.commands import (
    CommandNotFound,
    MissingRequiredArgument,
    BadArgument,
    CommandOnCooldown,
)
from config import Config

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class DeepBot(commands.Bot):
    """Main bot class."""

    def __init__(self):
        Config.validate()

        # Convert owner_id to int if it exists
        owner_id = int(Config.BOT_OWNER_ID) if Config.BOT_OWNER_ID else None

        super().__init__(
            command_prefix=Config.BOT_PREFIX,
            intents=Config.get_discord_intents(),
            owner_id=owner_id,
            strip_after_prefix=True,
        )
        self.debug_mode = Config.DEBUG_MODE

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Bot has connected to Discord")
        logger.info(f"Bot has connected to {len(self.guilds)} guilds")
        logger.info(f"Bot has connected to {len(self.users)} users")

        # Set bot activity
        activity = discord.Activity(
            type=discord.ActivityType.listening, name=f"{Config.BOT_PREFIX}help"
        )
        await self.change_presence(activity=activity)

    async def on_command_error(self, ctx, error):
        """Handle command errors."""
        if isinstance(error, CommandNotFound):
            # Ignore command not found errors
            return

        elif isinstance(error, MissingRequiredArgument):
            await ctx.send(f"❌ Missing required argument: `{error.param.name}`")

        elif isinstance(error, BadArgument):
            await ctx.send(f"❌ Invalid argument: {error}")

        elif isinstance(error, CommandOnCooldown):
            await ctx.send(
                f"⏰ Command is on cooldown. Try again in {error.retry_after:.2f} seconds."
            )

        else:
            # Log the error and send a generic message
            logger.error(f"Command error in {ctx.command}: {error}", exc_info=True)
            await ctx.send("❌ An error occurred while executing the command.")

    async def setup_hook(self):
        """Called when the bot is starting up."""
        # Load configuration
        Config.load_blacklist()
        
        # Load cogs
        try:
            await self.load_extension("bot.cogs.basic")
            logger.info("Loaded basic cog")
        except Exception as e:
            logger.error(f"Failed to load basic cog: {e}")
        try:
            await self.load_extension("bot.cogs.summary")
            logger.info("Loaded summary cog")
        except Exception as e:
            logger.error(f"Failed to load summary cog: {e}")
        try:
            await self.load_extension("bot.cogs.admin")
            logger.info("Loaded admin cog")
        except Exception as e:
            logger.error(f"Failed to load admin cog: {e}")
        try:
            await self.load_extension("bot.cogs.rag")
            logger.info("Loaded rag cog")
        except Exception as e:
            logger.error(f"Failed to load rag cog: {e}")
        try:
            await self.load_extension("bot.cogs.chatbot")
            logger.info("Loaded chatbot cog")
        except Exception as e:
            logger.error(f"Failed to load chatbot cog: {e}")


async def main():
    """Main function to run the bot."""
    bot = DeepBot()
    try:
        await bot.start(Config.DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot has been stopped with keyboard interrupt")
        await bot.close()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        await bot.close()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
