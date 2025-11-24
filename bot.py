"""
Main Discord bot file.
This is the entry point for your Discord bot.
"""

import asyncio
import logging
import sys
from datetime import datetime, time, timezone
import discord
from discord.ext import commands
from discord.ext.commands import (
    CommandNotFound,
    MissingRequiredArgument,
    BadArgument,
    CommandOnCooldown,
)
from config import Config
from bot.cronjob_tasks import CronjobTasks

# Set up logging
# In Railway/Docker, file logging may not persist, so we primarily use stdout
# Railway captures stdout/stderr automatically
log_handlers = [logging.StreamHandler(sys.stdout)]
try:
    # Try to add file handler, but don't fail if it doesn't work (e.g., in containers)
    log_handlers.append(logging.FileHandler("bot.log"))
except (PermissionError, OSError):
    # File logging not available (e.g., read-only filesystem in some containers)
    pass

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
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
            help_command=None,  # Disable default help command, we'll use custom one
        )
        self.debug_mode = Config.DEBUG_MODE
        self.cronjob_tasks = None
        self.cronjob_task = None

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
        
        # Start cronjob scheduler if enabled and not already running
        if Config.CRONJOB_ENABLED:
            if self.cronjob_task is None or self.cronjob_task.done():
                self.cronjob_tasks = CronjobTasks(self)
                self.cronjob_task = asyncio.create_task(self._cronjob_scheduler())
                logger.info(f"Started cronjob scheduler (runs daily at {Config.CRONJOB_SCHEDULE_TIME} UTC)")
        else:
            logger.info("Cronjob scheduler is disabled (set CRONJOB_ENABLED=true to enable)")

    async def _cronjob_scheduler(self):
        """Background task that runs cronjob tasks daily at configured time (UTC)."""
        await self.wait_until_ready()
        
        # Parse schedule time from config (format: "HH:MM" or "H:MM")
        schedule_time_str = Config.CRONJOB_SCHEDULE_TIME
        try:
            # Parse time string (e.g., "06:00" or "6:00")
            time_parts = schedule_time_str.split(":")
            if len(time_parts) != 2:
                raise ValueError(f"Invalid time format: {schedule_time_str}")
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(f"Invalid time: hour must be 0-23, minute must be 0-59")
            target_time = time(hour, minute, 0)
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid CRONJOB_SCHEDULE_TIME format '{schedule_time_str}'. Expected HH:MM (e.g., '14:00'). Using default 14:00 (6 AM Pacific). Error: {e}")
            target_time = time(14, 0, 0)  # Default to 14:00 UTC (6 AM Pacific)
        
        while not self.is_closed():
            try:
                # Calculate next run time
                now = datetime.now(timezone.utc)
                
                # If it's already past 6 AM today, schedule for tomorrow
                if now.time() >= target_time:
                    # Schedule for tomorrow
                    from datetime import timedelta
                    next_run = datetime.combine(
                        (now + timedelta(days=1)).date(), target_time, timezone.utc
                    )
                else:
                    next_run = datetime.combine(
                        now.date(), target_time, timezone.utc
                    )
                
                # Calculate seconds until next run
                wait_seconds = (next_run - now).total_seconds()
                
                logger.info(
                    f"Cronjob scheduler: Next run in {wait_seconds/3600:.1f} hours "
                    f"({next_run.strftime('%Y-%m-%d %H:%M:%S UTC')})"
                )
                
                # Wait until next run time
                await asyncio.sleep(wait_seconds)
                
                # Run cronjob tasks
                logger.info("Running scheduled cronjob tasks...")
                try:
                    await self.cronjob_tasks.run_all_tasks()
                except Exception as e:
                    logger.error(f"Error running cronjob tasks: {e}", exc_info=True)
                
            except asyncio.CancelledError:
                logger.info("Cronjob scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cronjob scheduler: {e}", exc_info=True)
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)

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
        
        # Bot knowledge will be initialized automatically when ChunkedMemoryService
        # instances are created by the cogs (RAG, Summary, Chatbot)
        # Each instance will initialize bot knowledge in the background if not already indexed
        
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
        try:
            await self.load_extension("bot.cogs.social_credit_commands")
            logger.info("Loaded social credit commands cog")
        except Exception as e:
            logger.error(f"Failed to load social credit commands cog: {e}")


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
