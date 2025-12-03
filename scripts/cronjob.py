"""
Cronjob script for Railway deployment.

This script runs two main tasks:
1. Load_server: Loads all messages not yet ingested from Discord
2. Snapshot: Posts 5 messages from 1, 2, 3, 4, 5 years ago on this day
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import discord
from discord.ext import commands

from config import Config
from storage.messages import MessageStorage
from bot.loaders.message_loader import MessageLoader
from storage.chunked_memory import ChunkedMemoryService

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class CronjobBot(commands.Bot):
    """Minimal bot for running cronjob tasks."""
    
    def __init__(self):
        Config.validate()
        Config.load_blacklist()
        
        owner_id = int(Config.BOT_OWNER_ID) if Config.BOT_OWNER_ID else None
        
        super().__init__(
            command_prefix=Config.BOT_PREFIX,
            intents=Config.get_discord_intents(),
            owner_id=owner_id,
            strip_after_prefix=True,
            help_command=None,
        )
        
        self.message_storage = MessageStorage()
        self.message_loader = MessageLoader(self.message_storage, config=Config)
        self.logger = logger

    async def on_ready(self):
        """Called when the bot is ready."""
        self.logger.info(f"Cronjob bot connected to Discord")
        self.logger.info(f"Connected to {len(self.guilds)} guilds")
        
        # Run cronjob tasks
        try:
            await self.run_cronjob_tasks()
        except Exception as e:
            self.logger.error(f"Error running cronjob tasks: {e}", exc_info=True)
            sys.exit(1)
        finally:
            await self.close()

    async def run_cronjob_tasks(self):
        """Run all cronjob tasks."""
        self.logger.info("Starting cronjob tasks...")
        
        # Task 1: Load server (ingest all unprocessed messages)
        await self.load_server_task()
        
        # Task 2: Snapshot (post messages from years ago)
        await self.snapshot_task()
        
        self.logger.info("Cronjob tasks completed successfully")

    async def load_server_task(self):
        """Load all messages from all channels in the server."""
        self.logger.info("Starting load_server task...")
        
        try:
            # Get the first guild (assuming single guild deployment)
            if not self.guilds:
                self.logger.warning("No guilds found, skipping load_server task")
                return
            
            guild = self.guilds[0]
            self.logger.info(f"Processing guild: {guild.name}")
            
            # Get all text channels (excluding chatbot channel)
            chatbot_channel_id = Config.CHATBOT_CHANNEL_ID
            text_channels = [
                ch for ch in guild.text_channels 
                if ch.permissions_for(guild.me).read_message_history
                and ch.id != chatbot_channel_id
            ]
            
            if not text_channels:
                self.logger.warning("No accessible text channels found")
                return
            
            self.logger.info(f"Processing {len(text_channels)} channels")
            
            # Process each channel
            for idx, channel in enumerate(text_channels, 1):
                try:
                    self.logger.info(f"Processing channel {idx}/{len(text_channels)}: #{channel.name}")
                    
                    # Stage 1: Load messages from Discord → SQLite
                    channel_stats = await self.message_loader.load_channel_messages(
                        channel=channel,
                        limit=None  # Load all messages
                    )
                    
                    self.logger.info(
                        f"Loaded {channel_stats.get('successfully_loaded', 0)} messages "
                        f"from #{channel.name}"
                    )
                    
                    # Stage 2: Chunk and embed (only if messages were loaded)
                    if channel_stats.get('successfully_loaded', 0) > 0:
                        try:
                            chunked_service = ChunkedMemoryService(config=Config)
                            
                            chunk_stats = await chunked_service.ingest_channel(
                                channel_id=str(channel.id)
                            )
                            
                            self.logger.info(
                                f"Created {chunk_stats.get('total_chunks_created', 0)} chunks "
                                f"for #{channel.name}"
                            )
                        except Exception as e:
                            self.logger.error(f"Chunking failed for #{channel.name}: {e}", exc_info=True)
                            # Continue with next channel
                    
                except Exception as e:
                    self.logger.error(f"Failed to process channel #{channel.name}: {e}", exc_info=True)
                    # Continue with next channel
            
            self.logger.info("load_server task completed")
            
        except Exception as e:
            self.logger.error(f"Error in load_server task: {e}", exc_info=True)
            raise

    async def snapshot_task(self):
        """Post snapshot messages from 1-5 years ago."""
        self.logger.info("Starting snapshot task...")
        
        try:
            # Check if snapshot channel is configured
            snapshot_channel_id = Config.SNAPSHOT_CHANNEL_ID
            if not snapshot_channel_id or snapshot_channel_id <= 0:
                self.logger.warning("SNAPSHOT_CHANNEL_ID not configured, skipping snapshot task")
                return
            
            # Get the snapshot channel
            snapshot_channel = self.get_channel(snapshot_channel_id)
            if not snapshot_channel:
                self.logger.error(f"Snapshot channel {snapshot_channel_id} not found")
                return
            
            # Get today's date
            today = datetime.now()
            
            # Get messages from 1, 2, 3, 4, 5 years ago
            snapshot_messages = []
            
            # Get all channels to search (once, outside the loop)
            if not self.guilds:
                self.logger.warning("No guilds found, skipping snapshot")
                return
            
            guild = self.guilds[0]
            chatbot_channel_id = Config.CHATBOT_CHANNEL_ID
            text_channels = [
                ch for ch in guild.text_channels 
                if ch.permissions_for(guild.me).read_message_history
                and ch.id != chatbot_channel_id
                and ch.id != snapshot_channel_id  # Don't search in snapshot channel itself
            ]
            
            if not text_channels:
                self.logger.warning("No channels to search for snapshots")
                return
            
            for years_ago in range(1, 6):
                try:
                    target_date = today.replace(year=today.year - years_ago)
                    start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_date = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                    
                    self.logger.info(f"Searching for messages from {years_ago} year(s) ago ({target_date.date()})")
                    
                    # Search across all channels for messages on this date
                    found_messages = []
                    for channel in text_channels:
                        try:
                            channel_messages = self.message_storage.get_messages_by_date_range(
                                channel_id=str(channel.id),
                                start_date=start_date.isoformat(),
                                end_date=end_date.isoformat(),
                                limit=10  # Get more messages per channel to have options
                            )
                            
                            # Filter out bot messages and empty messages
                            for msg in channel_messages:
                                if not msg.get('is_bot', False) and msg.get('content', '').strip():
                                    found_messages.append({
                                        **msg,
                                        'channel_name': channel.name
                                    })
                            
                            # If we found enough messages, stop searching channels
                            if len(found_messages) >= 5:
                                break
                        except Exception as e:
                            self.logger.warning(f"Error searching channel #{channel.name}: {e}")
                            continue
                    
                    # Take up to 5 messages (or all if less than 5)
                    if found_messages:
                        snapshot_messages.append({
                            'years_ago': years_ago,
                            'date': target_date,
                            'messages': found_messages[:5]
                        })
                        self.logger.info(f"Found {len(found_messages[:5])} message(s) from {years_ago} year(s) ago")
                    else:
                        self.logger.info(f"No messages found from {years_ago} year(s) ago - skipping")
                        
                except ValueError as e:
                    # Handle leap year edge case (Feb 29)
                    if "day is out of range" in str(e):
                        self.logger.warning(f"Leap year edge case for {years_ago} years ago - skipping")
                        continue
                    raise
            
            # Post snapshot messages
            if snapshot_messages:
                await self.post_snapshot(snapshot_channel, snapshot_messages)
            else:
                self.logger.info("No snapshot messages found")
            
            self.logger.info("snapshot task completed")
            
        except Exception as e:
            self.logger.error(f"Error in snapshot task: {e}", exc_info=True)
            raise

    async def post_snapshot(self, channel: discord.TextChannel, snapshot_messages: List[Dict]):
        """Post snapshot messages to the channel."""
        try:
            embed = discord.Embed(
                title="📸 On This Day",
                description="Messages from years past",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            # Get guild_id from the channel (fallback if message doesn't have it)
            default_guild_id = channel.guild.id if channel.guild else None
            
            for snapshot in snapshot_messages:
                years_ago = snapshot['years_ago']
                date = snapshot['date']
                messages = snapshot['messages']
                
                if not messages:
                    continue
                
                # Format messages for this year with links
                message_texts = []
                for msg in messages:
                    author = msg.get('author_display_name') or msg.get('author_name', 'Unknown')
                    content = msg.get('content', '')[:200]  # Limit length
                    channel_name = msg.get('channel_name', 'unknown')
                    message_id = msg.get('message_id')
                    channel_id = msg.get('channel_id')
                    guild_id = msg.get('guild_id') or default_guild_id
                    
                    # Build message link
                    message_link = None
                    if guild_id and channel_id and message_id:
                        message_link = f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
                    
                    # Format message with optional link (with extra spacing)
                    msg_text = f"**{author}** in #{channel_name}: {content}"
                    if message_link:
                        msg_text += f"\n[🔗]({message_link})\n"
                    message_texts.append(msg_text)
                
                if message_texts:
                    field_value = "\n".join(message_texts)
                    if len(field_value) > 1024:  # Discord embed field limit
                        field_value = field_value[:1021] + "..."
                    
                    embed.add_field(
                        name=f"{years_ago} year{'s' if years_ago > 1 else ''} ago ({date.strftime('%B %d, %Y')})",
                        value=field_value,
                        inline=False
                    )
            
            if embed.fields:
                await channel.send(embed=embed)
                self.logger.info(f"Posted snapshot to #{channel.name}")
            else:
                self.logger.info("No snapshot messages to post")
                
        except Exception as e:
            self.logger.error(f"Error posting snapshot: {e}", exc_info=True)
            raise


async def main():
    """Main function to run the cronjob bot."""
    bot = CronjobBot()
    try:
        await bot.start(Config.DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Cronjob bot stopped")
    except Exception as e:
        logger.error(f"Error running cronjob bot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

