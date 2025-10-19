import discord
from discord.ext import commands
import asyncio
import logging
from config import Config
from typing import List, Dict, Optional
from datetime import datetime
from services.memory_service import MemoryService
from utils.discord_utils import format_discord_message 

class MessageLoader: 
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service 
        self.logger = logging.getLogger(__name__) 
    

    async def load_channel_messages(self, channel: discord.TextChannel, limit: Optional[int] = None, before: Optional[discord.Message] = None, after: Optional[discord.Message] = None) -> Dict[str, int]:
        """Load messages from a specific channel

        Args:
            channel: The channel to load messages from
            limit: The maximum number of messages to load
            before: Load messages before this message
            after: Load messages after this message
        
        Returns:
            Dict with stats on loading  
        """

        self.logger.info(f"Loading messages from #{channel.name} ({channel.id})")

        stats = {
            "total_processed": 0,
            "successfully_loaded": 0, 
            "skipped_bot_messages": 0,
            "skipped_empty_messages": 0,
            "skipped_blacklisted": 0,
            "skipped_commands": 0,
            "errors": 0,
            "start_time": datetime.now(),
            "end_time": None, 
        }

        try:

            batch_size = 50  
            processed_in_batch = 0
            
            async for message in channel.history(limit=limit, before=before, after=after, oldest_first=False):
                stats["total_processed"] += 1
                
                # Skip messages that shouldn't be stored
                if message.author.bot:
                    stats["skipped_bot_messages"] += 1
                    continue
                elif message.author.id in Config.BLACKLIST_IDS:
                    stats["skipped_blacklisted"] += 1
                    continue
                elif not message.content.strip():
                    stats["skipped_empty_messages"] += 1
                    continue
                elif message.content.startswith(Config.BOT_PREFIX):
                    stats["skipped_commands"] += 1
                    continue
                else:
                    try:
                        message_data = format_discord_message(message)
                        success = await self.memory_service.store_message(message_data)
                        
                        if success:
                            stats["successfully_loaded"] += 1
                        else:
                            stats["errors"] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error processing message {message.id}: {e}")
                        stats["errors"] += 1
                
                processed_in_batch += 1
                
                if processed_in_batch >= batch_size:
                    self.logger.info(f"Processed {stats['total_processed']} messages from #{channel.name}")
                    await asyncio.sleep(0.5)  
                    processed_in_batch = 0 
                
                if stats["total_processed"] % 10 == 0:
                    await asyncio.sleep(0.1) 

            stats['end_time'] = datetime.now()
            duration = (stats['end_time'] - stats['start_time']).total_seconds()
            
            self.logger.info(
                f"Completed loading from #{channel.name}: "
                f"{stats['successfully_loaded']} stored, "
                f"{stats['skipped_bot_messages']} bot messages skipped, "
                f"{stats['skipped_blacklisted']} blacklisted users skipped, "
                f"{stats['skipped_empty_messages']} empty messages skipped, "
                f"{stats['skipped_commands']} commands skipped, "
                f"{stats['errors']} errors, "
                f"took {duration:.1f} seconds"
            )
            
            return stats


        except Exception as e:
            self.logger.error(f"Error loading messages from #{channel.name} ({channel.id}): {e}")
            stats["end_time"] = datetime.now() 
            stats["errors"] += 1
            return stats