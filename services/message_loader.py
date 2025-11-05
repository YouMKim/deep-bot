import discord
from discord.ext import commands
import asyncio
import logging
from discord.ext.commands import HTTPException
from config import Config, Callable, Optional, Dict
from datetime import datetime
from services.message_storage import MessageStorage
from utils.discord_utils import format_discord_message 

class MessageLoader: 
    def __init__(self, message_storage: MessageStorage):
        self.message_storage = message_storage
        self.logger = logging.getLogger(__name__) 
        self.rate_limit_delay = Config.MESSAGE_FETCH_DELAY
        self.batch_size = Config.MESSAGE_FETCH_BATCH_SIZE
        self.progress_callback: Optional[Callable[[int, int], None]] = None
        
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        self.progress_callback = callback 

    async def _rate_limit_delay(self):
        await asyncio.sleep(self.rate_limit_dealy)
    
    async def _handle_rate_limit_error(self, error: HTTPException, retry_count:int):
        if error.status == 429:
            retry_after = getattr(error, 'retry_after', None) or (2 ** retry_count)
            wait_time = min(retry_after, 60)  
            
            self.logger.warning(
                f"Rate limited! Waiting {wait_time}s before retry "
                f"(attempt {retry_count + 1}/{Config.MESSAGE_FETCH_MAX_RETRIES})"
            )
            await asyncio.sleep(wait_time)
            return True  
        return False 

    async def load_channel_messages(
        self,
        channel: discord.TextChannel,
        limit: Optional[int] = None,
        before: Optional = None,
        after: Optional = None,
        rate_limit_delay: Optional[float] = None
    ) -> Dict[str, Any]:
        if rate_limit_delay:
            self._rate_limit_delay()

        channel_id = str(channel.id)
        self.logger.info(f"Loading messages from #{channel.name} ({channel_id})")

        checkpoint = self.message_storage.get_checkpoint(channel_id)
        resume_from_oldest = False 

        if checkpoint and before is None and after is None:
            if checkpoint.get('oldest_message_timestamp'):
                try:
                    oldest_timestamp = checkpoint['oldest_message_timestamp']
                    oldest_message_id = checkpoint['oldest_message_id']


        

        except Exception as e:
            self.logger.error(f"Error loading messages from #{channel.name} ({channel.id}): {e}")
            stats["end_time"] = datetime.now() 
            stats["errors"] += 1
            return stats