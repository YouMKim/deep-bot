import discord
from discord import HTTPException
import asyncio
import logging
from typing import Callable, Optional, Dict, Any, List, Tuple
from datetime import datetime
from config import Config
from storage.messages import MessageStorage
from bot.utils.discord_utils import format_discord_message 

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
        await asyncio.sleep(self.rate_limit_delay)
    
    async def _handle_rate_limit_error(self, error: HTTPException, retry_count: int):
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

    async def _handle_checkpoint_resume(
        self, 
        channel: discord.TextChannel, 
        checkpoint: Optional[Dict], 
        before: Optional, 
        after: Optional
    ) -> Tuple[Optional[discord.Message], bool]:
        """
        Handle checkpoint resume logic.
        
        Returns:
            Tuple of (after_message, did_resume) where:
            - after_message: Message to use as 'after' parameter, or None
            - did_resume: True if we successfully resumed from checkpoint
        """
        if not (checkpoint and before is None and after is None):
            return None, False
        
        if not checkpoint.get('last_message_id'):
            return None, False
        
        try:
            last_message_id = checkpoint['last_message_id']
            last_timestamp = checkpoint['last_fetch_timestamp']
            
            self.logger.info(
                f"Resuming from checkpoint: last_message_id={last_message_id}, "
                f"last_timestamp={last_timestamp}"
            )
            
            try:
                after_message = await channel.fetch_message(int(last_message_id))
                return after_message, True
            except (discord.NotFound, discord.HTTPException):
                self.logger.warning(
                    f"Could not fetch checkpoint message {last_message_id}, starting from scratch"
                )
                return None, False
        except Exception as e:
            self.logger.error(f"Error resuming from checkpoint: {e}")
            return None, False

    def _should_skip_message(self, message: discord.Message, stats: Dict[str, Any]) -> bool:
        if message.author.bot:
            stats["skipped_bot_messages"] += 1
            return True
        elif not message.content.strip():
            stats["skipped_empty_messages"] += 1
            return True
        elif message.content.startswith(Config.BOT_PREFIX):
            stats["skipped_commands"] += 1
            return True 
        return False 

    def _save_message_batch(
        self, 
        channel_id: str, 
        message_batch: List[Dict], 
        stats: Dict[str, Any]
    ) -> bool:
        if not message_batch:
            return True
        
        success = self.message_storage.save_channel_messages(channel_id, message_batch)
        if success:
            stats["successfully_loaded"] += len(message_batch)
            stats["batches_saved"] += 1
        else:
            stats["errors"] += len(message_batch)
        return success

    async def _report_progress(
        self, 
        stats: Dict[str, Any], 
        limit: Optional[int], 
        channel_name: str, 
        channel_id: str
    ):
        # Report progress every N messages OR on first message (for immediate feedback)
        # This ensures users see progress even for small loads
        should_report = (
            stats["total_processed"] == 1 or  # Always report first message
            stats["total_processed"] % Config.MESSAGE_FETCH_PROGRESS_INTERVAL == 0
        )
        
        if not should_report:
            return
        
        if not self.progress_callback:
            return
        
        elapsed = (datetime.now() - stats["start_time"]).total_seconds()
        rate = stats["total_processed"] / elapsed if elapsed > 0 else 0
        
        progress = {
            "processed": stats["total_processed"],
            "limit": limit or "unlimited",
            "rate": rate,
            "successful": stats["successfully_loaded"],
            "channel_name": channel_name,
            "channel_id": channel_id
        }
        
        if asyncio.iscoroutinefunction(self.progress_callback):
            await self.progress_callback(progress)
        else:
            self.progress_callback(progress)

    async def _handle_message_error(
        self,
        error: Exception,
        message: discord.Message,
        channel_name: str,
        channel_id: str,
        stats: Dict[str, Any],
        retry_count: int
    ) -> Tuple[bool, int]:
        """
        Handle errors during message processing.
        
        Returns:
            Tuple of (should_break, new_retry_count) where:
            - should_break: True if we should break out of the loop
            - new_retry_count: Updated retry count
        """
        if isinstance(error, HTTPException):
            if await self._handle_rate_limit_error(error, retry_count):
                stats["rate_limit_errors"] += 1
                retry_count += 1
                
                if retry_count >= Config.MESSAGE_FETCH_MAX_RETRIES:
                    self.logger.error(f"Max retries exceeded for #{channel_name} ({channel_id})")
                    return True, retry_count
                return False, retry_count
            else:
                stats["errors"] += 1
                self.logger.error(f"HTTP error: {error}")
                return False, retry_count
        else:
            stats["errors"] += 1
            self.logger.error(
                f"Error processing message {message.id} in #{channel_name} ({channel_id}): {error}"
            )
            return False, retry_count

    def _initialize_stats(self, did_resume: bool) -> Dict[str, Any]:
        """Initialize statistics dictionary."""
        return {
            "total_processed": 0,
            "successfully_loaded": 0,
            "skipped_bot_messages": 0,
            "skipped_empty_messages": 0,
            "skipped_commands": 0,
            "errors": 0,
            "rate_limit_errors": 0,
            "batches_saved": 0,
            "resumed_from_checkpoint": did_resume,
            "start_time": datetime.now(),
            "end_time": None,
        }

    def _log_completion(self, channel_name: str, stats: Dict[str, Any]):
        """Log completion statistics."""
        duration = (stats["end_time"] - stats["start_time"]).total_seconds()
        
        self.logger.info(
            f"Completed loading from #{channel_name}: "
            f"{stats['successfully_loaded']} stored in {stats['batches_saved']} batches, "
            f"{stats['skipped_bot_messages']} bot messages skipped, "
            f"{stats['skipped_empty_messages']} empty messages skipped, "
            f"{stats['skipped_commands']} commands skipped, "
            f"{stats['rate_limit_errors']} rate limit errors, "
            f"{stats['errors']} errors, "
            f"took {duration:.1f} seconds"
        ) 

    async def load_channel_messages(
        self,
        channel: discord.TextChannel,
        limit: Optional[int] = None,
        before: Optional = None,
        after: Optional = None,
        rate_limit_delay: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Load messages from a Discord channel and store them in SQLite.
        
        This is Stage 1 of the message processing pipeline:
        - Stage 1: Fetch from Discord → Store in SQLite (this method)
        - Stage 2: Read from SQLite → Chunk → Embed → Store in ChromaDB (later phase)
        """
        if rate_limit_delay:
            self.rate_limit_delay = rate_limit_delay

        channel_id = str(channel.id)
        self.logger.info(f"Loading messages from #{channel.name} ({channel_id})")

        # Handle checkpoint resume
        checkpoint = self.message_storage.get_checkpoint(channel_id)
        after_message, did_resume = await self._handle_checkpoint_resume(
            channel, checkpoint, before, after
        )
        if after_message:
            after = after_message

        # Initialize statistics
        stats = self._initialize_stats(did_resume)
        
        try:
            retry_count = 0
            message_batch: List[Dict] = []

            # IMPORTANT: Discord API defaults to newest-to-oldest (reverse chronological)
            # We MUST use oldest_first=True to fetch chronologically from oldest to newest
            # This is required for checkpoint/resume to work correctly
            async for message in channel.history(
                limit=limit,
                before=before,
                after=after,
                oldest_first=True  
            ):
                try:
                    stats["total_processed"] += 1
                    
                    # Report progress immediately (especially for first message)
                    # This gives users instant feedback
                    await self._report_progress(stats, limit, channel.name, channel_id)
                    
                    # Check if message should be skipped
                    if self._should_skip_message(message, stats):
                        await self._rate_limit_delay()
                        continue
                    
                    # Format and add message to batch
                    message_data = format_discord_message(message)
                    message_batch.append(message_data)

                    # Save batch when it reaches batch size
                    if len(message_batch) >= self.batch_size:
                        self._save_message_batch(channel_id, message_batch, stats)
                        message_batch = []  
                    
                    # Rate limit delay between messages 
                    await self._rate_limit_delay()
                    
                    # Reset retry count on success
                    retry_count = 0
                    
                except Exception as e:
                    should_break, retry_count = await self._handle_message_error(
                        e, message, channel.name, channel_id, stats, retry_count
                    )
                    if should_break:
                        break
                    continue
            
            # Save any remaining messages in the batch
            self._save_message_batch(channel_id, message_batch, stats)
            
            stats["end_time"] = datetime.now()
            self._log_completion(channel.name, stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Fatal error loading messages from #{channel.name} ({channel_id}): {e}")
            stats["end_time"] = datetime.now()
            stats["errors"] += 1
            return stats

