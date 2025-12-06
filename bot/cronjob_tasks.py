"""
Cronjob tasks that run as background tasks in the bot.
"""

import logging
import random
from datetime import datetime
from typing import List, Dict, Optional

import discord
from discord.ext import commands

from config import Config
from storage.messages import MessageStorage
from bot.loaders.message_loader import MessageLoader
from storage.chunked_memory import ChunkedMemoryService
from bot.utils.year_stats import calculate_user_stats

logger = logging.getLogger(__name__)


class CronjobTasks:
    """Handles scheduled cronjob tasks."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.message_storage = MessageStorage()
        self.message_loader = MessageLoader(self.message_storage, config=Config)
        self.logger = logger
        # Reuse ChunkedMemoryService to avoid creating multiple instances
        self._chunked_service = None

    async def load_server_task(self):
        """Load all messages from all channels in the server."""
        self.logger.info("Starting load_server task...")
        
        try:
            # Get the first guild (assuming single guild deployment)
            if not self.bot.guilds:
                self.logger.warning("No guilds found, skipping load_server task")
                return
            
            guild = self.bot.guilds[0]
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
                            # Reuse service instance to avoid memory bloat
                            if self._chunked_service is None:
                                self._chunked_service = ChunkedMemoryService(config=Config)
                            
                            chunk_stats = await self._chunked_service.ingest_channel(
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
            snapshot_channel = self.bot.get_channel(snapshot_channel_id)
            if not snapshot_channel:
                self.logger.error(f"Snapshot channel {snapshot_channel_id} not found")
                return
            
            # Get today's date
            today = datetime.now()
            
            # Get messages from 1, 2, 3, 4, 5 years ago
            snapshot_messages = []
            
            # Get all channels to search (once, outside the loop)
            if not self.bot.guilds:
                self.logger.warning("No guilds found, skipping snapshot")
                return
            
            guild = self.bot.guilds[0]
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
                                limit=10  # Get messages for random selection
                            )
                            
                            # Filter out bot messages and empty messages
                            for msg in channel_messages:
                                if not msg.get('is_bot', False) and msg.get('content', '').strip():
                                    found_messages.append({
                                        **msg,
                                        'channel_name': channel.name
                                    })
                        except Exception as e:
                            self.logger.warning(f"Error searching channel #{channel.name}: {e}")
                            continue
                    
                    # Pick 1 random message (or all if less than 1)
                    if found_messages:
                        # Randomly select 1 message
                        selected_message = random.choice(found_messages)
                        snapshot_messages.append({
                            'years_ago': years_ago,
                            'date': target_date,
                            'messages': [selected_message]
                        })
                        self.logger.info(f"Found and selected 1 random message from {years_ago} year(s) ago")
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
                
                # Format the single message for this year
                msg = messages[0]  # Only one message now
                author = msg.get('author_display_name') or msg.get('author_name', 'Unknown')
                content = msg.get('content', '')
                channel_name = msg.get('channel_name', 'unknown')
                message_id = msg.get('message_id')
                channel_id = msg.get('channel_id')
                guild_id = msg.get('guild_id') or default_guild_id  # Use message's guild_id or fallback
                
                # Build message link if we have all required IDs
                message_link = None
                if guild_id and channel_id and message_id:
                    message_link = f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
                
                # Format the message with link (with extra newlines for spacing)
                if message_link:
                    # Add link with multiple newlines before and after for better spacing
                    field_value = f"**{author}** in #{channel_name}:\n{content}\n\n[🔗 View Original Message]({message_link})\n\n"
                else:
                    field_value = f"**{author}** in #{channel_name}:\n{content}"
                
                # Truncate if too long (Discord embed field limit is 1024 chars)
                if len(field_value) > 1024:
                    # Truncate content but keep the link with proper spacing
                    if message_link:
                        header_len = len(f"\n\n**{author}** in #{channel_name}:\n")
                        link_text = f"\n[🔗 View Original Message]({message_link})\n"
                        link_len = len(link_text)
                        available = 1024 - header_len - link_len - 10  # -10 for "..."
                        content_truncated = content[:available] if available > 0 else ""
                        field_value = f"**{author}** in #{channel_name}:\n{content_truncated}...{link_text}"
                    else:
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

    async def year_in_review_task(self):
        """Process one user's year-in-review statistics."""
        self.logger.info("Starting year-in-review task...")
        
        try:
            # Check if year-in-review is enabled
            if not Config.YEAR_IN_REVIEW_ENABLED:
                self.logger.info("Year-in-review is disabled, skipping")
                return
            
            # Date range: January 1, 2025 to December 4, 2025
            start_date = datetime(2025, 1, 1, 0, 0, 0)
            end_date = datetime(2025, 12, 4, 23, 59, 59)
            
            start_date_str = start_date.isoformat()
            end_date_str = end_date.isoformat()
            
            # Get next unprocessed user
            user = self.message_storage.get_next_unprocessed_user(start_date_str, end_date_str)
            
            if not user:
                self.logger.info("All users have been processed for year-in-review")
                return
            
            user_id = user['author_id']
            user_display_name = user['author_display_name']
            
            self.logger.info(f"Processing year-in-review for user: {user_display_name} ({user_id})")
            
            # Process the user
            await self.process_next_user_year_review(user_id, user_display_name, start_date_str, end_date_str)
            
            self.logger.info("year-in-review task completed")
            
        except Exception as e:
            self.logger.error(f"Error in year-in-review task: {e}", exc_info=True)
            raise

    async def process_next_user_year_review(
        self,
        user_id: str,
        user_display_name: str,
        start_date_str: str,
        end_date_str: str
    ):
        """Process a single user's year-in-review statistics."""
        try:
            # Get user's messages from date range
            messages = self.message_storage.get_user_messages_by_date_range(
                author_id=user_id,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            if not messages:
                self.logger.info(f"No messages found for user {user_display_name} in date range")
                # Mark as completed even if no messages
                self.message_storage.mark_year_in_review_completed(user_id, user_display_name)
                return
            
            # Get guild_id from first message (for generating links)
            guild_id = messages[0].get('guild_id') if messages else None
            
            # Calculate statistics
            stats = calculate_user_stats(messages, guild_id=guild_id)
            
            # Get channel to post to
            channel_id = Config.YEAR_IN_REVIEW_CHANNEL_ID or Config.SNAPSHOT_CHANNEL_ID
            if not channel_id or channel_id <= 0:
                self.logger.warning("YEAR_IN_REVIEW_CHANNEL_ID not configured, skipping post")
                # Still mark as completed
                self.message_storage.mark_year_in_review_completed(user_id, user_display_name)
                return
            
            channel = self.bot.get_channel(channel_id)
            if not channel:
                self.logger.error(f"Year-in-review channel {channel_id} not found")
                # Still mark as completed
                self.message_storage.mark_year_in_review_completed(user_id, user_display_name)
                return
            
            # Post the year-in-review embed
            await self.post_year_in_review(channel, user_display_name, stats, guild_id)
            
            # Mark as completed
            self.message_storage.mark_year_in_review_completed(user_id, user_display_name)
            
        except Exception as e:
            self.logger.error(f"Error processing year-in-review for user {user_display_name}: {e}", exc_info=True)
            raise

    async def post_year_in_review(
        self,
        channel: discord.TextChannel,
        user_display_name: str,
        stats: Dict,
        guild_id: Optional[str] = None
    ):
        """Post year-in-review statistics as a Discord embed."""
        try:
            embed = discord.Embed(
                title=f"📊 {user_display_name}'s 2025 Year in Review",
                description="Statistics from January 1 - December 4, 2025",
                color=discord.Color.gold(),
                timestamp=datetime.now()
            )
            
            # Overview section
            basic = stats.get('basic_counts', {})
            overview_text = (
                f"**Messages:** {basic.get('total_messages', 0):,}\n"
                f"**Words:** {basic.get('total_words', 0):,}\n"
                f"**Characters:** {basic.get('total_chars', 0):,}\n"
                f"**Avg per message:** {basic.get('avg_words', 0):.1f} words"
            )
            embed.add_field(name="📈 Overview", value=overview_text, inline=False)
            
            # Extremes section
            extremes = stats.get('extremes', {})
            extremes_text = ""
            if extremes.get('longest'):
                longest = extremes['longest']
                preview = longest['content'][:100] + "..." if len(longest['content']) > 100 else longest['content']
                link_text = f" [🔗]({longest['link']})" if longest.get('link') else ""
                extremes_text += f"**Longest:** {longest['char_count']:,} chars - {preview}{link_text}\n"
            
            if extremes.get('shortest'):
                shortest = extremes['shortest']
                preview = shortest['content'][:50] + "..." if len(shortest['content']) > 50 else shortest['content']
                link_text = f" [🔗]({shortest['link']})" if shortest.get('link') else ""
                extremes_text += f"**Shortest:** {shortest['char_count']:,} chars - \"{preview}\"{link_text}\n"
            
            if extremes_text:
                embed.add_field(name="📝 Extremes", value=extremes_text, inline=False)
            
            # Activity patterns
            activity = stats.get('activity', {})
            activity_text = ""
            if activity.get('most_active_hour') is not None:
                hour = activity['most_active_hour']
                # Format hour as 12-hour with AM/PM
                if hour == 0:
                    hour_display = "12 AM"
                elif hour < 12:
                    hour_display = f"{hour} AM"
                elif hour == 12:
                    hour_display = "12 PM"
                else:
                    hour_display = f"{hour - 12} PM"
                activity_text += f"**Most active hour:** {hour_display} ({activity.get('most_active_hour_count', 0)} messages)\n"
            
            if activity.get('most_active_day'):
                activity_text += f"**Most active day:** {activity['most_active_day']} ({activity.get('most_active_day_count', 0)} messages)\n"
            
            if activity.get('most_active_month'):
                activity_text += f"**Most active month:** {activity['most_active_month']} ({activity.get('most_active_month_count', 0)} messages)\n"
            
            if activity.get('peak_day'):
                peak = activity['peak_day']
                peak_date = peak.get('date')
                if peak_date:
                    if isinstance(peak_date, str):
                        try:
                            peak_date = datetime.fromisoformat(peak_date.replace('Z', '+00:00')).date()
                        except (ValueError, AttributeError):
                            pass
                    if hasattr(peak_date, 'strftime'):
                        date_str = peak_date.strftime('%B %d, %Y')
                    else:
                        date_str = str(peak_date)
                    activity_text += f"**Peak day:** {date_str} ({peak['count']} messages)\n"
            
            if activity.get('active_days'):
                activity_text += f"**Active days:** {activity['active_days']}/338\n"
            
            if activity.get('longest_streak'):
                activity_text += f"**Longest streak:** {activity['longest_streak']} consecutive days\n"
            
            if activity.get('time_preference'):
                time_pref = activity['time_preference'].title()
                activity_text += f"**Time preference:** {time_pref}\n"
            
            if activity_text:
                embed.add_field(name="⏰ Activity", value=activity_text, inline=False)
            
            # Channel preferences
            channels = stats.get('channels', {})
            channel_text = ""
            if channels.get('most_active_channel'):
                most_active = channels['most_active_channel']
                channel_text += f"**Most active:** #{most_active['name']} ({most_active['count']} messages)\n"
            
            if channels.get('top_channels'):
                top_channels_list = []
                for ch in channels['top_channels'][:3]:
                    top_channels_list.append(f"#{ch['name']} ({ch['count']})")
                if top_channels_list:
                    channel_text += f"**Top 3:** {', '.join(top_channels_list)}\n"
            
            if channels.get('total_channels'):
                channel_text += f"**Channels active in:** {channels['total_channels']} channels"
            
            if channel_text:
                embed.add_field(name="📍 Channels", value=channel_text, inline=False)
            
            # Emoji usage
            emojis = stats.get('emojis', {})
            emoji_text = ""
            if emojis.get('total_emojis'):
                emoji_text += f"**Total emojis:** {emojis['total_emojis']}\n"
            
            if emojis.get('top_emojis'):
                top_emoji_list = []
                for emoji_data in emojis['top_emojis']:
                    top_emoji_list.append(f"{emoji_data['emoji']} ({emoji_data['count']})")
                if top_emoji_list:
                    emoji_text += f"**Top 3:** {', '.join(top_emoji_list)}\n"
            
            if emojis.get('emoji_usage_pct'):
                emoji_text += f"**Emoji usage:** {emojis['emoji_usage_pct']}% of messages\n"
            
            if emojis.get('most_emoji_message'):
                most_emoji = emojis['most_emoji_message']
                preview = most_emoji['content'][:50] + "..." if len(most_emoji['content']) > 50 else most_emoji['content']
                link_text = f" [🔗]({most_emoji['link']})" if most_emoji.get('link') else ""
                emoji_text += f"**Most emojis in one message:** {most_emoji['emoji_count']} - \"{preview}\"{link_text}"
            
            if emoji_text:
                embed.add_field(name="😀 Emojis", value=emoji_text, inline=False)
            
            # Word analysis
            words = stats.get('words', [])
            if words:
                word_list = [f"\"{w['word']}\" ({w['count']})" for w in words[:5]]
                word_text = f"**Top 5:** {', '.join(word_list)}"
                embed.add_field(name="💬 Words", value=word_text, inline=False)
            
            await channel.send(embed=embed)
            self.logger.info(f"Posted year-in-review for {user_display_name}")
            
        except Exception as e:
            self.logger.error(f"Error posting year-in-review: {e}", exc_info=True)
            raise

    async def run_all_tasks(self):
        """Run all cronjob tasks."""
        self.logger.info("Starting cronjob tasks...")
        
        # Task 1: Load server (ingest all unprocessed messages)
        await self.load_server_task()
        
        # Task 2: Snapshot (post messages from years ago)
        await self.snapshot_task()
        
        # Task 3: Year-in-review (process one user's stats)
        await self.year_in_review_task()
        
        self.logger.info("Cronjob tasks completed successfully")

