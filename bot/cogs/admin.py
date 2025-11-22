import discord
from discord.ext import commands
import asyncio
import logging
from storage.messages import MessageStorage
from bot.loaders.message_loader import MessageLoader
from ai.service import AIService
from bot.utils.discord_utils import format_discord_message
from typing import List


class Admin(commands.Cog):
    """Admin-only commands for managing the bot and loading messages."""

    def __init__(self, bot):
        from config import Config
        
        self.bot = bot
        self.config = Config
        self.message_storage = MessageStorage()
        self.message_loader = MessageLoader(self.message_storage, config=self.config)
        self.logger = logging.getLogger(__name__)
        self.ai_service = None
        
        # Initialize social credit manager if enabled
        self.social_credit_manager = None
        if self.config.SOCIAL_CREDIT_ENABLED:
            from ai.social_credit import SocialCreditManager
            self.social_credit_manager = SocialCreditManager()

    async def cog_command_error(self, ctx, error):
        """Handle errors in admin commands"""
        if isinstance(error, commands.NotOwner):
            # Apply social credit penalty for unauthorized admin command attempt
            if self.config.SOCIAL_CREDIT_ENABLED and self.social_credit_manager:
                try:
                    user_id = str(ctx.author.id)
                    display_name = ctx.author.display_name
                    
                    # Apply penalty
                    new_score = await self.social_credit_manager.apply_penalty(
                        user_id,
                        "unauthorized_admin_command",
                        display_name
                    )
                    
                    # Get old tier and new tier for notification
                    old_tier = self.social_credit_manager.get_tone_tier(new_score + 200)
                    new_tier = self.social_credit_manager.get_tone_tier(new_score)
                    
                    message = f"🚨 **SOCIAL CREDIT VIOLATION** 🚨\n"
                    message += f"{ctx.author.mention} attempted to access admin command!\n"
                    message += f"**-200 SOCIAL CREDIT**\n"
                    message += f"New score: {new_score}"
                    
                    if old_tier != new_tier:
                        message += f"\n\nTier changed: {old_tier} → {new_tier}"
                    
                    await ctx.send(message)
                    return
                except Exception as e:
                    self.logger.error(f"Failed to apply admin command penalty: {e}", exc_info=True)
                    # Fall through to default message
            
            await ctx.send("🚫 **Access Denied!** You don't have permission to use admin commands. Only the bot admin can use these commands.")
        else:
            self.logger.error(f"Error in admin command: {error}")
            await ctx.send(f"❌ An error occurred: {error}")

    @commands.command(name='whoami', help='Check who the bot thinks is the owner')
    async def whoami(self, ctx):
        """Check who the bot thinks is the owner"""
        embed = discord.Embed(
            title="🤖 Bot Owner Information",
            color=discord.Color.blue()
        )
        
        # Show configured owner ID
        embed.add_field(
            name="Configured Owner ID",
            value=f"`{self.config.BOT_OWNER_ID}`",
            inline=False
        )
        
        # Show your user ID
        embed.add_field(
            name="Your User ID",
            value=f"`{ctx.author.id}`",
            inline=False
        )
        
        # Check if you're the owner
        is_owner = str(ctx.author.id) == str(self.config.BOT_OWNER_ID)
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
        
        if self.config.BLACKLIST_IDS:
            blacklist_str = "\n".join([f"• `{user_id}`" for user_id in self.config.BLACKLIST_IDS])
            embed.add_field(
                name=f"Blacklisted User IDs ({len(self.config.BLACKLIST_IDS)})",
                value=blacklist_str,
                inline=False
            )
            
            # Check if current author is blacklisted
            is_blacklisted = ctx.author.id in self.config.BLACKLIST_IDS
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
        
        if self.config.BLACKLIST_IDS:
            embed.add_field(
                name="Blacklist Types",
                value=f"[{', '.join([type(x).__name__ for x in self.config.BLACKLIST_IDS[:3]])}]",
                inline=False
            )
        
        await ctx.send(embed=embed)

    @commands.command(name='reload_blacklist', help='Reload the blacklist from environment variables (Admin only)')
    @commands.is_owner()
    async def reload_blacklist(self, ctx):
        """Reload the blacklist from environment variables"""
        import os
        
        try:
            # Show before state
            before_count = len(self.config.BLACKLIST_IDS)
            raw_env = os.getenv("BLACKLIST_IDS", "")
            
            # Reload
            self.config.load_blacklist()
            
            # Show after state
            after_count = len(self.config.BLACKLIST_IDS)
            
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
            
            if self.config.BLACKLIST_IDS:
                blacklist_str = "\n".join([f"• `{user_id}`" for user_id in self.config.BLACKLIST_IDS])
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
    @commands.is_owner()
    async def load_channel(self, ctx, limit: int = None):
        """Load all messages from the current channel into memory"""
        
        try:
            if limit and limit > 100000:
                warning = await ctx.send(
                    f"⚠️ **Warning:** Loading {limit:,} messages may take a very long time. "
                    f"Use `!load_channel` without limit to load all messages (recommended). "
                    f"This can be safely interrupted and resumed."
                )
            
            status_msg = await ctx.send(f"🔄 Loading messages from #{ctx.channel.name}...")
            
            async def progress_callback(progress):
                await status_msg.edit(
                    content=f"🔄 Loading... {progress['processed']} processed, "
                            f"{progress['successful']} saved ({progress['rate']:.1f} msg/s)"
                )
            
            self.message_loader.set_progress_callback(progress_callback)
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
            
            if stats.get('resumed_from_checkpoint'):
                embed.add_field(
                    name="🔄 Resume Status",
                    value="✅ Resumed from checkpoint",
                    inline=True
                )
            
            await status_msg.edit(content="", embed=embed)
            
            # Stage 2: Trigger chunking and vector storage in background
            if stats['successfully_loaded'] > 0:
                await ctx.send("🔄 Starting chunking and embedding process...")
                
                # Create background task for chunking
                async def chunk_in_background():
                    try:
                        from storage.chunked_memory import ChunkedMemoryService
                        chunked_service = ChunkedMemoryService(config=self.config)
                        
                        # Progress callback for chunking
                        chunking_status_msg = None
                        
                        async def chunking_progress_callback(progress):
                            nonlocal chunking_status_msg
                            try:
                                msg = (
                                    f"🔄 Chunking {progress['strategy']}: "
                                    f"{progress['total_processed']} messages processed, "
                                    f"{progress['chunks_created']} chunks created"
                                )
                                if chunking_status_msg:
                                    await chunking_status_msg.edit(content=msg)
                                else:
                                    chunking_status_msg = await ctx.send(msg)
                            except Exception:
                                pass  # Ignore progress update errors
                        
                        chunked_service.set_progress_callback(chunking_progress_callback)
                        
                        # Run the ingestion
                        chunk_stats = await chunked_service.ingest_channel(
                            channel_id=str(ctx.channel.id)
                        )
                        
                        # Send completion message
                        embed = discord.Embed(
                            title="✅ Chunking Complete",
                            description=f"Vector storage complete for #{ctx.channel.name}",
                            color=discord.Color.green()
                        )
                        
                        embed.add_field(
                            name="📊 Overall Statistics",
                            value=(
                                f"**Strategies Processed:** {chunk_stats['strategies_processed']}\n"
                                f"**Total Messages:** {chunk_stats['total_messages_processed']}\n"
                                f"**Total Chunks:** {chunk_stats['total_chunks_created']}\n"
                                f"**Errors:** {chunk_stats['total_errors']}\n"
                                f"**Duration:** {chunk_stats['duration_seconds']:.1f}s"
                            ),
                            inline=False
                        )
                        
                        # Add per-strategy details
                        strategy_summary = []
                        for strategy_name, details in chunk_stats['strategy_details'].items():
                            strategy_summary.append(
                                f"**{strategy_name}**: {details['chunks_created']} chunks "
                                f"({details['messages_processed']} msgs)"
                            )
                        
                        if strategy_summary:
                            embed.add_field(
                                name="📋 Per-Strategy Results",
                                value="\n".join(strategy_summary),
                                inline=False
                            )
                        
                        await ctx.send(embed=embed)
                        
                    except Exception as e:
                        self.logger.error(f"Chunking failed: {e}", exc_info=True)
                        await ctx.send(f"⚠️ Chunking failed: {e}")
                
                # Launch background task
                asyncio.create_task(chunk_in_background())
            
        except Exception as e:
            await ctx.send(f"❌ Error loading channel messages: {e}")

    @commands.command(name='check_storage', help='Check message storage statistics for current channel')
    async def check_storage(self, ctx):
        """Check message storage statistics for the current channel"""
        channel_id = str(ctx.channel.id)
        stats = self.message_storage.get_channel_stats(channel_id)
        
        embed = discord.Embed(
            title="💾 Message Storage Statistics",
            description=f"Storage info for #{ctx.channel.name}",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="📊 Message Count",
            value=f"{stats['message_count']} messages",
            inline=True
        )
        
        if stats['oldest_timestamp']:
            embed.add_field(
                name="📅 Oldest Message",
                value=stats['oldest_timestamp'],
                inline=True
            )
        
        if stats['newest_timestamp']:
            embed.add_field(
                name="📅 Newest Message",
                value=stats['newest_timestamp'],
                inline=True
            )
        
        if stats.get('checkpoint'):
            checkpoint = stats['checkpoint']
            embed.add_field(
                name="🔄 Checkpoint Info",
                value=(
                    f"**Last Message ID:** `{checkpoint['last_message_id']}`\n"
                    f"**Total Messages:** {checkpoint['total_messages']}\n"
                    f"**Last Fetch:** {checkpoint['last_fetch_timestamp']}"
                ),
                inline=False
            )
        else:
            embed.add_field(
                name="🔄 Checkpoint Info",
                value="No checkpoint found",
                inline=False
            )
        
        embed.add_field(
            name="🗄️ Database Location",
            value=f"`{self.message_storage.db_path}`",
            inline=False
        )
        
        await ctx.send(embed=embed)

    @commands.command(name='checkpoint_info', help='Show checkpoint information for current channel')
    async def checkpoint_info(self, ctx):
        """Show checkpoint details for the current channel"""
        channel_id = str(ctx.channel.id)
        checkpoint = self.message_storage.get_checkpoint(channel_id)
        
        embed = discord.Embed(
            title="🔄 Checkpoint Information",
            description=f"Checkpoint details for #{ctx.channel.name}",
            color=discord.Color.orange()
        )
        
        if checkpoint:
            embed.add_field(
                name="📝 Last Message ID",
                value=f"`{checkpoint['last_message_id']}`",
                inline=False
            )
            
            embed.add_field(
                name="📊 Total Messages",
                value=f"{checkpoint['total_messages']} messages",
                inline=True
            )
            
            embed.add_field(
                name="🕐 Last Fetch Timestamp",
                value=checkpoint['last_fetch_timestamp'],
                inline=True
            )
            
            if checkpoint.get('oldest_message_id'):
                embed.add_field(
                    name="📅 Oldest Message ID",
                    value=f"`{checkpoint['oldest_message_id']}`",
                    inline=False
                )
            
            if checkpoint.get('oldest_message_timestamp'):
                embed.add_field(
                    name="📅 Oldest Message Timestamp",
                    value=checkpoint['oldest_message_timestamp'],
                    inline=True
                )
            
            if checkpoint.get('newest_message_timestamp'):
                embed.add_field(
                    name="📅 Newest Message Timestamp",
                    value=checkpoint['newest_message_timestamp'],
                    inline=True
                )
        else:
            embed.add_field(
                name="⚠️ No Checkpoint",
                value="No checkpoint found for this channel. Messages have not been loaded yet.",
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='chunk_status', help='Show chunking progress and statistics for current channel')
    async def chunk_status(self, ctx):
        """Show chunking progress and statistics for the current channel"""
        channel_id = str(ctx.channel.id)
        
        embed = discord.Embed(
            title="📦 Chunking Status",
            description=f"Chunking and vector storage status for #{ctx.channel.name}",
            color=discord.Color.blue()
        )
        
        try:
            # Get message storage stats
            channel_stats = self.message_storage.get_channel_stats(channel_id)
            
            embed.add_field(
                name="💾 Message Storage",
                value=f"**Total Messages:** {channel_stats['message_count']}",
                inline=False
            )
            
            # Get chunking checkpoints for all strategies
            from chunking.constants import ChunkStrategy
            checkpoint_info = []
            
            for strategy in ChunkStrategy:
                checkpoint = self.message_storage.get_chunking_checkpoint(
                    channel_id, strategy.value
                )
                if checkpoint:
                    checkpoint_info.append(
                        f"**{strategy.value}**: Last processed `{checkpoint['last_message_id']}` "
                        f"at {checkpoint['last_message_timestamp'][:10]}"
                    )
                else:
                    checkpoint_info.append(f"**{strategy.value}**: Not started")
            
            if checkpoint_info:
                embed.add_field(
                    name="🔄 Chunking Checkpoints",
                    value="\n".join(checkpoint_info),
                    inline=False
                )
            else:
                embed.add_field(
                    name="🔄 Chunking Checkpoints",
                    value="No checkpoints found",
                    inline=False
                )
            
            # Get vector storage stats
            from storage.chunked_memory import ChunkedMemoryService
            chunked_service = ChunkedMemoryService(config=self.config)
            strategy_stats = chunked_service.get_strategy_stats()
            
            stats_info = []
            total_chunks = 0
            for strategy_name, count in strategy_stats.items():
                stats_info.append(f"**{strategy_name}**: {count:,} chunks")
                total_chunks += count
            
            if stats_info:
                embed.add_field(
                    name="📊 Vector Storage (Chunks per Strategy)",
                    value="\n".join(stats_info),
                    inline=False
                )
                
                embed.add_field(
                    name="📈 Total Chunks",
                    value=f"{total_chunks:,} chunks across all strategies",
                    inline=False
                )
            else:
                embed.add_field(
                    name="📊 Vector Storage",
                    value="No chunks found in vector database",
                    inline=False
                )
            
            # Calculate completion percentage
            if channel_stats['message_count'] > 0:
                completion_info = []
                for strategy in ChunkStrategy:
                    checkpoint = self.message_storage.get_chunking_checkpoint(
                        channel_id, strategy.value
                    )
                    if checkpoint:
                        # This is approximate - we can't easily determine exact percentage
                        completion_info.append(f"**{strategy.value}**: ✅ Processed")
                    else:
                        completion_info.append(f"**{strategy.value}**: ❌ Not started")
                
                embed.add_field(
                    name="✅ Completion Status",
                    value="\n".join(completion_info),
                    inline=False
                )
            
        except Exception as e:
            self.logger.error(f"Error getting chunk status: {e}", exc_info=True)
            embed.add_field(
                name="❌ Error",
                value=f"Failed to retrieve status: {e}",
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='reset_chunk_checkpoint', help='Delete chunk checkpoint to force re-processing (Admin only)')
    @commands.is_owner()
    async def reset_chunk_checkpoint(self, ctx, strategy: str = None):
        """
        Delete chunk checkpoint(s) to force re-processing from the beginning.
        
        Usage:
            !reset_chunk_checkpoint - Delete all checkpoints for current channel
            !reset_chunk_checkpoint single - Delete checkpoint for 'single' strategy only
        """
        try:
            channel_id = str(ctx.channel.id)
            
            # Delete checkpoint(s)
            deleted = self.message_storage.delete_chunking_checkpoint(channel_id, strategy)
            
            if deleted:
                if strategy:
                    await ctx.send(
                        f"✅ Deleted checkpoint for strategy `{strategy}`. "
                        f"Run `!rechunk {strategy}` to re-process from the beginning."
                    )
                else:
                    await ctx.send(
                        f"✅ Deleted all checkpoints for this channel. "
                        f"Run `!rechunk` to re-process from the beginning."
                    )
            else:
                await ctx.send(
                    f"ℹ️ No checkpoint found for {'strategy `' + strategy + '`' if strategy else 'this channel'}."
                )
                
        except Exception as e:
            self.logger.error(f"Failed to reset checkpoint: {e}", exc_info=True)
            await ctx.send(f"❌ Failed to reset checkpoint: {e}")

    @commands.command(name='rechunk', help='Re-run chunking from last checkpoint (Admin only)')
    @commands.is_owner()
    async def rechunk(self, ctx, strategy: str = None):
        """
        Re-run chunking for messages that haven't been chunked yet.
        
        Usage:
            !rechunk - Re-chunk all strategies from their last checkpoints
            !rechunk single - Re-chunk only the 'single' strategy
        """
        try:
            channel_id = str(ctx.channel.id)
            
            # Check if there are any messages to chunk
            channel_stats = self.message_storage.get_channel_stats(channel_id)
            
            if channel_stats['message_count'] == 0:
                await ctx.send("❌ No messages found in storage. Run `!load_channel` first.")
                return
            
            # Determine which strategies to process
            from chunking.constants import ChunkStrategy
            
            if strategy:
                # Validate strategy name
                try:
                    strategies = [ChunkStrategy(strategy.lower())]
                    strategy_name = strategy.lower()
                except ValueError:
                    valid_strategies = ", ".join([s.value for s in ChunkStrategy])
                    await ctx.send(
                        f"❌ Invalid strategy: `{strategy}`\n"
                        f"Valid strategies: {valid_strategies}"
                    )
                    return
            else:
                # Use None to let ingest_channel use config defaults
                strategies = None
                default_strats = self.config.CHUNKING_DEFAULT_STRATEGIES
                strategy_name = f"default strategies ({default_strats})"
            
            # Show initial status
            status_msg = await ctx.send(
                f"🔄 Starting chunking for {channel_stats['message_count']} messages "
                f"using {strategy_name}..."
            )
            
            # Create background task for chunking
            async def chunk_in_background():
                try:
                    from storage.chunked_memory import ChunkedMemoryService
                    chunked_service = ChunkedMemoryService(config=self.config)
                    
                    # Progress callback for chunking
                    chunking_status_msg = None
                    
                    async def chunking_progress_callback(progress):
                        nonlocal chunking_status_msg
                        try:
                            msg = (
                                f"🔄 Chunking {progress['strategy']}: "
                                f"{progress['total_processed']} messages processed, "
                                f"{progress['chunks_created']} chunks created"
                            )
                            if chunking_status_msg:
                                await chunking_status_msg.edit(content=msg)
                            else:
                                chunking_status_msg = await ctx.send(msg)
                        except Exception:
                            pass  # Ignore progress update errors
                    
                    chunked_service.set_progress_callback(chunking_progress_callback)
                    
                    # Run the ingestion
                    chunk_stats = await chunked_service.ingest_channel(
                        channel_id=channel_id,
                        strategies=strategies
                    )
                    
                    # Send completion message
                    embed = discord.Embed(
                        title="✅ Chunking Complete",
                        description=f"Vector storage complete for #{ctx.channel.name}",
                        color=discord.Color.green()
                    )
                    
                    embed.add_field(
                        name="📊 Overall Statistics",
                        value=(
                            f"**Strategies Processed:** {chunk_stats['strategies_processed']}\n"
                            f"**Total Messages:** {chunk_stats['total_messages_processed']}\n"
                            f"**Total Chunks:** {chunk_stats['total_chunks_created']}\n"
                            f"**Errors:** {chunk_stats['total_errors']}\n"
                            f"**Duration:** {chunk_stats['duration_seconds']:.1f}s"
                        ),
                        inline=False
                    )
                    
                    # Add per-strategy details
                    strategy_summary = []
                    for strategy_name, details in chunk_stats['strategy_details'].items():
                        strategy_summary.append(
                            f"**{strategy_name}**: {details['chunks_created']} chunks "
                            f"({details['messages_processed']} msgs)"
                        )
                    
                    if strategy_summary:
                        embed.add_field(
                            name="📋 Per-Strategy Results",
                            value="\n".join(strategy_summary),
                            inline=False
                        )
                    
                    await ctx.send(embed=embed)
                    
                except Exception as e:
                    self.logger.error(f"Chunking failed: {e}", exc_info=True)
                    await ctx.send(f"⚠️ Chunking failed: {e}")
            
            # Launch background task
            asyncio.create_task(chunk_in_background())
            
        except Exception as e:
            self.logger.error(f"Error in rechunk command: {e}", exc_info=True)
            await ctx.send(f"❌ Error: {e}")
    
    @commands.command(name='rag_settings', help='Show current RAG technique settings (Admin only)')
    @commands.is_owner()
    async def rag_settings(self, ctx):
        """Show current RAG technique settings."""
        
        embed = discord.Embed(
            title="⚙️ RAG Technique Settings",
            description="Current configuration for advanced RAG techniques",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="🔍 Hybrid Search",
            value="✅ Enabled" if self.config.RAG_USE_HYBRID_SEARCH else "❌ Disabled",
            inline=True
        )
        
        embed.add_field(
            name="🔄 Multi-Query",
            value="✅ Enabled" if self.config.RAG_USE_MULTI_QUERY else "❌ Disabled",
            inline=True
        )
        
        embed.add_field(
            name="📝 HyDE",
            value="✅ Enabled" if self.config.RAG_USE_HYDE else "❌ Disabled",
            inline=True
        )
        
        embed.add_field(
            name="🎯 Reranking",
            value="✅ Enabled" if self.config.RAG_USE_RERANKING else "❌ Disabled",
            inline=True
        )
        
        embed.add_field(
            name="📊 Max Output Tokens",
            value=f"{self.config.RAG_MAX_OUTPUT_TOKENS}",
            inline=True
        )
        
        embed.add_field(
            name="🔍 Default Chunking Strategy",
            value=f"`{self.config.RAG_DEFAULT_STRATEGY}`",
            inline=True
        )
        
        embed.set_footer(text="Use !rag_set <setting> <value> to change settings")
        
        await ctx.send(embed=embed)
    
    @commands.command(name='rag_set', help='Set a RAG technique setting (Admin only)')
    @commands.is_owner()
    async def rag_set(self, ctx, setting: str = None, value: str = None):
        """
        Set a RAG technique setting.
        
        Usage:
            !rag_set RAG_USE_HYBRID_SEARCH true
            !rag_set RAG_MAX_OUTPUT_TOKENS 2000
        """
        if not setting or value is None:
            await ctx.send(
                "❌ Usage: `!rag_set <setting> <value>`\n\n"
                "Valid settings:\n"
                "• `RAG_USE_HYBRID_SEARCH` (true/false)\n"
                "• `RAG_USE_MULTI_QUERY` (true/false)\n"
                "• `RAG_USE_HYDE` (true/false)\n"
                "• `RAG_USE_RERANKING` (true/false)\n"
                "• `RAG_MAX_OUTPUT_TOKENS` (integer)\n"
                "• `RAG_DEFAULT_STRATEGY` (single/tokens/author/temporal/conversation/sliding_window)"
            )
            return
        
        try:
            # Update setting
            self.config.update_rag_setting(setting, value)
            
            # Get updated value
            updated_value = getattr(self.config, setting)
            
            embed = discord.Embed(
                title="✅ Setting Updated",
                description=f"**{setting}** = `{updated_value}`",
                color=discord.Color.green()
            )
            embed.set_footer(text="Setting updated (in-memory, resets to .env defaults on restart)")
            
            await ctx.send(embed=embed)
        except ValueError as e:
            await ctx.send(f"❌ {str(e)}")
        except Exception as e:
            self.logger.error(f"Error updating RAG setting: {e}", exc_info=True)
            await ctx.send(f"❌ Error updating setting: {e}")
    
    @commands.command(name='rag_reset', help='Reset all RAG settings to defaults (Admin only)')
    @commands.is_owner()
    async def rag_reset(self, ctx):
        """Reset all RAG settings to defaults."""
        
        try:
            self.config.reset_rag_settings()
            
            embed = discord.Embed(
                title="✅ Settings Reset",
                description="All RAG settings have been reset to defaults",
                color=discord.Color.green()
            )
            
            embed.add_field(
                name="Default Values",
                value=(
                    "• Hybrid Search: ✅ Enabled\n"
                    "• Multi-Query: ✅ Enabled\n"
                    "• HyDE: ✅ Enabled\n"
                    "• Reranking: ✅ Enabled\n"
                    "• Max Output Tokens: 1000\n"
                    "• Default Strategy: author"
                ),
                inline=False
            )
            
            embed.set_footer(text="Settings reset to .env defaults (in-memory)")
            
            await ctx.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error resetting RAG settings: {e}", exc_info=True)
            await ctx.send(f"❌ Error resetting settings: {e}")
    
    @commands.command(name='rag_enable_all', help='Enable all RAG techniques at once (Admin only)')
    @commands.is_owner()
    async def rag_enable_all(self, ctx):
        """Enable all RAG techniques (Hybrid Search, Multi-Query, HyDE, Reranking)."""
        
        try:
            # Enable all techniques
            self.config.update_rag_setting('RAG_USE_HYBRID_SEARCH', True)
            self.config.update_rag_setting('RAG_USE_MULTI_QUERY', True)
            self.config.update_rag_setting('RAG_USE_HYDE', True)
            self.config.update_rag_setting('RAG_USE_RERANKING', True)
            
            embed = discord.Embed(
                title="✅ All RAG Techniques Enabled",
                description="All RAG techniques have been enabled",
                color=discord.Color.green()
            )
            
            embed.add_field(
                name="Enabled Techniques",
                value=(
                    "✅ Hybrid Search\n"
                    "✅ Multi-Query\n"
                    "✅ HyDE\n"
                    "✅ Reranking"
                ),
                inline=False
            )
            
            embed.set_footer(text="Settings updated (in-memory, resets to .env defaults on restart)")
            
            await ctx.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error enabling RAG techniques: {e}", exc_info=True)
            await ctx.send(f"❌ Error enabling techniques: {e}")
    
    @commands.command(name='compare_rag', help='Compare RAG output with different technique combinations (Admin only)')
    @commands.is_owner()
    async def compare_rag(self, ctx, *, question: str):
        """
        Compare RAG output with different technique combinations.
        
        Runs the query 6 times:
        1. All techniques OFF (baseline)
        2. Hybrid Search ON only
        3. Multi-Query ON only
        4. HyDE ON only
        5. Reranking ON only
        6. All techniques ON
        """
        
        if not question:
            await ctx.send("❌ Please provide a question to compare.")
            return
        
        # Validate question
        try:
            from rag.validation import QueryValidator
            question = QueryValidator.validate(question)
        except ValueError as e:
            await ctx.send(f"❌ {str(e)}")
            return
        
        status_msg = await ctx.send("🔄 Running comparison queries... This may take a while.")
        
        try:
            from rag.pipeline import RAGPipeline
            from rag.models import RAGConfig
            
            pipeline = RAGPipeline(config=self.config)
            
            # Define test configurations
            test_configs = [
                ("Baseline (All OFF)", {
                    'use_hybrid_search': False,
                    'use_multi_query': False,
                    'use_hyde': False,
                    'use_reranking': False,
                }),
                ("Hybrid Search Only", {
                    'use_hybrid_search': True,
                    'use_multi_query': False,
                    'use_hyde': False,
                    'use_reranking': False,
                }),
                ("Multi-Query Only", {
                    'use_hybrid_search': False,
                    'use_multi_query': True,
                    'use_hyde': False,
                    'use_reranking': False,
                }),
                ("HyDE Only", {
                    'use_hybrid_search': False,
                    'use_multi_query': False,
                    'use_hyde': True,
                    'use_reranking': False,
                }),
                ("Reranking Only", {
                    'use_hybrid_search': False,
                    'use_multi_query': False,
                    'use_hyde': False,
                    'use_reranking': True,
                }),
                ("All Techniques ON", {
                    'use_hybrid_search': True,
                    'use_multi_query': True,
                    'use_hyde': True,
                    'use_reranking': True,
                }),
            ]
            
            results = []
            total_cost = 0.0
            
            for i, (name, config_overrides) in enumerate(test_configs):
                await status_msg.edit(
                    content=f"🔄 Running comparison ({i+1}/6): {name}..."
                )
                
                config = RAGConfig(**config_overrides)
                result = await pipeline.answer_question(question, config)
                
                results.append({
                    'name': name,
                    'result': result,
                })
                total_cost += result.cost
            
            await status_msg.delete()
            
            # Create comparison embed
            embed = discord.Embed(
                title="🔬 RAG Technique Comparison",
                description=f"**Question:** {question}",
                color=discord.Color.purple()
            )
            
            # Add each result as a field (truncate answers to ~200 chars)
            for result_data in results:
                name = result_data['name']
                result = result_data['result']
                
                answer_preview = result.answer[:200] + "..." if len(result.answer) > 200 else result.answer
                
                embed.add_field(
                    name=f"{name}",
                    value=(
                        f"**Answer:** {answer_preview}\n"
                        f"**Cost:** ${result.cost:.4f} | "
                        f"**Sources:** {len(result.sources)} | "
                        f"**Tokens:** {result.tokens_used}"
                    ),
                    inline=False
                )
            
            embed.add_field(
                name="💰 Total Cost",
                value=f"${total_cost:.4f}",
                inline=True
            )
            
            embed.set_footer(text="Use !rag_set to enable/disable techniques globally")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            self.logger.error(f"Error in compare_rag: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error running comparison: {e}")
    
    @commands.command(name='ai_provider', help='Switch AI provider (Admin only)')
    @commands.is_owner()
    async def ai_provider(self, ctx, provider: str = None):
        """
        Get or set the AI provider. (Admin only)
        
        Usage:
            !ai_provider - Show current provider
            !ai_provider openai - Switch to OpenAI
            !ai_provider anthropic - Switch to Anthropic
            !ai_provider gemini - Switch to Gemini
        """
        
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
                value="• `openai` - GPT models (fast, versatile)\n• `anthropic` - Claude models (advanced reasoning)\n• `gemini` - Gemini models (Google AI)",
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
        if provider.lower() not in ["openai", "anthropic", "gemini"]:
            await ctx.send("❌ Invalid provider. Use `openai`, `anthropic`, or `gemini`")
            return
        
        # Switch provider
        try:
            summary_cog.ai_service = AIService(provider_name=provider.lower())
            
            # Update Basic cog
            basic_cog = self.bot.get_cog("Basic")
            if basic_cog:
                basic_cog.ai_service = AIService(provider_name=provider.lower())
            
            # Update Admin cog if it has AI service
            if self.ai_service:
                self.ai_service = AIService(provider_name=provider.lower())
            
            # Update RAG cog
            rag_cog = self.bot.get_cog("RAG")
            if rag_cog:
                rag_cog.pipeline.ai_service = AIService(provider_name=provider.lower())
            
            # Update Chatbot cog (CRITICAL - was missing!)
            chatbot_cog = self.bot.get_cog("Chatbot")
            if chatbot_cog:
                chatbot_cog.ai_service = AIService(provider_name=provider.lower())
                # Also update RAG pipeline in chatbot if it exists
                if hasattr(chatbot_cog, 'rag_pipeline'):
                    chatbot_cog.rag_pipeline.ai_service = AIService(provider_name=provider.lower())
            
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
            
            # Show which cogs were updated
            updated_cogs = ["Summary"]
            if basic_cog:
                updated_cogs.append("Basic")
            if self.ai_service:
                updated_cogs.append("Admin")
            if rag_cog:
                updated_cogs.append("RAG")
            if chatbot_cog:
                updated_cogs.append("Chatbot")
            
            embed.add_field(
                name="Updated Cogs",
                value=", ".join(updated_cogs),
                inline=True
            )
            
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Error switching provider: {e}")
    
    @commands.command(name='reindex_bot_knowledge', help='Re-index bot documentation into RAG system (Admin only)')
    @commands.is_owner()
    async def reindex_bot_knowledge(self, ctx, force: str = "false"):
        """
        Re-index bot documentation into the RAG system. (Admin only)
        
        Usage:
            !reindex_bot_knowledge - Re-index if not already indexed
            !reindex_bot_knowledge true - Force re-index even if already indexed
        """
        status_msg = await ctx.send("🔄 Re-indexing bot knowledge...")
        
        try:
            # Get ChunkedMemoryService from Summary cog
            summary_cog = self.bot.get_cog("Summary")
            if not summary_cog or not hasattr(summary_cog, 'chunked_memory_service'):
                await status_msg.edit(content="❌ Summary cog not loaded. Cannot access chunked memory service.")
                return
            
            chunked_memory = summary_cog.chunked_memory_service
            if not hasattr(chunked_memory, 'reindex_bot_knowledge'):
                await status_msg.edit(content="❌ Bot knowledge service not available.")
                return
            
            force_bool = force.lower() == "true"
            success = await chunked_memory.reindex_bot_knowledge(force=force_bool)
            
            if success:
                await status_msg.edit(content="✅ Bot knowledge successfully re-indexed!")
            else:
                await status_msg.edit(content="❌ Failed to re-index bot knowledge. Check logs for details.")
                
        except Exception as e:
            self.logger.error(f"Error re-indexing bot knowledge: {e}", exc_info=True)
            await status_msg.edit(content=f"❌ Error: {e}")

async def setup(bot):
    await bot.add_cog(Admin(bot))
