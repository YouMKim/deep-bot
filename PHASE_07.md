# Phase 7: Bot Commands Integration

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Bot Commands Integration

### Learning Objectives
- Integrate all components
- Design user-friendly commands
- Practice async progress reporting
- Learn component orchestration

### Implementation Steps

#### Step 7.1: Add Chunk Channel Command

Update `cogs/admin.py`:

```python
from services.message_storage import MessageStorage
from services.message_loader import MessageLoader
from services.chunking_service import ChunkingService
from services.chunked_memory_service import ChunkedMemoryService
from services.embedding_service import EmbeddingServiceFactory
from services.vector_store_factory import VectorStoreFactory
from utils.discord_utils import format_discord_message
import discord
from discord.ext import commands
from config import Config

@commands.command(name='chunk_channel')
@commands.is_owner()
async def chunk_channel(
    self, 
    ctx, 
    limit: int = None,
    rate_limit_delay: float = None
):
    """
    Fetch, chunk, and store messages for a channel.
    
    Usage: !chunk_channel [limit] [--rate-limit-delay SECONDS]
    
    Learning: Component orchestration - bringing all pieces together.
    """
    # Manual owner check
    if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
        await ctx.send("🚫 **Access Denied!** Only the bot owner can use this command.")
        return
    
    try:
        # Initialize services
        message_storage = MessageStorage()
        message_loader = MessageLoader()
        chunking_service = ChunkingService()
        
        # Initialize vector store and embedding provider
        vector_store = VectorStoreFactory.create()
        embedding_provider = EmbeddingServiceFactory.create()
        chunked_memory = ChunkedMemoryService(vector_store, embedding_provider)
        
        # Check checkpoint
        checkpoint = message_storage.get_checkpoint(str(ctx.channel.id))
        
        # Fetch messages
        status_msg = await ctx.send("🔄 Fetching messages...")
        
        # Progress callback
        async def on_progress(progress):
            try:
                await status_msg.edit(
                    content=f"🔄 Fetching... {progress['processed']} messages "
                           f"({progress['rate']:.1f} msg/s) | "
                           f"Stored: {progress.get('successful', 0)}"
                )
            except Exception as e:
                # Ignore edit errors (message might be deleted)
                pass
        
        message_loader.set_progress_callback(on_progress)
        
        # Determine fetch strategy
        after_message = None
        if checkpoint:
            try:
                after_message = await ctx.channel.fetch_message(checkpoint['last_message_id'])
                fetch_type = "incremental"
            except Exception:
                # Message doesn't exist, do full fetch
                after_message = None
                fetch_type = "full"
        else:
            fetch_type = "full"
        
        # Load messages from Discord
        stats = await message_loader.load_channel_messages(
            ctx.channel,
            limit=limit,
            after=after_message,
            rate_limit_delay=rate_limit_delay
        )
        
        # Format and save messages
        await status_msg.edit(content="💾 Saving messages to database...")
        messages = []
        async for message in ctx.channel.history(
            limit=limit, 
            after=after_message,
            oldest_first=False
        ):
            if not message.author.bot and message.content.strip():
                if not message.content.startswith(Config.BOT_PREFIX):
                    messages.append(format_discord_message(message))
        
        if messages:
            message_storage.save_channel_messages(str(ctx.channel.id), messages)
        
        # Load all messages for chunking (from database)
        await status_msg.edit(content="📦 Loading messages from database...")
        all_messages = message_storage.load_channel_messages(str(ctx.channel.id))
        
        if not all_messages:
            await status_msg.edit(content="❌ No messages found to chunk.")
            return
        
        # Chunk messages
        await status_msg.edit(content="📦 Chunking messages...")
        chunks_dict = chunking_service.chunk_messages(all_messages)
        
        # Store chunks
        await status_msg.edit(content="💾 Storing chunks in vector DB...")
        chunked_memory.store_all_strategies(chunks_dict)
        
        # Show results
        embed = discord.Embed(
            title="✅ Chunking Complete",
            description=f"Processed {len(all_messages)} messages",
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="Fetch Stats",
            value=(
                f"**Type:** {fetch_type}\n"
                f"**Processed:** {stats['total_processed']}\n"
                f"**Stored:** {stats['successfully_loaded']}\n"
                f"**Skipped:** {stats['skipped_bot_messages'] + stats['skipped_empty_messages']}"
            ),
            inline=False
        )
        
        stats_dict = chunked_memory.get_strategy_stats()
        chunk_info = "\n".join([f"**{s.title()}:** {c}" for s, c in stats_dict.items()])
        embed.add_field(
            name="Chunk Stats",
            value=chunk_info,
            inline=False
        )
        
        await status_msg.edit(content="", embed=embed)
        
    except Exception as e:
        self.logger.error(f"Error in chunk_channel: {e}", exc_info=True)
        await ctx.send(f"❌ Error: {e}")

@commands.command(name='chunk_stats')
@commands.is_owner()
async def chunk_stats(self, ctx, channel_id: str = None):
    """Show chunking statistics"""
    if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
        await ctx.send("🚫 **Access Denied!**")
        return
    
    try:
        vector_store = VectorStoreFactory.create()
        embedding_provider = EmbeddingServiceFactory.create()
        chunked_memory = ChunkedMemoryService(vector_store, embedding_provider)
        
        stats = chunked_memory.get_strategy_stats()
        
        embed = discord.Embed(
            title="📊 Chunking Statistics",
            color=discord.Color.blue()
        )
        
        for strategy, count in stats.items():
            embed.add_field(name=strategy.title(), value=str(count), inline=True)
        
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

@commands.command(name='chunk_checkpoint')
@commands.is_owner()
async def chunk_checkpoint(self, ctx, channel_id: str = None):
    """Show checkpoint information"""
    if str(ctx.author.id) != str(Config.BOT_OWNER_ID):
        await ctx.send("🚫 **Access Denied!**")
        return
    
    try:
        channel_id = channel_id or str(ctx.channel.id)
        message_storage = MessageStorage()
        checkpoint = message_storage.get_checkpoint(channel_id)
        
        if checkpoint:
            embed = discord.Embed(
                title="📍 Checkpoint Information",
                color=discord.Color.blue()
            )
            embed.add_field(name="Channel ID", value=channel_id, inline=False)
            embed.add_field(name="Last Message ID", value=checkpoint['last_message_id'], inline=False)
            embed.add_field(name="Last Fetch", value=checkpoint['last_fetch_timestamp'], inline=False)
            embed.add_field(name="Total Messages", value=str(checkpoint['total_messages']), inline=True)
            await ctx.send(embed=embed)
        else:
            await ctx.send("❌ No checkpoint found for this channel.")
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")
```

### Common Pitfalls - Phase 7

1. **Owner check**: Always verify owner before expensive operations
2. **Error handling**: Wrap everything in try/except
3. **Progress updates**: Handle edit errors gracefully
4. **Message limits**: Discord has message length limits
5. **Async/await**: All Discord operations must be awaited