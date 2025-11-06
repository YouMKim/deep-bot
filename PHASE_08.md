# Phase 8: Summary Enhancement

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Summary Enhancement

### Learning Objectives
- Understand caching patterns
- Learn performance optimization
- Practice graceful degradation

### Implementation Steps

#### Step 8.1: Enhance Summary Command

Update `bot/cogs/summary.py`:

```python
from storage.messages import MessageStorage
from config import Config

async def _fetch_messages_with_fallback(self, ctx, count: int) -> List[dict]:
    """
    Fetch messages using DB-first approach with Discord API fallback.
    
    Learning: Cache-first pattern improves performance.
    """
    message_storage = MessageStorage()
    checkpoint = message_storage.get_checkpoint(str(ctx.channel.id))
    
    if checkpoint and Config.SUMMARY_USE_STORED_MESSAGES:
        # Get from storage
        stored_messages = message_storage.get_recent_messages(
            str(ctx.channel.id),
            limit=count
        )
        
        # Check for gaps (new messages on Discord)
        try:
            # Try to fetch the latest message from Discord
            latest_discord_msg = await ctx.channel.fetch_message(
                checkpoint['last_message_id']
            )
            # If successful and we have enough stored messages, use them
            if stored_messages and len(stored_messages) >= count:
                return stored_messages[:count]
        except Exception:
            # Message doesn't exist or we're missing new ones
            pass
        
        # Fetch gap if needed
        if stored_messages:
            try:
                after_msg = await ctx.channel.fetch_message(
                    checkpoint['last_message_id']
                )
                new_messages = []
                async for msg in ctx.channel.history(
                    limit=count,
                    after=after_msg
                ):
                    if not msg.author.bot and msg.content.strip():
                        if not msg.content.startswith(ctx.prefix):
                            new_messages.append(format_discord_message(msg))
                
                # Merge and return
                all_messages = stored_messages + new_messages
                return all_messages[:count]
            except Exception as e:
                self.logger.warning(f"Error fetching gap: {e}, using stored only")
                return stored_messages[:count]
    
    # Fallback to Discord API
    return await self._fetch_messages(ctx, count)

# Update the summary command to use the new method
@commands.command(name="summary", help="Generate a summary of previous messages")
async def summary(self, ctx, count: int = 50):
    status_msg = await ctx.send("🔍 Fetching messages...")
    
    # Use DB-first approach
    messages = await self._fetch_messages_with_fallback(ctx, count)
    
    if not messages:
        await status_msg.edit(content="❌ No messages found to summarize.")
        return
    
    # Rest of summary logic...
    await status_msg.edit(content=f"📊 Analyzing {len(messages)} messages...")
    # ... existing summary code ...
```