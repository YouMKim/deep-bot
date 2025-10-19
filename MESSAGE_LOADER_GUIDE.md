# 📥 Message Loader Service Implementation Guide

*Learn how to build a robust Discord message loading pipeline step by step*

## 🎯 **Learning Objectives**

By implementing this service, you will learn:
- **Discord API Rate Limiting**: How to respect API limits while loading messages
- **Batch Processing**: Efficiently handle large amounts of data
- **Error Handling**: Build resilient systems that continue working despite failures
- **Async Programming**: Master async/await patterns for I/O operations
- **Data Pipeline Design**: Create reusable, maintainable data processing services

---

## 🏗️ **Architecture Overview**

### **What We're Building**
A service that can:
1. Load all messages from a Discord channel
2. Load messages from a specific time period
3. Store them in your vector database
4. Provide detailed statistics about the loading process
5. Handle errors gracefully and respect rate limits

### **Key Components**
- **MessageLoader Class**: Main service for loading messages
- **Discord Integration**: Commands to trigger loading from Discord
- **Memory Integration**: Store loaded messages in your vector database
- **Statistics Tracking**: Monitor loading progress and success rates

---

## 📚 **Step-by-Step Implementation**

### **Step 1: Create the MessageLoader Service**

Create `services/message_loader.py`:

```python
"""
Message loading service for Discord bot.
Handles loading messages from Discord channels into memory.
"""

import discord
from discord.ext import commands
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime
from services.memory_service import MemoryService


class MessageLoader:
    """Service for loading Discord messages into memory storage."""
    
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service
        self.logger = logging.getLogger(__name__)
    
    # TODO: Implement the methods below
```

### **Step 2: Use the Utility Function**

Since you've already created `utils/discord_utils.py` with the formatting function, import and use it:

```python
from utils.discord_utils import format_discord_message

# In your loading methods, use:
message_data = format_discord_message(message)
```

**🤔 Learning Question**: Why do we need to format Discord messages? What information are we extracting and why?

### **Step 3: Implement Channel Message Loading**

This is the core method. Implement it step by step:

```python
async def load_channel_messages(
    self, 
    channel: discord.TextChannel, 
    limit: Optional[int] = None,
    before: Optional[discord.Message] = None,
    after: Optional[discord.Message] = None
) -> Dict[str, int]:
    """
    Load messages from a specific channel into memory.
    
    Args:
        channel: Discord channel to load messages from
        limit: Maximum number of messages to load (None = all)
        before: Load messages before this message
        after: Load messages after this message
        
    Returns:
        Dict with loading statistics
    """
    self.logger.info(f"Starting to load messages from #{channel.name}")
    
    # Initialize statistics tracking
    stats = {
        'total_processed': 0,
        'successfully_stored': 0,
        'skipped_bot': 0,
        'skipped_empty': 0,
        'errors': 0,
        'start_time': datetime.now(),
        'end_time': None
    }
    
    try:
        # TODO: Implement the loading logic here
        # Hint: Use channel.history() to iterate through messages
        # Hint: Process messages in batches to avoid memory issues
        # Hint: Add rate limiting with asyncio.sleep()
        # Hint: Track statistics as you process each message
        
        stats['end_time'] = datetime.now()
        duration = (stats['end_time'] - stats['start_time']).total_seconds()
        
        self.logger.info(
            f"Completed loading from #{channel.name}: "
            f"{stats['successfully_stored']} stored, "
            f"{stats['skipped_bot']} bot messages skipped, "
            f"{stats['skipped_empty']} empty messages skipped, "
            f"{stats['errors']} errors, "
            f"took {duration:.1f} seconds"
        )
        
        return stats
        
    except Exception as e:
        self.logger.error(f"Error loading messages from #{channel.name}: {e}")
        stats['end_time'] = datetime.now()
        return stats
```

**🔧 Implementation Hints**:
1. Use `async for message in channel.history(limit=limit, before=before, after=after, oldest_first=False)`
2. Skip bot messages: `if message.author.bot: continue`
3. Skip empty messages: `if not message.content.strip(): continue`
4. Skip command messages: `if message.content.startswith('!'): continue`
5. Add rate limiting: `await asyncio.sleep(0.1)` every 10 messages
6. Log progress every 50 messages

### **Step 4: Implement Recent Messages Loading**

```python
async def load_recent_messages(
    self, 
    channel: discord.TextChannel, 
    hours: int = 24
) -> Dict[str, int]:
    """
    Load messages from the last N hours.
    
    Args:
        channel: Discord channel to load from
        hours: Number of hours to look back
        
    Returns:
        Dict with loading statistics
    """
    from datetime import timedelta
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    cutoff_discord = discord.utils.utcnow() - timedelta(hours=hours)
    
    self.logger.info(f"Loading messages from last {hours} hours in #{channel.name}")
    
    return await self.load_channel_messages(
        channel=channel,
        after=cutoff_discord
    )
```

**🤔 Learning Question**: Why do we need to convert `datetime.utcnow()` to `discord.utils.utcnow()`? What's the difference?

### **Step 5: Implement Channel Message Counting**

```python
async def get_channel_message_count(self, channel: discord.TextChannel) -> int:
    """Get the total number of messages in a channel (approximate)."""
    try:
        count = 0
        async for _ in channel.history(limit=None):
            count += 1
            if count % 1000 == 0:
                await asyncio.sleep(0.1)  # Rate limiting
        return count
    except Exception as e:
        self.logger.error(f"Error counting messages in #{channel.name}: {e}")
        return 0
```

**🤔 Learning Question**: Why is this count "approximate"? What could cause it to be inaccurate?

---

## 🎮 **Step 6: Add Discord Commands**

Add these commands to your `cogs/summary.py`:

### **6.1 Import the MessageLoader**

```python
from services.message_loader import MessageLoader

class Summary(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.ai_service = AIService()
        self.memory_service = MemoryService()
        self.message_loader = MessageLoader(self.memory_service)  # Add this line
```

### **6.2 Add Load Channel Command**

```python
@commands.command(name='load_channel', help='Load all messages from current channel into memory')
async def load_channel(self, ctx, limit: int = None):
    """Load all messages from the current channel into memory"""
    try:
        # TODO: Add safety check for limit (max 10,000)
        # TODO: Show initial status message
        # TODO: Call message_loader.load_channel_messages()
        # TODO: Create and send results embed with statistics
        # TODO: Handle errors gracefully
        
    except Exception as e:
        await ctx.send(f"❌ Error loading channel messages: {e}")
```

### **6.3 Add Load Recent Command**

```python
@commands.command(name='load_recent', help='Load messages from last N hours')
async def load_recent(self, ctx, hours: int = 24):
    """Load messages from the last N hours"""
    try:
        # TODO: Add safety check for hours (max 168 = 1 week)
        # TODO: Show initial status message
        # TODO: Call message_loader.load_recent_messages()
        # TODO: Create and send results embed with statistics
        # TODO: Handle errors gracefully
        
    except Exception as e:
        await ctx.send(f"❌ Error loading recent messages: {e}")
```

### **6.4 Add Channel Info Command**

```python
@commands.command(name='channel_info', help='Get information about the current channel')
async def channel_info(self, ctx):
    """Get information about the current channel"""
    try:
        # TODO: Show "analyzing channel" message
        # TODO: Get message count using message_loader.get_channel_message_count()
        # TODO: Get memory stats using memory_service.get_channel_stats()
        # TODO: Create embed showing channel stats and memory coverage
        # TODO: Handle errors gracefully
        
    except Exception as e:
        await ctx.send(f"❌ Error getting channel info: {e}")
```

---

## 🛡️ **Safety Features to Implement**

### **Rate Limiting**
```python
# Add small delays to avoid hitting Discord API limits
if processed_in_batch % 10 == 0:
    await asyncio.sleep(0.1)
```

### **Safety Limits**
```python
# Prevent loading too many messages at once
if limit and limit > 10000:
    await ctx.send("❌ Limit cannot exceed 10,000 messages for safety.")
    return
```

### **Error Handling**
```python
# Continue processing even if individual messages fail
try:
    # Process message
    success = await self.memory_service.store_message(message_data)
    if success:
        stats['successfully_stored'] += 1
    else:
        stats['errors'] += 1
except Exception as e:
    self.logger.error(f"Error processing message {message.id}: {e}")
    stats['errors'] += 1
```

### **Progress Reporting**
```python
# Log progress for long operations
if processed_in_batch % 50 == 0:
    self.logger.info(f"Processed {processed_in_batch} messages from #{channel.name}")
```

---

## 🧪 **Testing Your Implementation**

### **Test 1: Basic Functionality**
1. Run your bot
2. Use `!load_recent 1` to load messages from the last hour
3. Check if messages appear in memory with `!memory_stats`

### **Test 2: Error Handling**
1. Try loading from a channel the bot can't access
2. Verify it handles the error gracefully

### **Test 3: Rate Limiting**
1. Load a large channel (if available)
2. Monitor the logs to see rate limiting in action

### **Test 4: Statistics**
1. Load some messages
2. Use `!channel_info` to verify statistics are accurate

---

## 🤔 **Learning Questions to Answer**

1. **Why do we need rate limiting?** What happens if we load messages too fast?

2. **Why skip bot messages?** What problems could storing bot messages cause?

3. **What's the difference between `load_channel` and `load_recent`?** When would you use each?

4. **How does the message counting work?** Why might it be "approximate"?

5. **Why use `oldest_first=False`?** What does this mean for the order of loaded messages?

6. **What information do we extract from each message?** Why is each field important?

---

## 🎯 **Success Criteria**

Your implementation is complete when you can:

- ✅ Load all messages from a channel with `!load_channel`
- ✅ Load recent messages with `!load_recent 24`
- ✅ View channel statistics with `!channel_info`
- ✅ See detailed loading statistics after each operation
- ✅ Handle errors gracefully without crashing
- ✅ Respect Discord API rate limits
- ✅ Store messages in your vector database successfully

---

## 🚀 **Next Steps**

Once you've implemented the message loader:

1. **Test it thoroughly** with different channels and time periods
2. **Monitor the logs** to understand how it works
3. **Experiment with different limits** to see how it affects performance
4. **Move on to RAG implementation** - you'll have a solid foundation!

---

## 💡 **Pro Tips**

- **Start small**: Test with recent messages first, then try larger loads
- **Monitor logs**: Use logging to understand what's happening
- **Test edge cases**: Empty channels, channels with only bot messages, etc.
- **Be patient**: Large channels can take time to load
- **Check memory usage**: Monitor your system resources during large loads

---

**Happy coding! This is a great learning exercise that will teach you a lot about async programming, error handling, and building robust data pipelines.** 🚀
