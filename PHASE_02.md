# Phase 2: Rate Limiting & Message Loading to SQLite

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

Rate Limiting & Message Loading to SQLite

### Learning Objectives
- Understand Discord API rate limits
- Implement exponential backoff
- Design progress reporting
- Learn async error handling
- Understand two-stage message processing pipeline
- Learn batch collection and storage patterns

### Design Principles
- **Two-Stage Pipeline**: Fetch → Store in SQLite (Stage 1), then Chunk → Embed → ChromaDB (Stage 2)
- **Configurable Behavior**: Rate limits as configuration
- **Error Recovery**: Graceful handling of rate limit errors
- **Progress Reporting**: Observer pattern for progress updates
- **Checkpoint/Resume**: Resume from last message if interrupted
- **Batch Efficiency**: Collect messages in batches before storing

### Two-Stage Message Processing

**Stage 1 (This Phase)**: Fetch messages from Discord → Store in SQLite (`MessageStorage`)
- Raw message data stored for later processing
- Checkpoint system allows resuming interrupted loads
- Batch collection for efficiency

**Stage 2 (Later Phases)**: Read from SQLite → Chunk → Embed → Store in ChromaDB (`MemoryService`)
- Process stored messages offline
- No need to re-fetch from Discord
- Can re-process with different chunking/embedding strategies

### Implementation Steps

#### Step 2.1: Add Rate Limiting Configuration

The rate limiting configuration is already in `config.py` (lines 38-42):

```python
# Discord Message rate limits
MESSAGE_FETCH_DELAY: float = float(os.getenv("MESSAGE_FETCH_DELAY", "1.0"))
MESSAGE_FETCH_BATCH_SIZE: int = int(os.getenv("MESSAGE_FETCH_BATCH_SIZE", "100"))
MESSAGE_FETCH_PROGRESS_INTERVAL: int = int(os.getenv("MESSAGE_FETCH_PROGRESS_INTERVAL", "100"))
MESSAGE_FETCH_MAX_RETRIES: int = int(os.getenv("MESSAGE_FETCH_MAX_RETRIES", "5"))
```

**Configuration Options:**
- `MESSAGE_FETCH_DELAY`: Delay between API requests (default: 1.0s, safe for Discord)
- `MESSAGE_FETCH_BATCH_SIZE`: Messages to collect before storing (default: 100)
- `MESSAGE_FETCH_PROGRESS_INTERVAL`: Progress callback frequency (default: 100 messages)
- `MESSAGE_FETCH_MAX_RETRIES`: Max retries on rate limit errors (default: 5)

#### Step 2.2: Rework MessageLoader to Use MessageStorage

Modify `services/message_loader.py` to use `MessageStorage` instead of `MemoryService`:

```python
import discord
import asyncio
import logging
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime
from discord import HTTPException
from config import Config
from storage.messages import MessageStorage
from utils.discord_utils import format_discord_message


class MessageLoader:
    """
    Load messages from Discord channels and store them in SQLite.
    
    This is Stage 1 of the message processing pipeline:
    - Stage 1: Fetch from Discord → Store in SQLite (this class)
    - Stage 2: Read from SQLite → Chunk → Embed → Store in ChromaDB (later phase)
    """
    
    def __init__(self, message_storage: MessageStorage):
        self.message_storage = message_storage
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = Config.MESSAGE_FETCH_DELAY
        self.batch_size = Config.MESSAGE_FETCH_BATCH_SIZE
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback for progress updates (Observer pattern).
        
        Learning: Callbacks enable decoupled progress reporting.
        """
        self.progress_callback = callback
    
    async def _rate_limit_delay(self):
        """Wait between requests to respect rate limits"""
        await asyncio.sleep(self.rate_limit_delay)
    
    async def _handle_rate_limit_error(
        self, 
        error: HTTPException, 
        retry_count: int
    ) -> bool:
        """
        Handle 429 (rate limit) errors with exponential backoff.
        
        Learning: Exponential backoff prevents overwhelming the API.
        """
        if error.status == 429:  # Too Many Requests
            retry_after = getattr(error, 'retry_after', None) or (2 ** retry_count)
            wait_time = min(retry_after, 60)  # Cap at 60 seconds
            
            self.logger.warning(
                f"Rate limited! Waiting {wait_time}s before retry "
                f"(attempt {retry_count + 1}/{Config.MESSAGE_FETCH_MAX_RETRIES})"
            )
            
            await asyncio.sleep(wait_time)
            return True  # Retry
        return False  # Don't retry
    
    async def load_channel_messages(
        self,
        channel: discord.TextChannel,
        limit: Optional[int] = None,
        before: Optional[discord.Message] = None,
        after: Optional[discord.Message] = None,
        rate_limit_delay: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Load messages from a Discord channel and store them in SQLite.
        
        Uses checkpoint/resume to continue from oldest message if interrupted.
        Fetches messages oldest-first to ensure chronological order.
        Collects messages in batches for efficient storage.
        
        Args:
            channel: The Discord channel to load messages from
            limit: Maximum number of messages to load (None = all)
            before: Load messages before this message (overrides checkpoint)
            after: Load messages after this message (overrides checkpoint)
            rate_limit_delay: Override default rate limit delay
        
        Returns:
            Dict with statistics about the loading operation
        """
        if rate_limit_delay:
            self.rate_limit_delay = rate_limit_delay
        
        channel_id = str(channel.id)
        self.logger.info(f"Loading messages from #{channel.name} ({channel_id})")
        
        # Check for existing checkpoint to resume
        checkpoint = self.message_storage.get_checkpoint(channel_id)
        resume_from_oldest = False
        
        if checkpoint and before is None and after is None:
            # Resume from newest message we've already fetched
            # When fetching oldest_first=True, we've fetched chronologically from oldest to newest
            # So we resume from the newest message (last_message_id) to continue forward
            if checkpoint.get('last_message_id'):
                try:
                    last_message_id = checkpoint.get('last_message_id')
                    last_timestamp = checkpoint.get('last_fetch_timestamp')
                    
                    self.logger.info(
                        f"Resuming from checkpoint: last_message_id={last_message_id}, "
                        f"last_timestamp={last_timestamp}, "
                        f"total_messages={checkpoint['total_messages']}"
                    )
                    
                    # Use last_message_id (newest we've fetched) as the 'after' parameter
                    # This continues fetching messages AFTER the newest we already have
                    try:
                        after_message = await channel.fetch_message(int(last_message_id))
                        after = after_message
                        resume_from_oldest = True
                    except (discord.NotFound, discord.HTTPException):
                        self.logger.warning(
                            f"Could not fetch checkpoint message {last_message_id}, "
                            f"will fetch all and rely on INSERT OR IGNORE"
                        )
                except Exception as e:
                    self.logger.warning(f"Error processing checkpoint: {e}, will start fresh")
        
        # Checkpoint Logic Explanation:
        # The condition "if checkpoint and before is None and after is None" means:
        # - Only auto-resume if checkpoint exists AND user hasn't specified explicit boundaries
        # - If no checkpoint exists: starts fetching from beginning (normal behavior)
        # - If checkpoint exists but before/after specified: respects explicit boundaries (ignores checkpoint)
        # - If checkpoint exists and no boundaries: auto-resumes from last_message_id (newest we've fetched)
        #
        # Scenarios:
        # 1. No checkpoint, no boundaries → Fetch all from beginning ✓
        # 2. Checkpoint exists, no boundaries → Auto-resume from last_message_id ✓
        # 3. Checkpoint exists, explicit boundaries → Use boundaries, ignore checkpoint ✓
        # 4. No checkpoint, explicit boundaries → Use boundaries ✓
        
        stats = {
            "total_processed": 0,
            "successfully_loaded": 0,
            "skipped_bot_messages": 0,
            "skipped_empty_messages": 0,
            "skipped_blacklisted": 0,
            "skipped_commands": 0,
            "errors": 0,
            "rate_limit_errors": 0,
            "batches_saved": 0,
            "resumed_from_checkpoint": resume_from_oldest,
            "start_time": datetime.now(),
            "end_time": None,
        }
        
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
                oldest_first=True  # REQUIRED: Fetch from oldest to newest
            ):
                try:
                    stats["total_processed"] += 1
                    
                    # Skip messages that shouldn't be stored
                    if message.author.bot:
                        stats["skipped_bot_messages"] += 1
                        await self._rate_limit_delay()  # Still respect rate limit
                        continue
                    elif message.author.id in Config.BLACKLIST_IDS:
                        stats["skipped_blacklisted"] += 1
                        await self._rate_limit_delay()
                        continue
                    elif not message.content.strip():
                        stats["skipped_empty_messages"] += 1
                        await self._rate_limit_delay()
                        continue
                    elif message.content.startswith(Config.BOT_PREFIX):
                        stats["skipped_commands"] += 1
                        await self._rate_limit_delay()
                        continue
                    
                    # Format message for storage
                    message_data = format_discord_message(message)
                    message_batch.append(message_data)
                    
                    # Save batch when it reaches batch size
                    if len(message_batch) >= self.batch_size:
                        success = self.message_storage.save_channel_messages(
                            channel_id, message_batch
                        )
                        if success:
                            stats["successfully_loaded"] += len(message_batch)
                            stats["batches_saved"] += 1
                        else:
                            stats["errors"] += len(message_batch)
                        message_batch = []  # Clear batch
                    
                    # Rate limit delay between messages
                    await self._rate_limit_delay()
                    
                    # Progress reporting
                    if stats["total_processed"] % Config.MESSAGE_FETCH_PROGRESS_INTERVAL == 0:
                        if self.progress_callback:
                            elapsed = (datetime.now() - stats["start_time"]).total_seconds()
                            rate = stats["total_processed"] / elapsed if elapsed > 0 else 0
                            progress = {
                                "processed": stats["total_processed"],
                                "limit": limit or "unlimited",
                                "rate": rate,
                                "successful": stats["successfully_loaded"],
                                "channel_name": channel.name,
                                "channel_id": channel_id
                            }
                            if asyncio.iscoroutinefunction(self.progress_callback):
                                await self.progress_callback(progress)
                            else:
                                self.progress_callback(progress)
                    
                    # Reset retry count on success
                    retry_count = 0
                    
                except HTTPException as e:
                    # Handle rate limit errors
                    if await self._handle_rate_limit_error(e, retry_count):
                        stats["rate_limit_errors"] += 1
                        retry_count += 1
                        
                        if retry_count >= Config.MESSAGE_FETCH_MAX_RETRIES:
                            self.logger.error("Max retries exceeded")
                            break
                        continue
                    else:
                        stats["errors"] += 1
                        self.logger.error(f"HTTP error: {e}")
                
                except Exception as e:
                    stats["errors"] += 1
                    self.logger.error(f"Error processing message: {e}")
            
            # Save any remaining messages in the batch
            if message_batch:
                success = self.message_storage.save_channel_messages(
                    channel_id, message_batch
                )
                if success:
                    stats["successfully_loaded"] += len(message_batch)
                    stats["batches_saved"] += 1
                else:
                    stats["errors"] += len(message_batch)
            
            stats["end_time"] = datetime.now()
            duration = (stats["end_time"] - stats["start_time"]).total_seconds()
            
            self.logger.info(
                f"Completed loading from #{channel.name}: "
                f"{stats['successfully_loaded']} stored in {stats['batches_saved']} batches, "
                f"{stats['skipped_bot_messages']} bot messages skipped, "
                f"{stats['skipped_blacklisted']} blacklisted users skipped, "
                f"{stats['skipped_empty_messages']} empty messages skipped, "
                f"{stats['skipped_commands']} commands skipped, "
                f"{stats['rate_limit_errors']} rate limit errors, "
                f"{stats['errors']} errors, "
                f"took {duration:.1f} seconds"
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Fatal error loading messages: {e}")
            stats["end_time"] = datetime.now()
            stats["errors"] += 1
            return stats
```

**Key Changes:**
- Uses `MessageStorage` instead of `MemoryService`
- Collects messages in batches (default: 100)
- Uses `save_channel_messages()` for efficient batch inserts
- **Checkpoint/Resume**: Automatically checks for existing checkpoint and resumes from oldest message
- **Oldest-First Fetching**: Uses `oldest_first=True` to ensure chronological order
- Saves remaining batch at the end

**Key Learning Points:**
- **Two-Stage Pipeline**: Separate fetching from chunking/embedding
- **Batch Collection**: Collect messages before storing (more efficient)
- **Checkpoint/Resume**: Automatically resumes from `oldest_message_id` if checkpoint exists
  - Checks checkpoint before starting
  - Fetches the oldest message we have to use as `after` parameter
  - Falls back to fetching all if checkpoint message can't be fetched (relies on `INSERT OR IGNORE`)
- **Oldest-First Order**: `oldest_first=True` ensures messages are fetched chronologically
  - Important for checkpoint/resume to work correctly
  - Ensures we don't miss messages when resuming
- **Rate Limiting**: `await asyncio.sleep()` between requests
- **Exponential Backoff**: `2^retry_count` seconds, capped at 60s
- **Observer Pattern**: Progress callback for UI updates
- **Error Recovery**: Retry on rate limit, fail on other errors
- **Async Callbacks**: Check if callback is coroutine function

#### Step 2.3: Usage Example

Example of using MessageLoader with MessageStorage:

```python
from bot.loaders.message_loader import MessageLoader
from storage.messages import MessageStorage

# Initialize storage and loader
storage = MessageStorage("data/raw_messages/messages.db")
loader = MessageLoader(storage)

# Set progress callback (optional)
async def on_progress(progress):
    print(f"Progress: {progress['processed']} messages, "
          f"Rate: {progress['rate']:.2f} msg/s, "
          f"Saved: {progress['successful']}")

loader.set_progress_callback(on_progress)

# Load messages from a channel
channel = bot.get_channel(123456789)  # Your channel ID
stats = await loader.load_channel_messages(channel, limit=1000)

# Check results
print(f"✅ Loaded {stats['successfully_loaded']} messages in {stats['batches_saved']} batches")
print(f"   Skipped: {stats['skipped_bot_messages']} bot, "
      f"{stats['skipped_empty_messages']} empty, "
      f"{stats['skipped_commands']} commands")

# Checkpoint allows resuming later
checkpoint = storage.get_checkpoint(str(channel.id))
if checkpoint:
    print(f"Checkpoint: {checkpoint['total_messages']} messages stored")
```

**Checkpoint/Resume Pattern:**

**Important: Understanding Discord API Behavior**

Discord's `channel.history()` API **defaults to newest-to-oldest** (reverse chronological order).
You **must** use `oldest_first=True` to fetch chronologically from oldest to newest.

**Understanding Checkpoint Fields (CRITICAL CONCEPT):**

⚠️ **Don't confuse "last" (position in fetch) with "newest/oldest" (position in time)!**

When fetching with `oldest_first=True`, we process messages chronologically from past to present:

| Field | Time Position | Fetch Position | Purpose |
|-------|---------------|----------------|---------|
| `oldest_message_id` | Earliest in time | First fetched | Metadata only (stats/reporting) |
| `last_message_id` | Latest in time | Last fetched | **Required for resume** |
| `last_fetch_timestamp` | Latest in time | Last fetched | **Required for resume** |
| `total_messages` | N/A | N/A | Stats/reporting |

**Visual Timeline Example:**

```
TIME →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
        [1 year ago]  [3 months ago]  [1 month ago]  [now]
              ↑              ↑
        oldest_msg    last_msg       <-- We stopped here
        (first        (last
         fetched)      fetched)

To resume: Fetch AFTER last_msg (3 months ago) → [1 month ago] → [now]
```

**Why we use `last_message_id` (NOT `oldest_message_id`) for resume:**

With `oldest_first=True`:
1. We fetch chronologically: [1 year ago] → [3 months ago] (stopped)
2. Checkpoint stores:
   - `oldest_message_id` = 1 year ago ← **Just metadata** (shows our starting point)
   - `last_message_id` = 3 months ago ← **Resume point** (where we stopped)
3. To resume: Use `after=last_message_id` (3 months ago)
4. Continue fetching: [3 months ago] → [1 month ago] → [now]

**Key Insight:** `last_message_id` is the **most recent message in time** we've already fetched, so we resume AFTER it to get newer messages.

**The Checkpoint Logic:**

```python
# Checkpoint check happens before fetching
checkpoint = self.message_storage.get_checkpoint(channel_id)

if checkpoint and before is None and after is None:
    # Auto-resume from checkpoint
    # Explanation:
    # - last_message_id = the newest message we've already fetched
    # - We want to fetch messages AFTER this (i.e., newer messages)
    # - oldest_first=True ensures we continue chronologically forward
    after_message = await channel.fetch_message(int(checkpoint['last_message_id']))
    after = after_message  # Continue fetching AFTER this message
```

**Terminology Clarification:**

| Term | Meaning |
|------|---------|
| "oldest" | Earliest in **chronological time** (furthest in the past) |
| "newest" / "latest" | Latest in **chronological time** (most recent) |
| "first" | Position in **fetch order** (what we fetched first) |
| "last" | Position in **fetch order** (what we fetched last) |

When using `oldest_first=True`:
- **First fetched** = **Oldest in time** ✓
- **Last fetched** = **Newest in time** ✓

This alignment is why we name the field `last_message_id` (it's both the last fetched AND the newest in time).

**Important: Understanding the Condition**

The condition `if checkpoint and before is None and after is None` means:
- **`checkpoint`**: Must exist (not `None`)
- **`before is None`**: User hasn't specified an explicit "before" boundary
- **`after is None`**: User hasn't specified an explicit "after" boundary

**All Possible Scenarios:**

| Scenario | Checkpoint Exists? | `before`/`after` Specified? | Behavior |
|----------|-------------------|----------------------------|----------|
| **First run** | ❌ No | ❌ No | Fetches all messages from beginning (oldest-first) |
| **Resume** | ✅ Yes | ❌ No | **Auto-resumes** from `last_message_id` in checkpoint |
| **Explicit bounds (with checkpoint)** | ✅ Yes | ✅ Yes | **Uses explicit bounds**, ignores checkpoint (user's intent takes priority) |
| **Explicit bounds (no checkpoint)** | ❌ No | ✅ Yes | Uses explicit bounds normally |

**Key Insights:** 
- If there's **no checkpoint**, the condition is `False` (because `None` is falsy), so the code skips the resume block and fetches normally
- The checkpoint is **only used** when it exists AND the user hasn't given explicit boundaries
- This allows users to override checkpoint behavior by specifying `before`/`after` parameters
- **For resume**: We use `last_message_id` (newest fetched), NOT `oldest_message_id` (oldest fetched)
- **`oldest_message_id` is metadata**: Useful for reporting "we have messages from X to Y", but not needed for resume logic

**Example Usage:**

```python
# Scenario 1: First run - no checkpoint exists
stats = await loader.load_channel_messages(channel, limit=None)
# Logs: "Loading messages from #channel-name (123456789)"
# Behavior: Fetches all from beginning (oldest-first)
# Result: Checkpoint created after first batch with:
#   - last_message_id = newest message in batch
#   - oldest_message_id = oldest message in batch (metadata)

# Scenario 2: Resume - checkpoint exists, no explicit boundaries
stats = await loader.load_channel_messages(channel, limit=None)
# Logs: "Resuming from checkpoint: last_message_id=..., last_timestamp=..., total_messages=..."
# Behavior: Fetches messages AFTER the newest message we already have
# Uses oldest_first=True to continue chronologically
# Result: Checkpoint updated with new last_message_id

# Scenario 3: Override checkpoint - explicit boundaries specified
stats = await loader.load_channel_messages(channel, before=some_message, limit=100)
# Logs: "Loading messages from #channel-name (123456789)"
# Behavior: Uses explicit `before` parameter, ignores checkpoint
# Result: Fetches 100 messages before `some_message`

# Check if resume happened
if stats.get('resumed_from_checkpoint'):
    print("✅ Resumed from checkpoint")
else:
    print("🆕 Started fresh")
```

**How Checkpoint/Resume Works:**
1. Before fetching, loader checks `MessageStorage.get_checkpoint(channel_id)`
2. If checkpoint exists and no explicit `before`/`after` parameters:
   - Fetches the `last_message_id` from checkpoint (newest message we've fetched)
   - Uses it as the `after` parameter in `channel.history()`
   - This ensures we only fetch messages newer than what we already have
3. If checkpoint message can't be fetched (deleted, etc.):
   - Falls back to fetching all messages
   - Relies on `INSERT OR IGNORE` to skip duplicates
4. Messages are fetched with `oldest_first=True`:
   - Ensures chronological order (oldest to newest)
   - Makes checkpoint/resume logic straightforward
   - We resume from the newest message we have (`last_message_id`), continuing forward in time

**What's Required vs Optional in Checkpoint:**

| Field | Required for Resume? | Purpose |
|-------|---------------------|---------|
| `last_message_id` | ✅ **Yes** | Resume from this message (newest we've fetched) |
| `last_fetch_timestamp` | ✅ **Yes** | Timestamp for resume logic |
| `total_messages` | ❌ No | Reporting/statistics only |
| `oldest_message_id` | ❌ No | **Metadata only** - for stats/reporting "messages from X to Y" |
| `oldest_message_timestamp` | ❌ No | **Metadata only** - for stats/reporting |

### Common Pitfalls - Phase 2

1. **Forgetting await**: `asyncio.sleep()` must be awaited
2. **Not checking callback type**: Use `iscoroutinefunction()` for async callbacks
3. **Rate limit too low**: < 0.5s will get you banned
4. **Not handling retry_after**: Discord tells you how long to wait
5. **Progress updates too frequent**: Can slow down fetching
6. **Wrong fetch order**: Must use `oldest_first=True` for checkpoint/resume to work correctly
7. **Not handling checkpoint message deletion**: If checkpoint message is deleted, fall back gracefully
8. **Forgetting to save final batch**: Always save remaining messages after loop completes

### Debugging Tips - Phase 2

- **Monitor rate**: Log actual request rate
- **Check retry_after**: Discord sends this in 429 responses
- **Test with small limits**: Start with 100 messages
- **Watch for bans**: If you get 403, you're banned temporarily

### Performance Considerations - Phase 2

- **Rate limit delay**: 1.0s is safe, 0.5s is risky
- **Progress updates**: Every 100 messages is good balance
- **Batch size**: 100 messages per batch is efficient (balances memory vs database calls)
- **Batch inserts**: `save_channel_messages()` is 100x faster than individual inserts
- **Checkpoint overhead**: Minimal - only updates after each batch save

### Benefits of Two-Stage Design

1. **Separation of Concerns**: Fetching/storage separate from chunking/embedding
2. **Resumability**: Can pause and resume large message loads using checkpoints
   - Automatically resumes from oldest message if interrupted
   - No need to re-fetch already stored messages
   - Handles very long channel histories gracefully
3. **Chronological Order**: `oldest_first=True` ensures messages are processed in order
   - Makes checkpoint logic straightforward
   - Ensures no messages are missed when resuming
4. **Efficiency**: Batch inserts much faster than individual operations
5. **Flexibility**: Can re-process SQLite data later without re-fetching from Discord
6. **Rate Limit Safety**: Respects Discord API limits without embedding overhead
7. **Offline Processing**: Can chunk and embed stored messages later without Discord API calls
8. **Robustness**: Handles checkpoint message deletion gracefully (falls back to full fetch with duplicate detection)