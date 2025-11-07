# Phase 19: Production Automation & Background Tasks

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

## Overview

**Time Estimate:** 3-4 hours
**Difficulty:** ⭐⭐⭐ (Intermediate-Advanced)
**Prerequisites:** Phase 6.8 (Incremental Sync), Phase 18 (Security), Deployment Setup

Transform your bot from a manual, command-driven system into a fully automated production service that continuously syncs messages, monitors health, and recovers from failures - perfect for cloud deployment!

### Learning Objectives
- Implement background tasks with discord.py tasks extension
- Design real-time message processing with event listeners
- Build health monitoring and alerting systems
- Handle failures gracefully with retry logic
- Deploy automated services to cloud platforms

### Why This Matters

**Manual Bot (Development):**
```
User: !chunk_channel #general
Bot: Processing 1000 messages... done!

[10 minutes later, 50 new messages arrive]

User: !sync_vectors all
Bot: Processing 50 new messages... done!

❌ Requires constant manual intervention
❌ New messages sit unprocessed for hours
❌ No monitoring or alerts
```

**Automated Bot (Production):**
```
[Bot starts]
✅ Auto-sync running every 30 minutes
✅ Real-time message processing
✅ Health checks every 5 minutes
✅ Alerts on failures
✅ Metrics dashboard

[No user intervention needed!]
```

---

## Part 1: Background Sync Tasks

### Design Principles

Production background tasks need:
- **Scheduled Execution** - Run at fixed intervals (cron-like)
- **Error Handling** - Don't crash on transient failures
- **Graceful Shutdown** - Stop cleanly when bot restarts
- **Idempotency** - Safe to run multiple times
- **Monitoring** - Track last run time, success/failure

### Step 19.1: Automated Incremental Sync

Create `bot/tasks/auto_sync.py`:

```python
"""
Background task for automatic vector DB synchronization.

Learning: discord.py tasks extension provides cron-like scheduling.
Perfect for periodic maintenance tasks in production.
"""

from discord.ext import tasks, commands
from storage.chunked_memory import ChunkedMemoryService
from chunking.service import ChunkingService
from embedding.factory import EmbeddingServiceFactory
from storage.vectors.factory import VectorStoreFactory
from storage.sync_tracker import SyncCheckpoint
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


class AutoSyncTask(commands.Cog):
    """
    Automatically sync vector DBs with new messages.

    Learning: This runs in the background without user intervention.
    Critical for keeping RAG up-to-date in production!
    """

    def __init__(self, bot, sync_interval_minutes: int = 30):
        self.bot = bot
        self.sync_interval_minutes = sync_interval_minutes

        # Initialize services (lazy loaded)
        self._chunked_memory = None
        self._chunking_service = None

        # Metrics
        self.last_sync_time = None
        self.total_syncs = 0
        self.failed_syncs = 0

        # Start the task
        self.auto_sync.change_interval(minutes=sync_interval_minutes)
        self.auto_sync.start()

        logger.info(f"AutoSync task initialized (interval: {sync_interval_minutes}m)")

    def _get_services(self):
        """Lazy load services (only when needed)."""
        if not self._chunked_memory:
            embedding_provider = EmbeddingServiceFactory.create()
            vector_store = VectorStoreFactory.create()

            self._chunked_memory = ChunkedMemoryService(
                vector_store=vector_store,
                embedding_provider=embedding_provider
            )

            self._chunking_service = ChunkingService()

        return self._chunked_memory, self._chunking_service

    @tasks.loop(minutes=30)  # Default, overridden in __init__
    async def auto_sync(self):
        """
        Automatically sync vector databases with new messages.

        Runs every N minutes (configurable).
        Only syncs strategies that have new messages (efficient!).
        """
        try:
            logger.info("Starting auto-sync...")
            start_time = datetime.now()

            chunked_memory, chunking_service = self._get_services()

            # Strategies to auto-sync (you can configure this)
            strategies = ["temporal", "conversation"]  # Sync critical strategies only

            total_new_messages = 0
            total_new_chunks = 0

            for strategy in strategies:
                try:
                    result = await chunked_memory.sync_strategy_incremental(
                        strategy_name=strategy,
                        chunking_service=chunking_service
                    )

                    if not result["skipped"]:
                        total_new_messages += result["new_messages_processed"]
                        total_new_chunks += result["new_chunks_created"]

                        logger.info(
                            f"Auto-synced {strategy}: "
                            f"+{result['new_messages_processed']} msgs → "
                            f"+{result['new_chunks_created']} chunks"
                        )

                except Exception as e:
                    logger.error(f"Auto-sync failed for {strategy}: {e}", exc_info=True)
                    self.failed_syncs += 1

            duration = (datetime.now() - start_time).total_seconds()

            if total_new_messages > 0:
                logger.info(
                    f"Auto-sync complete: "
                    f"{total_new_messages} msgs → {total_new_chunks} chunks "
                    f"in {duration:.2f}s"
                )
            else:
                logger.debug("Auto-sync: No new messages")

            self.last_sync_time = datetime.now()
            self.total_syncs += 1

        except Exception as e:
            logger.error(f"Auto-sync task failed: {e}", exc_info=True)
            self.failed_syncs += 1

    @auto_sync.before_loop
    async def before_auto_sync(self):
        """Wait until bot is ready before starting sync loop."""
        await self.bot.wait_until_ready()
        logger.info("Bot ready - auto-sync task starting...")

    @auto_sync.error
    async def auto_sync_error(self, error):
        """Handle errors in auto-sync task."""
        logger.error(f"Auto-sync error handler: {error}", exc_info=True)
        # Don't crash - just log and continue

    def cog_unload(self):
        """Stop task when cog is unloaded."""
        self.auto_sync.cancel()
        logger.info("Auto-sync task stopped")

    @commands.command(name='autosync_status')
    async def autosync_status(self, ctx):
        """Show auto-sync task status."""
        import discord

        embed = discord.Embed(
            title="🔄 Auto-Sync Status",
            color=discord.Color.blue()
        )

        # Task status
        status = "✅ Running" if self.auto_sync.is_running() else "❌ Stopped"
        embed.add_field(name="Status", value=status, inline=True)

        # Interval
        embed.add_field(
            name="Interval",
            value=f"{self.sync_interval_minutes} minutes",
            inline=True
        )

        # Last sync
        if self.last_sync_time:
            last_sync = self.last_sync_time.strftime("%Y-%m-%d %H:%M:%S")
            embed.add_field(name="Last Sync", value=last_sync, inline=True)

        # Metrics
        success_rate = (
            (self.total_syncs - self.failed_syncs) / max(self.total_syncs, 1) * 100
        )

        embed.add_field(
            name="📊 Metrics",
            value=(
                f"**Total Syncs:** {self.total_syncs}\n"
                f"**Failed:** {self.failed_syncs}\n"
                f"**Success Rate:** {success_rate:.1f}%"
            ),
            inline=False
        )

        # Next run
        if self.auto_sync.next_iteration:
            next_run = self.auto_sync.next_iteration.strftime("%Y-%m-%d %H:%M:%S")
            embed.set_footer(text=f"Next run: {next_run}")

        await ctx.send(embed=embed)


async def setup(bot):
    """Load the auto-sync cog."""
    # Get sync interval from config (default: 30 minutes)
    from config import Config
    interval = getattr(Config, 'AUTO_SYNC_INTERVAL_MINUTES', 30)

    await bot.add_cog(AutoSyncTask(bot, sync_interval_minutes=interval))
```

---

## Part 2: Real-Time Message Processing

### Step 19.2: Event-Driven Message Capture

Create `bot/listeners/message_listener.py`:

```python
"""
Real-time message processing listener.

Learning: Event listeners enable immediate processing of new messages
instead of polling/batch processing.
"""

from discord.ext import commands
from storage.messages import MessageStorage
from bot.utils.discord_utils import format_discord_message
import logging

logger = logging.getLogger(__name__)


class MessageListener(commands.Cog):
    """
    Listen for new messages and store them in real-time.

    Learning: This ensures the message DB is always up-to-date.
    The auto-sync task (Phase 19.1) will then chunk and embed them.
    """

    def __init__(self, bot):
        self.bot = bot
        self.message_storage = MessageStorage()
        self.messages_stored = 0

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        Event: New message posted in any channel.

        Learning: This fires for EVERY message the bot can see.
        Need to filter carefully to avoid processing bot messages!
        """

        # Skip bot messages
        if message.author.bot:
            return

        # Skip DMs (optional - you might want to support DMs)
        if not message.guild:
            return

        # Skip empty messages
        if not message.content.strip():
            return

        try:
            # Format message for storage
            msg_data = format_discord_message(message)

            # Store in SQLite
            success = self.message_storage.store_message(msg_data)

            if success:
                self.messages_stored += 1
                logger.debug(
                    f"Stored message {message.id} from {message.author.name}"
                )

        except Exception as e:
            logger.error(f"Failed to store message {message.id}: {e}")

    @commands.Cog.listener()
    async def on_message_edit(self, before, after):
        """
        Event: Message edited.

        Learning: Should we update stored messages?
        - YES: If you want accurate history
        - NO: If you want immutable audit log

        Default: Update the message
        """
        if after.author.bot:
            return

        try:
            msg_data = format_discord_message(after)
            msg_data['is_edit'] = True
            msg_data['edited_at'] = after.edited_at.isoformat() if after.edited_at else None

            self.message_storage.store_message(msg_data)

            logger.debug(f"Updated edited message {after.id}")

        except Exception as e:
            logger.error(f"Failed to update edited message {after.id}: {e}")

    @commands.Cog.listener()
    async def on_message_delete(self, message):
        """
        Event: Message deleted.

        Learning: You might want to:
        - Mark as deleted (soft delete)
        - Remove from DB (hard delete)
        - Keep for audit purposes

        Default: Soft delete (mark as deleted)
        """
        if message.author.bot:
            return

        try:
            # Mark as deleted in DB (you'd need to add this column)
            # For now, just log
            logger.info(f"Message {message.id} deleted (not removed from DB)")

        except Exception as e:
            logger.error(f"Failed to handle deleted message {message.id}: {e}")

    @commands.command(name='listener_stats')
    async def listener_stats(self, ctx):
        """Show message listener statistics."""
        import discord

        embed = discord.Embed(
            title="📨 Message Listener Stats",
            color=discord.Color.green()
        )

        embed.add_field(
            name="Messages Stored (This Session)",
            value=f"{self.messages_stored:,}",
            inline=False
        )

        # Get total from DB
        total_in_db = self.message_storage.count_all_messages()

        embed.add_field(
            name="Total in Database",
            value=f"{total_in_db:,}",
            inline=False
        )

        await ctx.send(embed=embed)


async def setup(bot):
    """Load the message listener cog."""
    await bot.add_cog(MessageListener(bot))
```

---

## Part 3: Health Monitoring

### Step 19.3: Health Check Task

Create `bot/tasks/health_monitor.py`:

```python
"""
Health monitoring for production deployments.

Learning: Production systems need health checks for:
- Container orchestration (K8s, ECS)
- Uptime monitoring (UptimeRobot, Pingdom)
- Alerting (PagerDuty, Slack)
"""

from discord.ext import tasks, commands
from storage.sync_tracker import SyncCheckpoint
from storage.messages import MessageStorage
from datetime import datetime, timedelta
import logging
import psutil  # pip install psutil
import os

logger = logging.getLogger(__name__)


class HealthMonitor(commands.Cog):
    """
    Monitor bot health and system resources.

    Learning: Production systems need continuous health monitoring!
    """

    def __init__(self, bot):
        self.bot = bot
        self.health_checks_run = 0
        self.health_check_failures = []
        self.start_time = datetime.now()

        # Start health check task
        self.health_check.start()

    @tasks.loop(minutes=5)
    async def health_check(self):
        """
        Run health checks every 5 minutes.

        Checks:
        - Bot is connected to Discord
        - Database is accessible
        - Sync is not stale (last sync < 2 hours ago)
        - Memory usage is reasonable
        - Disk space is available
        """
        try:
            checks = {
                "discord_connected": self._check_discord(),
                "database_accessible": self._check_database(),
                "sync_not_stale": self._check_sync_freshness(),
                "memory_ok": self._check_memory(),
                "disk_ok": self._check_disk()
            }

            all_healthy = all(checks.values())

            if all_healthy:
                logger.info("✅ Health check passed")
            else:
                failures = [k for k, v in checks.items() if not v]
                logger.warning(f"⚠️ Health check failures: {failures}")
                self.health_check_failures.append({
                    "timestamp": datetime.now(),
                    "failures": failures
                })

                # Alert (could send to Slack, email, etc.)
                await self._send_alert(failures)

            self.health_checks_run += 1

        except Exception as e:
            logger.error(f"Health check error: {e}", exc_info=True)

    def _check_discord(self) -> bool:
        """Check if bot is connected to Discord."""
        return self.bot.is_ready() and not self.bot.is_closed()

    def _check_database(self) -> bool:
        """Check if database is accessible."""
        try:
            storage = MessageStorage()
            storage.count_all_messages()  # Simple query
            return True
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False

    def _check_sync_freshness(self) -> bool:
        """Check if sync is not too stale."""
        try:
            tracker = SyncCheckpoint()
            stats = tracker.get_sync_stats()

            if not stats["most_recent_sync"]:
                return True  # No sync yet is OK

            last_sync = datetime.fromisoformat(stats["most_recent_sync"])
            age = datetime.now() - last_sync

            # Alert if last sync was > 2 hours ago
            return age < timedelta(hours=2)

        except Exception as e:
            logger.error(f"Sync freshness check failed: {e}")
            return False

    def _check_memory(self) -> bool:
        """Check memory usage is reasonable."""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Alert if using > 1GB (adjust for your needs)
            if memory_mb > 1024:
                logger.warning(f"High memory usage: {memory_mb:.0f} MB")
                return False

            return True

        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return True  # Don't fail health check on monitoring errors

    def _check_disk(self) -> bool:
        """Check disk space is available."""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / 1024 / 1024 / 1024

            # Alert if < 1GB free
            if free_gb < 1:
                logger.warning(f"Low disk space: {free_gb:.2f} GB free")
                return False

            return True

        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return True

    async def _send_alert(self, failures):
        """
        Send alert on health check failures.

        Learning: In production, you'd send to:
        - Slack webhook
        - Email
        - PagerDuty
        - Discord channel for alerts

        For now, just log.
        """
        logger.error(f"🚨 ALERT: Health check failures: {failures}")

        # Optional: Send to a Discord channel
        # alert_channel = self.bot.get_channel(ALERT_CHANNEL_ID)
        # if alert_channel:
        #     await alert_channel.send(f"⚠️ Health check failures: {failures}")

    @health_check.before_loop
    async def before_health_check(self):
        """Wait until bot is ready."""
        await self.bot.wait_until_ready()

    def cog_unload(self):
        """Stop health check task."""
        self.health_check.cancel()

    @commands.command(name='health')
    async def health_status(self, ctx):
        """Show bot health status."""
        import discord

        embed = discord.Embed(
            title="🏥 Bot Health Status",
            color=discord.Color.green()
        )

        # Uptime
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds

        embed.add_field(name="⏱️ Uptime", value=uptime_str, inline=True)

        # Health checks
        embed.add_field(
            name="✅ Health Checks",
            value=f"{self.health_checks_run} run",
            inline=True
        )

        # Recent failures
        recent_failures = [
            f for f in self.health_check_failures
            if (datetime.now() - f['timestamp']) < timedelta(hours=24)
        ]

        embed.add_field(
            name="⚠️ Failures (24h)",
            value=len(recent_failures),
            inline=True
        )

        # System resources
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=1.0)

        embed.add_field(
            name="💾 Memory",
            value=f"{memory_mb:.0f} MB",
            inline=True
        )

        embed.add_field(
            name="⚡ CPU",
            value=f"{cpu_percent:.1f}%",
            inline=True
        )

        # Disk
        disk = psutil.disk_usage('/')
        free_gb = disk.free / 1024 / 1024 / 1024

        embed.add_field(
            name="💽 Disk Free",
            value=f"{free_gb:.2f} GB",
            inline=True
        )

        await ctx.send(embed=embed)


async def setup(bot):
    """Load the health monitor cog."""
    await bot.add_cog(HealthMonitor(bot))
```

---

## Part 4: Bot Setup & Configuration

### Step 19.4: Update Main Bot File

Update `bot.py`:

```python
"""
Main bot entry point with background tasks.
"""

import discord
from discord.ext import commands
import logging
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(
    command_prefix=Config.COMMAND_PREFIX,
    intents=intents,
    description="Discord RAG Chatbot with Background Automation"
)


@bot.event
async def on_ready():
    """Bot is ready and connected."""
    logger.info(f"✅ Bot connected as {bot.user.name} ({bot.user.id})")
    logger.info(f"Connected to {len(bot.guilds)} guild(s)")

    # Load background tasks
    await load_background_tasks()


async def load_background_tasks():
    """Load all background task cogs."""
    background_cogs = [
        'bot.tasks.auto_sync',        # Auto-sync vector DBs
        'bot.listeners.message_listener',  # Real-time message capture
        'bot.tasks.health_monitor',   # Health monitoring
    ]

    for cog in background_cogs:
        try:
            await bot.load_extension(cog)
            logger.info(f"✅ Loaded background task: {cog}")
        except Exception as e:
            logger.error(f"❌ Failed to load {cog}: {e}")


async def load_cogs():
    """Load all command cogs."""
    cogs = [
        'bot.cogs.basic',
        'bot.cogs.admin',
        'bot.cogs.summary',
        # Add more cogs as needed
    ]

    for cog in cogs:
        try:
            await bot.load_extension(cog)
            logger.info(f"✅ Loaded cog: {cog}")
        except Exception as e:
            logger.error(f"❌ Failed to load {cog}: {e}")


@bot.event
async def on_error(event, *args, **kwargs):
    """Global error handler."""
    logger.error(f"Error in {event}", exc_info=True)


async def main():
    """Main entry point."""
    async with bot:
        # Load cogs
        await load_cogs()

        # Start bot
        logger.info("🚀 Starting bot...")
        await bot.start(Config.DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## Part 5: Configuration

### Step 19.5: Add Config Options

Update `config.py`:

```python
# Add these to your Config class:

# Background task settings
AUTO_SYNC_INTERVAL_MINUTES = int(os.getenv("AUTO_SYNC_INTERVAL_MINUTES", "30"))
AUTO_SYNC_STRATEGIES = os.getenv("AUTO_SYNC_STRATEGIES", "temporal,conversation").split(",")

# Health monitoring
HEALTH_CHECK_INTERVAL_MINUTES = int(os.getenv("HEALTH_CHECK_INTERVAL_MINUTES", "5"))
ALERT_CHANNEL_ID = os.getenv("ALERT_CHANNEL_ID")  # Optional: Discord channel for alerts

# Resource limits
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "1024"))
MIN_DISK_GB = int(os.getenv("MIN_DISK_GB", "1"))
```

Update `.env`:

```bash
# Background Tasks
AUTO_SYNC_INTERVAL_MINUTES=30
AUTO_SYNC_STRATEGIES=temporal,conversation
HEALTH_CHECK_INTERVAL_MINUTES=5

# Alerts (optional)
ALERT_CHANNEL_ID=1234567890

# Resource Limits
MAX_MEMORY_MB=1024
MIN_DISK_GB=1
```

---

## Part 6: Production Deployment

### Step 19.6: Update Deployment Files

Update `Dockerfile` to include health check endpoint:

```dockerfile
# ... existing Dockerfile ...

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; import psutil; sys.exit(0 if psutil.virtual_memory().percent < 90 else 1)"
```

Update `docker-compose.yml`:

```yaml
services:
  bot:
    build: .
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; import psutil; sys.exit(0 if psutil.virtual_memory().percent < 90 else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

---

## Testing in Production

### Local Testing

```bash
# 1. Start bot
python bot.py

# Watch logs
tail -f logs/bot.log

# Expected output:
# ✅ Bot connected as YourBot (12345)
# ✅ Loaded background task: bot.tasks.auto_sync
# ✅ Loaded background task: bot.listeners.message_listener
# ✅ Loaded background task: bot.tasks.health_monitor
# Auto-sync task starting...
# Health check starting...
```

### Monitor Background Tasks

```bash
# In Discord:
!autosync_status
# 🔄 Auto-Sync Status
#    Status: ✅ Running
#    Interval: 30 minutes
#    Last Sync: 2025-01-07 10:30:00
#    Success Rate: 100%

!listener_stats
# 📨 Message Listener Stats
#    Messages Stored: 150
#    Total in Database: 10,150

!health
# 🏥 Bot Health Status
#    Uptime: 2:15:30
#    Health Checks: 27 run
#    Failures (24h): 0
#    Memory: 245 MB
#    CPU: 2.3%
#    Disk Free: 15.2 GB
```

---

## Cloud Deployment

### Railway Deployment

```bash
# Push to GitHub
git push origin main

# Deploy on Railway:
# 1. Connect GitHub repo
# 2. Add environment variables from .env
# 3. Deploy
# 4. Monitor logs for:
#    ✅ Auto-sync running
#    ✅ Health checks passing
```

### Monitoring in Production

Set up external monitoring:

1. **UptimeRobot** - Ping health endpoint every 5 minutes
2. **Sentry** - Error tracking and alerting
3. **Grafana** - Metrics dashboard (optional)

---

## Key Takeaways

✅ **Background tasks** = Automated, hands-free operation
✅ **Event listeners** = Real-time message capture
✅ **Health monitoring** = Early warning system
✅ **Resource limits** = Prevent runaway processes
✅ **Production-ready** = Deploy once, runs forever

**Impact:**
- **Zero manual intervention** needed
- **Real-time updates** (messages processed as they arrive)
- **Automatic recovery** from transient failures
- **Monitoring & alerting** for peace of mind

**What You've Built:**
A fully automated, production-grade Discord RAG bot that:
- ✅ Captures messages in real-time
- ✅ Syncs vector DBs automatically
- ✅ Monitors its own health
- ✅ Alerts on failures
- ✅ Runs efficiently in the cloud

🎉 **Congratulations!** You've completed the full Discord RAG Chatbot implementation - from basic bot to production-grade automated system!

---

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)
