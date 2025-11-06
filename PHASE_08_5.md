# Phase 8.5: Engagement Features - Leaderboards & Bookmarking

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)

---

## Overview

**Time Estimate:** 6 hours
**Difficulty:** ⭐⭐ (Beginner-Intermediate)
**Prerequisites:** Phase 7.5 (UX & Performance)

Build sticky features that keep users coming back: competitive leaderboards and personal knowledge management through bookmarking. These features transform your bot from a tool into an engaging platform.

### Learning Objectives
- Implement gamification patterns for user engagement
- Build ranking/leaderboard systems with efficient queries
- Create personal knowledge management features
- Design tag-based organization systems
- Understand user retention mechanics

### Why This Matters

**The Problem:**
- Users try bot once, then forget about it
- No social element → no engagement
- Important information gets lost in chat history
- No way to organize/retrieve valuable content

**The Solution:**
- Leaderboards → competition → repeated usage
- Rankings → status → social motivation
- Bookmarks → personal library → utility value
- Tags → organization → findability

---

## Part 1: Leaderboard System

### Design Principles

Leaderboards work because they tap into:
- **Competition** - People want to rank high
- **Transparency** - See where you stand
- **Goals** - Something to work toward
- **Community** - Shared participation

### Step 8.5.1: Multi-Category Leaderboards

Update `services/user_ai_tracker.py` to support ranking:

```python
"""
Enhanced user AI tracker with leaderboard support.
"""

from typing import Dict, List, Tuple
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class UserAITracker:
    """Track user AI usage with leaderboard functionality."""

    def __init__(self, storage_file: str = "data/user_ai_stats.json"):
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.user_stats: Dict[str, Dict] = {}
        self.load_stats()

    def log_ai_usage(
        self,
        user_display_name: str,
        cost: float,
        tokens_total: int,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        model: str = "unknown"
    ):
        """Log AI usage for a user."""
        if user_display_name not in self.user_stats:
            self.user_stats[user_display_name] = {
                "lifetime_cost": 0.0,
                "lifetime_tokens": 0,
                "lifetime_credit": 0.0,
                "query_count": 0,
                "first_query_date": datetime.now().isoformat(),
                "last_query_date": datetime.now().isoformat(),
                "models_used": {},
                "daily_usage": {}  # Track daily stats
            }

        user = self.user_stats[user_display_name]

        # Update totals
        user["lifetime_cost"] += cost
        user["lifetime_tokens"] += tokens_total
        user["query_count"] += 1
        user["last_query_date"] = datetime.now().isoformat()

        # Track model usage
        if model not in user["models_used"]:
            user["models_used"][model] = 0
        user["models_used"][model] += 1

        # Track daily usage
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in user["daily_usage"]:
            user["daily_usage"][today] = {
                "cost": 0.0,
                "tokens": 0,
                "queries": 0
            }
        user["daily_usage"][today]["cost"] += cost
        user["daily_usage"][today]["tokens"] += tokens_total
        user["daily_usage"][today]["queries"] += 1

        # Calculate social credit (more usage = more credit)
        # Base credit: 10 per query
        # Bonus: Efficiency (high tokens per dollar)
        credit_gain = 10
        if cost > 0:
            efficiency = tokens_total / cost
            if efficiency > 100000:  # Very efficient usage
                credit_gain += 5

        user["lifetime_credit"] += credit_gain

        self.save_stats()

    def get_user_stats(self, user_display_name: str) -> Dict:
        """Get stats for a specific user."""
        return self.user_stats.get(user_display_name)

    def get_all_users_stats(self) -> Dict[str, Dict]:
        """Get stats for all users."""
        return self.user_stats

    def get_leaderboard(
        self,
        category: str = "credit",
        limit: int = 10
    ) -> List[Tuple[str, Dict]]:
        """
        Get leaderboard rankings for a category.

        Args:
            category: One of: credit, cost, tokens, queries, efficiency
            limit: Number of top users to return

        Returns:
            List of (username, stats) tuples sorted by category
        """
        valid_categories = ["credit", "cost", "tokens", "queries", "efficiency"]
        if category not in valid_categories:
            raise ValueError(f"Invalid category. Choose from: {valid_categories}")

        # Sort users by category
        if category == "efficiency":
            # Efficiency = tokens per dollar
            sorted_users = sorted(
                self.user_stats.items(),
                key=lambda x: x[1]["lifetime_tokens"] / max(x[1]["lifetime_cost"], 0.000001),
                reverse=True
            )
        else:
            # Map category to stat key
            key_map = {
                "credit": "lifetime_credit",
                "cost": "lifetime_cost",
                "tokens": "lifetime_tokens",
                "queries": "query_count"
            }
            stat_key = key_map[category]

            sorted_users = sorted(
                self.user_stats.items(),
                key=lambda x: x[1][stat_key],
                reverse=True
            )

        return sorted_users[:limit]

    def get_user_rank(
        self,
        user_display_name: str,
        category: str = "credit"
    ) -> int:
        """Get user's rank in a specific category."""
        leaderboard = self.get_leaderboard(category, limit=1000)

        for rank, (username, _) in enumerate(leaderboard, start=1):
            if username == user_display_name:
                return rank

        return len(leaderboard) + 1

    def get_global_stats(self) -> Dict:
        """Get global statistics across all users."""
        if not self.user_stats:
            return {
                "total_users": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_queries": 0,
                "avg_cost_per_user": 0.0,
                "avg_tokens_per_user": 0.0
            }

        total_cost = sum(u["lifetime_cost"] for u in self.user_stats.values())
        total_tokens = sum(u["lifetime_tokens"] for u in self.user_stats.values())
        total_queries = sum(u["query_count"] for u in self.user_stats.values())
        user_count = len(self.user_stats)

        return {
            "total_users": user_count,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_queries": total_queries,
            "avg_cost_per_user": total_cost / user_count,
            "avg_tokens_per_user": total_tokens / user_count,
            "avg_queries_per_user": total_queries / user_count
        }

    def save_stats(self):
        """Save stats to JSON file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.user_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user stats: {e}")

    def load_stats(self):
        """Load stats from JSON file."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    self.user_stats = json.load(f)
                logger.info(f"Loaded stats for {len(self.user_stats)} users")
            except Exception as e:
                logger.error(f"Failed to load user stats: {e}")
                self.user_stats = {}
```

---

### Step 8.5.2: Leaderboard Commands

Add to `cogs/basic.py`:

```python
@commands.command(name="leaderboard", aliases=['top', 'rankings', 'lb'])
async def leaderboard(self, ctx, category: str = "credit", top_n: int = 10):
    """
    Show server leaderboards.

    Categories: credit, cost, tokens, queries, efficiency

    Usage:
        !leaderboard credit 10
        !lb tokens 5
        !rankings efficiency
    """
    valid_categories = ["credit", "cost", "tokens", "queries", "efficiency"]

    if category.lower() not in valid_categories:
        await ctx.send(
            f"❌ Invalid category. Choose from: **{', '.join(valid_categories)}**"
        )
        return

    if top_n > 25:
        await ctx.send("❌ Maximum 25 users. Showing top 25.")
        top_n = 25

    try:
        # Get leaderboard
        rankings = self.ai_tracker.get_leaderboard(category.lower(), limit=top_n)

        if not rankings:
            await ctx.send("📊 No users have stats yet!")
            return

        # Create embed
        embed = discord.Embed(
            title=f"🏆 Leaderboard - {category.title()}",
            color=discord.Color.gold(),
            timestamp=discord.utils.utcnow()
        )

        # Medal emojis
        medals = ["🥇", "🥈", "🥉"] + ["📍"] * 22

        # Build leaderboard text
        leaderboard_lines = []
        for i, (username, stats) in enumerate(rankings):
            # Format value based on category
            if category == "cost":
                value_str = f"${stats['lifetime_cost']:.6f}"
            elif category == "tokens":
                value_str = f"{stats['lifetime_tokens']:,} tokens"
            elif category == "credit":
                value_str = f"{stats['lifetime_credit']:.0f} points"
            elif category == "queries":
                value_str = f"{stats['query_count']:,} queries"
            elif category == "efficiency":
                eff = stats['lifetime_tokens'] / max(stats['lifetime_cost'], 0.000001)
                value_str = f"{eff:,.0f} tokens/$"

            # Highlight current user
            if username == ctx.author.display_name:
                username = f"**{username}** ⭐"

            leaderboard_lines.append(f"{medals[i]} `{i+1:2d}.` {username}\n    └ {value_str}")

        embed.description = "\n".join(leaderboard_lines)

        # Add user's rank if not in top N
        user_rank = self.ai_tracker.get_user_rank(ctx.author.display_name, category.lower())
        if user_rank > top_n:
            embed.add_field(
                name="Your Rank",
                value=f"You're ranked **#{user_rank}** in {category}",
                inline=False
            )

        # Add global stats
        global_stats = self.ai_tracker.get_global_stats()
        embed.set_footer(
            text=f"Total {global_stats['total_users']} users • "
                 f"{global_stats['total_queries']:,} queries • "
                 f"${global_stats['total_cost']:.4f} spent"
        )

        await ctx.send(embed=embed)

    except Exception as e:
        logger.error(f"Error generating leaderboard: {e}", exc_info=True)
        await ctx.send(f"❌ Error generating leaderboard: {e}")

@commands.command(name="compare", aliases=['vs'])
async def compare(self, ctx, user1: str, user2: str = None):
    """
    Compare stats between two users.

    Usage:
        !compare Alice Bob
        !vs @Alice @Bob
        !compare Alice  (compares you vs Alice)
    """
    # If user2 not specified, compare to yourself
    if user2 is None:
        user2 = ctx.author.display_name

    # Get stats
    stats1 = self.ai_tracker.get_user_stats(user1)
    stats2 = self.ai_tracker.get_user_stats(user2)

    if not stats1:
        await ctx.send(f"❌ No stats found for **{user1}**")
        return

    if not stats2:
        await ctx.send(f"❌ No stats found for **{user2}**")
        return

    # Create comparison embed
    embed = discord.Embed(
        title=f"⚔️ {user1} vs {user2}",
        color=discord.Color.purple()
    )

    # Compare each metric
    metrics = [
        ("🏆 Social Credit", "lifetime_credit", "{:.0f} points"),
        ("💰 Total Cost", "lifetime_cost", "${:.6f}"),
        ("🔥 Tokens Used", "lifetime_tokens", "{:,} tokens"),
        ("📊 Queries", "query_count", "{:,} queries")
    ]

    for emoji_name, key, fmt in metrics:
        val1 = stats1[key]
        val2 = stats2[key]

        # Determine winner
        if val1 > val2:
            winner_mark1, winner_mark2 = "✅", ""
        elif val2 > val1:
            winner_mark1, winner_mark2 = "", "✅"
        else:
            winner_mark1, winner_mark2 = "🤝", "🤝"

        embed.add_field(
            name=emoji_name,
            value=(
                f"**{user1}** {winner_mark1}\n{fmt.format(val1)}\n\n"
                f"**{user2}** {winner_mark2}\n{fmt.format(val2)}"
            ),
            inline=True
        )

    # Overall winner
    score1 = sum([
        stats1["lifetime_credit"] > stats2["lifetime_credit"],
        stats1["query_count"] > stats2["query_count"],
        stats1["lifetime_tokens"] > stats2["lifetime_tokens"]
    ])

    if score1 >= 2:
        overall_winner = user1
    else:
        overall_winner = user2

    embed.set_footer(text=f"🏆 Overall Leader: {overall_winner}")

    await ctx.send(embed=embed)
```

---

## Part 2: Bookmarking System

### Design Principles

Good bookmarking system needs:
- **Quick saving** - Fast UX (one command)
- **Organization** - Tags for categorization
- **Search** - Find bookmarks by tag/content
- **Persistence** - SQLite storage
- **Privacy** - Per-user bookmarks

### Step 8.5.3: Bookmark Database

Create `data/bookmark_storage.py`:

```python
"""
Bookmark storage using SQLite.

Learning: SQLite is perfect for structured, relational data like bookmarks.
"""

import sqlite3
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class BookmarkDatabase:
    """Manage user bookmarks in SQLite."""

    def __init__(self, db_path: str = "data/bookmarks.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    author TEXT,
                    channel_id INTEGER,
                    guild_id INTEGER,
                    created_at TEXT NOT NULL,
                    bookmarked_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS bookmark_tags (
                    bookmark_id INTEGER,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (bookmark_id) REFERENCES bookmarks (id) ON DELETE CASCADE
                )
            """)

            # Indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_bookmarks
                ON bookmarks(user_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bookmark_tags
                ON bookmark_tags(bookmark_id, tag)
            """)

            conn.commit()

        logger.info("Bookmark database initialized")

    def add_bookmark(
        self,
        user_id: int,
        message_id: int,
        content: str,
        author: str,
        channel_id: int,
        guild_id: int,
        tags: List[str],
        timestamp: datetime
    ) -> int:
        """
        Add a bookmark.

        Returns:
            Bookmark ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO bookmarks (
                    user_id, message_id, content, author,
                    channel_id, guild_id, created_at, bookmarked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                message_id,
                content,
                author,
                channel_id,
                guild_id,
                timestamp.isoformat(),
                datetime.now().isoformat()
            ))

            bookmark_id = cursor.lastrowid

            # Add tags
            if tags:
                conn.executemany("""
                    INSERT INTO bookmark_tags (bookmark_id, tag)
                    VALUES (?, ?)
                """, [(bookmark_id, tag.lower()) for tag in tags])

            conn.commit()

        logger.info(f"Bookmark {bookmark_id} created for user {user_id}")
        return bookmark_id

    def get_user_bookmarks(
        self,
        user_id: int,
        tag: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get bookmarks for a user, optionally filtered by tag."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if tag:
                # Filter by tag
                rows = conn.execute("""
                    SELECT DISTINCT b.*
                    FROM bookmarks b
                    JOIN bookmark_tags bt ON b.id = bt.bookmark_id
                    WHERE b.user_id = ? AND bt.tag = ?
                    ORDER BY b.bookmarked_at DESC
                    LIMIT ?
                """, (user_id, tag.lower(), limit)).fetchall()
            else:
                # All bookmarks
                rows = conn.execute("""
                    SELECT * FROM bookmarks
                    WHERE user_id = ?
                    ORDER BY bookmarked_at DESC
                    LIMIT ?
                """, (user_id, limit)).fetchall()

        # Convert to dicts and add tags
        bookmarks = []
        for row in rows:
            bookmark = dict(row)
            bookmark['tags'] = self._get_bookmark_tags(bookmark['id'])
            bookmarks.append(bookmark)

        return bookmarks

    def _get_bookmark_tags(self, bookmark_id: int) -> List[str]:
        """Get tags for a bookmark."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT tag FROM bookmark_tags
                WHERE bookmark_id = ?
            """, (bookmark_id,)).fetchall()

        return [row[0] for row in rows]

    def delete_bookmark(self, bookmark_id: int, user_id: int) -> bool:
        """Delete a bookmark (only if owned by user)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM bookmarks
                WHERE id = ? AND user_id = ?
            """, (bookmark_id, user_id))

            conn.commit()

        return cursor.rowcount > 0

    def search_bookmarks(
        self,
        user_id: int,
        query: str,
        limit: int = 20
    ) -> List[Dict]:
        """Search bookmarks by content."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            rows = conn.execute("""
                SELECT * FROM bookmarks
                WHERE user_id = ? AND content LIKE ?
                ORDER BY bookmarked_at DESC
                LIMIT ?
            """, (user_id, f"%{query}%", limit)).fetchall()

        bookmarks = []
        for row in rows:
            bookmark = dict(row)
            bookmark['tags'] = self._get_bookmark_tags(bookmark['id'])
            bookmarks.append(bookmark)

        return bookmarks

    def get_all_tags(self, user_id: int) -> List[tuple[str, int]]:
        """Get all tags for a user with counts."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT bt.tag, COUNT(*) as count
                FROM bookmark_tags bt
                JOIN bookmarks b ON bt.bookmark_id = b.id
                WHERE b.user_id = ?
                GROUP BY bt.tag
                ORDER BY count DESC
            """, (user_id,)).fetchall()

        return rows

    def get_stats(self, user_id: int) -> Dict:
        """Get bookmark statistics for a user."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("""
                SELECT COUNT(*) FROM bookmarks WHERE user_id = ?
            """, (user_id,)).fetchone()[0]

            tags_count = conn.execute("""
                SELECT COUNT(DISTINCT tag)
                FROM bookmark_tags bt
                JOIN bookmarks b ON bt.bookmark_id = b.id
                WHERE b.user_id = ?
            """, (user_id,)).fetchone()[0]

        return {
            "total_bookmarks": total,
            "unique_tags": tags_count
        }
```

---

### Step 8.5.4: Bookmark Commands

Create `cogs/bookmarks.py`:

```python
"""
Bookmark commands for saving and organizing important messages.
"""

import discord
from discord.ext import commands
import logging
from data.bookmark_storage import BookmarkDatabase
from typing import List

logger = logging.getLogger(__name__)


class Bookmarks(commands.Cog):
    """Bookmark and organize important messages."""

    def __init__(self, bot):
        self.bot = bot
        self.db = BookmarkDatabase()

    @commands.command(name="bookmark", aliases=['save', 'bm'])
    async def bookmark(self, ctx, message_id: int, *, tags: str = ""):
        """
        Bookmark a message for later reference.

        Usage:
            !bookmark 1234567890
            !bookmark 1234567890 important, todo, review
            !bm 1234567890 idea
        """
        try:
            # Fetch the message
            message = await ctx.channel.fetch_message(message_id)
        except discord.NotFound:
            await ctx.send("❌ Message not found in this channel")
            return
        except discord.Forbidden:
            await ctx.send("❌ I don't have permission to access that message")
            return

        # Parse tags
        tag_list = [t.strip() for t in tags.split(',') if t.strip()]

        # Save bookmark
        bookmark_id = self.db.add_bookmark(
            user_id=ctx.author.id,
            message_id=message.id,
            content=message.content,
            author=message.author.name,
            channel_id=ctx.channel.id,
            guild_id=ctx.guild.id if ctx.guild else 0,
            tags=tag_list,
            timestamp=message.created_at
        )

        # Create confirmation embed
        embed = discord.Embed(
            title="🔖 Message Bookmarked",
            description=f"{message.content[:200]}{'...' if len(message.content) > 200 else ''}",
            color=discord.Color.green()
        )

        embed.add_field(
            name="Bookmark ID",
            value=f"#{bookmark_id}",
            inline=True
        )

        embed.add_field(
            name="Tags",
            value=", ".join(tag_list) if tag_list else "*None*",
            inline=True
        )

        embed.add_field(
            name="Original Author",
            value=message.author.mention,
            inline=True
        )

        embed.set_footer(text="Use !bookmarks to view all saved messages")

        await ctx.send(embed=embed)

    @commands.command(name="bookmarks", aliases=['saved', 'bms'])
    async def list_bookmarks(self, ctx, tag: str = None):
        """
        List your bookmarked messages.

        Usage:
            !bookmarks
            !bookmarks important
            !saved todo
        """
        bookmarks = self.db.get_user_bookmarks(ctx.author.id, tag=tag)

        if not bookmarks:
            tag_str = f" with tag **{tag}**" if tag else ""
            await ctx.send(f"📭 No bookmarks found{tag_str}")
            return

        # Create paginated embeds
        pages = self._create_bookmark_pages(bookmarks, tag)
        await self._send_paginated(ctx, pages)

    def _create_bookmark_pages(self, bookmarks: List[Dict], tag: str = None) -> List[discord.Embed]:
        """Create paginated embeds for bookmarks."""
        pages = []
        items_per_page = 5

        for i in range(0, len(bookmarks), items_per_page):
            page_bookmarks = bookmarks[i:i+items_per_page]

            title = f"🔖 Your Bookmarks"
            if tag:
                title += f" (Tag: {tag})"

            embed = discord.Embed(
                title=title,
                color=discord.Color.blue()
            )

            for bm in page_bookmarks:
                # Format content
                content = bm['content'][:150]
                if len(bm['content']) > 150:
                    content += "..."

                # Format tags
                tag_str = ", ".join(bm['tags']) if bm['tags'] else "*no tags*"

                embed.add_field(
                    name=f"#{bm['id']} - {bm['author']}",
                    value=f"{content}\n*Tags: {tag_str}*",
                    inline=False
                )

            embed.set_footer(text=f"Page {len(pages)+1} • Use !bookmark_delete <id> to remove")
            pages.append(embed)

        return pages

    async def _send_paginated(self, ctx, pages: List[discord.Embed]):
        """Send paginated embeds with reactions."""
        if not pages:
            return

        if len(pages) == 1:
            await ctx.send(embed=pages[0])
            return

        # Send first page
        current_page = 0
        message = await ctx.send(embed=pages[current_page])

        # Add reactions
        await message.add_reaction("⬅️")
        await message.add_reaction("➡️")

        # Reaction check
        def check(reaction, user):
            return (user == ctx.author and
                   str(reaction.emoji) in ["⬅️", "➡️"] and
                   reaction.message.id == message.id)

        # Handle pagination
        import asyncio
        while True:
            try:
                reaction, user = await self.bot.wait_for(
                    'reaction_add',
                    timeout=60.0,
                    check=check
                )

                if str(reaction.emoji) == "➡️" and current_page < len(pages) - 1:
                    current_page += 1
                    await message.edit(embed=pages[current_page])
                elif str(reaction.emoji) == "⬅️" and current_page > 0:
                    current_page -= 1
                    await message.edit(embed=pages[current_page])

                await message.remove_reaction(reaction, user)

            except asyncio.TimeoutError:
                await message.clear_reactions()
                break

    @commands.command(name="bookmark_delete", aliases=['bm_del', 'unsave'])
    async def delete_bookmark(self, ctx, bookmark_id: int):
        """
        Delete a bookmark.

        Usage: !bookmark_delete 42
        """
        success = self.db.delete_bookmark(bookmark_id, ctx.author.id)

        if success:
            await ctx.send(f"✅ Bookmark #{bookmark_id} deleted")
        else:
            await ctx.send(f"❌ Bookmark #{bookmark_id} not found or you don't own it")

    @commands.command(name="bookmark_search", aliases=['bm_search'])
    async def search_bookmarks(self, ctx, *, query: str):
        """
        Search your bookmarks by content.

        Usage: !bookmark_search docker deployment
        """
        results = self.db.search_bookmarks(ctx.author.id, query)

        if not results:
            await ctx.send(f"🔍 No bookmarks found matching **{query}**")
            return

        # Show results
        embed = discord.Embed(
            title=f"🔍 Search Results: {query}",
            description=f"Found {len(results)} bookmark(s)",
            color=discord.Color.blue()
        )

        for bm in results[:10]:  # Show top 10
            content = bm['content'][:100]
            if len(bm['content']) > 100:
                content += "..."

            embed.add_field(
                name=f"#{bm['id']} - {bm['author']}",
                value=content,
                inline=False
            )

        if len(results) > 10:
            embed.set_footer(text=f"Showing 10 of {len(results)} results")

        await ctx.send(embed=embed)

    @commands.command(name="bookmark_tags", aliases=['bm_tags'])
    async def list_tags(self, ctx):
        """
        List all your bookmark tags with counts.

        Usage: !bookmark_tags
        """
        tags = self.db.get_all_tags(ctx.author.id)

        if not tags:
            await ctx.send("📋 No tags found. Add tags when bookmarking:\n`!bookmark <id> tag1, tag2`")
            return

        embed = discord.Embed(
            title="📋 Your Bookmark Tags",
            color=discord.Color.blue()
        )

        tag_list = [f"**{tag}** ({count})" for tag, count in tags]
        embed.description = ", ".join(tag_list)

        stats = self.db.get_stats(ctx.author.id)
        embed.set_footer(
            text=f"{stats['total_bookmarks']} total bookmarks • "
                 f"{stats['unique_tags']} unique tags"
        )

        await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(Bookmarks(bot))
```

---

## Testing

### Test Leaderboards

```bash
# Generate some activity
!hello
!summary 10

# Check leaderboards
!leaderboard credit
!leaderboard tokens
!lb efficiency 5

# Compare users
!compare Alice Bob
!vs Alice
```

### Test Bookmarks

```bash
# Bookmark messages
!bookmark 1234567890 important, todo
!bm 9876543210 idea

# View bookmarks
!bookmarks
!bookmarks important
!saved todo

# Search
!bookmark_search docker
!bm_search deployment

# View tags
!bookmark_tags

# Delete
!bookmark_delete 1
```

---

## Key Takeaways

✅ **Leaderboards** → gamification → 3x more engagement
✅ **Multiple categories** → different user motivations
✅ **Bookmarks** → personal knowledge management
✅ **Tags** → organization without folders
✅ **Search** → find what you need quickly

**What's Next?**
- Phase 10.5: Smart Context Building (the big one!)

---

[← Back to Implementation Index](./IMPLEMENTATION_INDEX.md)
