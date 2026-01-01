"""
Resolution tracking cog for New Year's resolutions.

Provides commands for creating, managing, and tracking resolutions with checkpoints.
"""

import discord
from discord.ext import commands
import logging
from typing import Optional

from config import Config
from storage.resolutions import ResolutionStorage
from storage.resolutions.resolutions import CheckFrequency
from ai.service import AIService
from ai.tracker import UserAITracker
from bot.cogs.resolution_views import (
    CheckInView,
    MultiResolutionCheckInView,
    CheckpointView,
    AddCheckpointView,
    ConfirmDeleteView,
    build_resolution_list_embed,
    build_completion_embed,
    build_check_in_embed
)

logger = logging.getLogger(__name__)


class Resolutions(commands.Cog):
    """Resolution tracking commands."""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config
        self.resolution_storage = ResolutionStorage()
        self.ai_service = AIService(provider_name=self.config.AI_DEFAULT_PROVIDER)
        self.ai_tracker = UserAITracker()
        
        logger.info("Resolutions cog initialized")
    
    def _validate_frequency(self, frequency: str) -> Optional[str]:
        """Validate and normalize frequency string."""
        freq_lower = frequency.lower().strip()
        valid_frequencies = {
            'weekly': 'weekly',
            'biweekly': 'biweekly',
            'bi-weekly': 'biweekly',
            'monthly': 'monthly'
        }
        return valid_frequencies.get(freq_lower)
    
    @commands.group(name="resolution", aliases=["res"], invoke_without_command=True)
    async def resolution(self, ctx):
        """Resolution commands. Use !resolution <subcommand>."""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(
                title="📋 Resolution Tracking Commands",
                description="Track your New Year's resolutions with check-ins and checkpoints!",
                color=discord.Color.blue()
            )
            
            # Main commands
            embed.add_field(
                name="🎯 Main Commands",
                value=(
                    "`!resolution set <freq> <text>` - Create a resolution\n"
                    "`!resolution list` - View your resolutions with progress\n"
                    "`!resolution list all` - Include completed resolutions\n"
                    "`!resolution check <id> <status> [notes]` - Record a check-in\n"
                    "`!resolution summary` - Get AI-powered progress analysis\n"
                    "`!resolution edit <id> <text>` - Edit resolution text\n"
                    "`!resolution delete <id>` - Delete a resolution\n"
                    "`!resolution delete all` - Delete ALL your resolutions"
                ),
                inline=False
            )
            
            # Frequencies
            embed.add_field(
                name="📅 Check-in Frequencies",
                value=(
                    "**weekly** - Same day every week (e.g., every Monday)\n"
                    "**biweekly** - Same day every 2 weeks (e.g., every other Monday)\n"
                    "**monthly** - Same day each month (e.g., the 15th)\n\n"
                    "💡 The check-in day is set when you create the resolution!"
                ),
                inline=False
            )
            
            # Check-in statuses
            embed.add_field(
                name="✅ Check-in Statuses",
                value=(
                    "**on_track** - Things are going well\n"
                    "**struggling** - Having challenges (still counts for streak!)\n\n"
                    "🔥 Both statuses keep your streak alive!"
                ),
                inline=False
            )
            
            # Features
            embed.add_field(
                name="🌟 Features",
                value=(
                    "• **Checkpoints** - Break goals into sub-tasks\n"
                    "• **Streaks** - Track consecutive check-ins\n"
                    "• **Milestones** - Celebrate 25%, 50%, 75%, 100% progress\n"
                    "• **Auto-reminders** - DM reminders if you miss a check-in\n"
                    "• **AI Summary** - Personalized progress insights"
                ),
                inline=False
            )
            
            # Examples
            embed.add_field(
                name="💡 Examples",
                value=(
                    "`!resolution set weekly \"Exercise 3x per week\"`\n"
                    "`!resolution set monthly \"Read 12 books this year\"`\n"
                    "`!resolution check 1 on_track \"Did 3 workouts!\"`\n"
                    "`!resolution summary`"
                ),
                inline=False
            )
            
            embed.set_footer(text="Use !checkpoint to manage sub-tasks • Check-ins happen automatically on your schedule!")
            
            await ctx.send(embed=embed)
    
    @resolution.command(name="set", help="Create a new resolution")
    async def resolution_set(self, ctx, frequency: str, *, text: str):
        """
        Create a new resolution.
        
        Usage: !resolution set <frequency> <text>
        Frequencies: weekly, biweekly, monthly
        
        Example: !resolution set weekly "Exercise 3 times per week"
        """
        # Validate frequency
        normalized_freq = self._validate_frequency(frequency)
        if not normalized_freq:
            await ctx.send(
                "❌ Invalid frequency. Use: `weekly`, `biweekly`, or `monthly`\n"
                "Example: `!resolution set weekly \"Exercise 3 times per week\"`"
            )
            return
        
        # Clean up text (remove quotes if wrapped)
        text = text.strip('"\'')
        
        if len(text) < 3:
            await ctx.send("❌ Resolution text is too short. Please be more specific.")
            return
        
        if len(text) > 500:
            await ctx.send("❌ Resolution text is too long (max 500 characters).")
            return
        
        user_id = str(ctx.author.id)
        user_display_name = ctx.author.display_name
        
        try:
            resolution_id = self.resolution_storage.create_resolution(
                user_id=user_id,
                text=text,
                frequency=normalized_freq,
                user_display_name=user_display_name
            )
            
            # Get the created resolution to show check-in day and per-user ID
            resolution = self.resolution_storage.get_resolution(resolution_id, user_id=user_id)
            check_day_display = resolution.get('check_day_display', 'N/A')
            next_check = resolution.get('next_check_date', '')[:10]  # Get just date part
            
            # Get user display ID (should be the last one in their list)
            user_resolutions = self.resolution_storage.get_user_resolutions(user_id)
            display_id = len(user_resolutions)  # Since it's the newest, it's the last one
            
            embed = discord.Embed(
                title="🎯 Resolution Created!",
                description=f"**{text}**",
                color=discord.Color.green()
            )
            
            # Show frequency with check-in day
            if normalized_freq in ['weekly', 'biweekly']:
                freq_desc = f"{normalized_freq.title()} (every {check_day_display})"
            else:
                freq_desc = f"{normalized_freq.title()} ({check_day_display})"
            
            embed.add_field(
                name="Check-in Schedule",
                value=freq_desc,
                inline=True
            )
            embed.add_field(
                name="First Check-in",
                value=next_check,
                inline=True
            )
            embed.add_field(
                name="Resolution ID",
                value=f"#{display_id}",
                inline=True
            )
            embed.add_field(
                name="Next Step",
                value=f"Add checkpoints with:\n`!checkpoint add {display_id} \"Your sub-task\"`",
                inline=False
            )
            embed.set_footer(text="Break your goal into smaller, achievable steps!")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error creating resolution: {e}", exc_info=True)
            await ctx.send("❌ Failed to create resolution. Please try again.")
    
    @resolution.command(name="list", help="View your resolutions")
    async def resolution_list(self, ctx, show_completed: Optional[str] = None):
        """
        View your resolutions with progress.
        
        Usage: !resolution list [all]
        Add 'all' to include completed resolutions.
        """
        user_id = str(ctx.author.id)
        include_completed = show_completed and show_completed.lower() == 'all'
        
        try:
            resolutions = self.resolution_storage.get_user_resolutions(
                user_id=user_id,
                include_completed=include_completed
            )
            
            embed = build_resolution_list_embed(
                resolutions=resolutions,
                user_display_name=ctx.author.display_name
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error listing resolutions: {e}", exc_info=True)
            await ctx.send("❌ Failed to load resolutions. Please try again.")
    
    def _resolve_resolution_id(self, user_id: str, target_id: int) -> Optional[int]:
        """
        Resolve a resolution ID - tries per-user display ID first, then global ID.
        
        Args:
            user_id: User's Discord ID
            target_id: ID from command (could be per-user or global)
        
        Returns:
            Global resolution ID or None if not found
        """
        # First try per-user display ID
        resolution = self.resolution_storage.get_resolution_by_user_display_id(user_id, target_id)
        if resolution:
            return resolution['id']
        
        # Fall back to global ID
        resolution = self.resolution_storage.get_resolution(target_id, user_id=user_id)
        if resolution and resolution['user_id'] == user_id:
            return resolution['id']
        
        return None
    
    @resolution.command(name="check", help="Record a check-in for a resolution")
    async def resolution_check(self, ctx, resolution_id: int, status: str, *, notes: Optional[str] = None):
        """
        Record a check-in response.
        
        Usage: !resolution check <id> <status> [notes]
        Statuses: on_track, struggling
        
        The ID can be your per-user ID (1, 2, 3...) or the global ID.
        
        Example: !resolution check 1 on_track "Made good progress this week!"
        """
        status_lower = status.lower().strip()
        valid_statuses = ['on_track', 'ontrack', 'struggling']
        
        if status_lower not in valid_statuses:
            await ctx.send(
                "❌ Invalid status. Use: `on_track` or `struggling`\n"
                "Example: `!resolution check 1 on_track \"Made progress!\"`"
            )
            return
        
        # Normalize status
        if status_lower == 'ontrack':
            status_lower = 'on_track'
        
        user_id = str(ctx.author.id)
        
        try:
            # Resolve ID (per-user or global)
            global_id = self._resolve_resolution_id(user_id, resolution_id)
            if not global_id:
                await ctx.send("❌ Resolution not found.")
                return
            
            # Get resolution to verify ownership
            resolution = self.resolution_storage.get_resolution(global_id, user_id=user_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            # Record check-in
            result = self.resolution_storage.record_check_in(
                resolution_id=global_id,
                status=status_lower,
                notes=notes.strip('"\'') if notes else None
            )
            
            if 'error' in result:
                await ctx.send(f"❌ Error: {result['error']}")
                return
            
            # Build response
            status_emoji = "✅" if status_lower == "on_track" else "💪"
            status_text = "On Track" if status_lower == "on_track" else "Struggling (but showing up!)"
            
            embed = discord.Embed(
                title=f"{status_emoji} Check-in Recorded",
                description=f"**{resolution['text']}**",
                color=discord.Color.green() if status_lower == "on_track" else discord.Color.blue()
            )
            
            # Streak info
            if result['new_streak'] > 0:
                embed.add_field(
                    name="🔥 Streak",
                    value=f"{result['new_streak']} check-ins",
                    inline=True
                )
            
            # Streak milestone
            if result.get('streak_milestone'):
                embed.add_field(
                    name="🎉 Milestone!",
                    value=f"{result['streak_milestone']}-week streak achieved!",
                    inline=True
                )
            
            # Next check-in
            next_date = result['next_check_date'][:10]
            embed.add_field(
                name="📅 Next Check-in",
                value=next_date,
                inline=True
            )
            
            if notes:
                embed.add_field(
                    name="Notes",
                    value=notes[:200] + "..." if len(notes) > 200 else notes,
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error recording check-in: {e}", exc_info=True)
            await ctx.send("❌ Failed to record check-in. Please try again.")
    
    @resolution.command(name="edit", help="Edit a resolution's text")
    async def resolution_edit(self, ctx, resolution_id: int, *, new_text: str):
        """
        Edit a resolution's text.
        
        Usage: !resolution edit <id> <new_text>
        The ID can be your per-user ID (1, 2, 3...) or the global ID.
        """
        user_id = str(ctx.author.id)
        new_text = new_text.strip('"\'')
        
        try:
            # Resolve ID (per-user or global)
            global_id = self._resolve_resolution_id(user_id, resolution_id)
            if not global_id:
                await ctx.send("❌ Resolution not found.")
                return
            
            resolution = self.resolution_storage.get_resolution(global_id, user_id=user_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            success = self.resolution_storage.update_resolution(
                resolution_id=global_id,
                text=new_text
            )
            
            if success:
                await ctx.send(f"✅ Resolution updated: **{new_text}**")
            else:
                await ctx.send("❌ Failed to update resolution.")
            
        except Exception as e:
            logger.error(f"Error editing resolution: {e}", exc_info=True)
            await ctx.send("❌ Failed to edit resolution. Please try again.")
    
    @resolution.command(name="delete", help="Delete a resolution or all resolutions")
    async def resolution_delete(self, ctx, target: str):
        """
        Delete a resolution or all resolutions.
        
        Usage: !resolution delete <id> OR !resolution delete all
        """
        user_id = str(ctx.author.id)
        
        try:
            # Handle "all" case
            if target.lower() == "all":
                resolutions = self.resolution_storage.get_user_resolutions(user_id, include_completed=True)
                
                if not resolutions:
                    await ctx.send("❌ You don't have any resolutions to delete.")
                    return
                
                # Create confirmation view for deleting all
                from bot.cogs.resolution_views import ConfirmDeleteAllView
                view = ConfirmDeleteAllView(
                    user_id=user_id,
                    resolution_storage=self.resolution_storage
                )
                
                total_checkpoints = sum(r['checkpoint_progress']['total'] for r in resolutions)
                await ctx.send(
                    f"⚠️ **DELETE ALL RESOLUTIONS?**\n\n"
                    f"This will permanently delete:\n"
                    f"• {len(resolutions)} resolution(s)\n"
                    f"• {total_checkpoints} checkpoint(s)\n"
                    f"• All check-in history\n\n"
                    f"**This cannot be undone!**",
                    view=view
                )
                return
            
            # Handle single resolution deletion
            try:
                target_id = int(target)
            except ValueError:
                await ctx.send("❌ Invalid format. Use `!resolution delete <id>` or `!resolution delete all`")
                return
            
            # Resolve ID (per-user or global)
            global_id = self._resolve_resolution_id(user_id, target_id)
            if not global_id:
                await ctx.send("❌ Resolution not found.")
                return
            
            resolution = self.resolution_storage.get_resolution(global_id, user_id=user_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            # Send confirmation with buttons
            view = ConfirmDeleteView(
                resolution_id=global_id,
                resolution_storage=self.resolution_storage
            )
            
            await ctx.send(
                f"⚠️ Are you sure you want to delete **\"{resolution['text']}\"**?\n"
                f"This will also delete all {resolution['checkpoint_progress']['total']} checkpoints.",
                view=view
            )
            
        except Exception as e:
            logger.error(f"Error deleting resolution: {e}", exc_info=True)
            await ctx.send("❌ Failed to delete resolution. Please try again.")
    
    @resolution.command(name="summary", help="Get AI analysis of your resolution progress")
    async def resolution_summary(self, ctx):
        """
        Get an AI-generated summary of your resolution progress.
        
        Usage: !resolution summary
        """
        user_id = str(ctx.author.id)
        user_display_name = ctx.author.display_name
        
        try:
            resolutions = self.resolution_storage.get_user_resolutions(
                user_id=user_id,
                include_completed=True
            )
            
            if not resolutions:
                await ctx.send(
                    "❌ You don't have any resolutions yet.\n"
                    "Create one with `!resolution set <frequency> <text>`"
                )
                return
            
            # Build context for AI
            context_parts = []
            for res in resolutions:
                progress = res['checkpoint_progress']
                check_ins = self.resolution_storage.get_check_in_history(res['id'], limit=10)
                
                res_context = f"Resolution: {res['text']}\n"
                res_context += f"- Status: {'Completed' if res['is_completed'] else 'Active'}\n"
                res_context += f"- Frequency: {res['frequency']}\n"
                res_context += f"- Checkpoints: {progress['completed']}/{progress['total']} ({progress['percentage']}%)\n"
                res_context += f"- Current streak: {res['current_streak']}, Best: {res['longest_streak']}\n"
                
                if check_ins:
                    on_track = sum(1 for c in check_ins if c['status'] == 'on_track')
                    struggling = sum(1 for c in check_ins if c['status'] == 'struggling')
                    skipped = sum(1 for c in check_ins if c['status'] == 'skipped')
                    res_context += f"- Recent check-ins: {on_track} on-track, {struggling} struggling, {skipped} skipped\n"
                
                context_parts.append(res_context)
            
            stats = self.resolution_storage.get_user_check_in_stats(user_id)
            
            prompt = f"""Analyze this user's resolution progress and provide encouraging, personalized feedback.

User: {user_display_name}

Overall Stats:
- Total check-ins: {stats['total']}
- On track: {stats['on_track']}
- Struggling: {stats['struggling']}
- Skipped: {stats['skipped']}

Resolutions:
{chr(10).join(context_parts)}

Provide a brief, encouraging summary (3-5 sentences) that:
1. Highlights their progress and wins
2. Identifies any patterns (struggling areas, strong streaks)
3. Offers one specific, actionable suggestion
4. Ends with motivation

Keep the tone friendly and supportive, like a helpful accountability partner."""

            status_msg = await ctx.send("🔍 Analyzing your progress...")
            
            result = await self.ai_service.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=0.7,
                user_id=user_id,
                user_display_name=user_display_name
            )
            
            # Track AI usage
            self.ai_tracker.log_ai_usage(
                user_display_name=user_display_name,
                cost=result['cost'],
                tokens_total=result['tokens_total']
            )
            
            # Build summary embed
            embed = discord.Embed(
                title=f"📊 {user_display_name}'s Resolution Progress",
                description=result['content'],
                color=discord.Color.gold()
            )
            
            # Add quick stats
            active_count = sum(1 for r in resolutions if not r['is_completed'])
            completed_count = sum(1 for r in resolutions if r['is_completed'])
            
            embed.add_field(
                name="📋 Resolutions",
                value=f"{active_count} active, {completed_count} completed",
                inline=True
            )
            
            if stats['total'] > 0:
                success_rate = round((stats['on_track'] + stats['struggling']) / stats['total'] * 100)
                embed.add_field(
                    name="✅ Check-in Rate",
                    value=f"{success_rate}% ({stats['total']} total)",
                    inline=True
                )
            
            embed.set_footer(text=f"Cost: ${result['cost']:.6f} | Model: {result['model']}")
            
            await status_msg.edit(content=None, embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            await ctx.send("❌ Failed to generate summary. Please try again.")
    
    # ==================== Checkpoint Commands ====================
    
    @commands.group(name="checkpoint", aliases=["cp"], invoke_without_command=True)
    async def checkpoint(self, ctx):
        """
        Checkpoint commands. Shows dropdown to complete checkpoints if no subcommand.
        
        Usage: !checkpoint (shows interactive dropdown)
        Or: !checkpoint add <resolution_id> <text>
        """
        if ctx.invoked_subcommand is None:
            user_id = str(ctx.author.id)
            
            try:
                # Get incomplete checkpoints
                checkpoints = self.resolution_storage.get_user_incomplete_checkpoints(user_id)
                
                if not checkpoints:
                    # Check if user has any resolutions
                    resolutions = self.resolution_storage.get_user_resolutions(user_id)
                    if not resolutions:
                        # Show help embed when no resolutions exist
                        embed = discord.Embed(
                            title="📝 Checkpoint Commands",
                            description="Checkpoints are sub-tasks that help you break down your resolutions into manageable steps!",
                            color=discord.Color.blue()
                        )
                        
                        embed.add_field(
                            name="🎯 Getting Started",
                            value=(
                                "1. Create a resolution first:\n"
                                "   `!resolution set weekly \"Your goal\"`\n\n"
                                "2. Get the Resolution ID from:\n"
                                "   `!resolution list`\n\n"
                                "3. Add checkpoints:\n"
                                "   `!checkpoint add <res_id> \"Sub-task\"`"
                            ),
                            inline=False
                        )
                        
                        embed.add_field(
                            name="📋 Checkpoint Commands",
                            value=(
                                "`!checkpoint` - Interactive dropdown to complete checkpoints\n"
                                "`!checkpoint add <res_id> <text>` - Add a sub-task\n"
                                "`!checkpoint complete <cp_id>` - Mark checkpoint done\n"
                                "`!checkpoint list <res_id>` - View all checkpoints\n"
                                "`!checkpoint delete <cp_id>` - Remove a checkpoint"
                            ),
                            inline=False
                        )
                        
                        embed.add_field(
                            name="💡 Examples",
                            value=(
                                "`!checkpoint add 1 \"Join a gym\"`\n"
                                "`!checkpoint add 1 \"Exercise 3x per week\"`\n"
                                "`!checkpoint add 1 \"Task 1\" | \"Task 2\" | \"Task 3\"` - Add multiple\n"
                                "`!checkpoint add_weekly 1 \"Daily leetcode\"` - Add 52 weekly checkpoints\n"
                                "`!checkpoint list 1` - See checkpoint IDs\n"
                                "`!checkpoint complete 3` - Complete checkpoint #3"
                            ),
                            inline=False
                        )
                        
                        embed.add_field(
                            name="🏆 Milestones",
                            value=(
                                "Celebrate when you hit:\n"
                                "• 25% complete - Great start!\n"
                                "• 50% complete - Halfway there!\n"
                                "• 75% complete - Almost there!\n"
                                "• 100% complete - Resolution complete! 🎉"
                            ),
                            inline=False
                        )
                        
                        embed.set_footer(text="Use !resolution set to create your first resolution!")
                        
                        await ctx.send(embed=embed)
                    else:
                        # Show help embed when all checkpoints are complete
                        embed = discord.Embed(
                            title="🎉 All Checkpoints Complete!",
                            description="Great job! You've finished all your checkpoints.",
                            color=discord.Color.gold()
                        )
                        
                        embed.add_field(
                            name="✨ What's Next?",
                            value=(
                                "Add more checkpoints:\n"
                                "`!checkpoint add <res_id> \"New sub-task\"`\n\n"
                                "Or view your progress:\n"
                                "`!resolution list` - See all resolutions\n"
                                "`!resolution summary` - Get AI analysis"
                            ),
                            inline=False
                        )
                        
                        await ctx.send(embed=embed)
                    return
                
                # Show help embed with dropdown
                embed = discord.Embed(
                    title="📝 Complete a Checkpoint",
                    description="Use the dropdown below to mark a checkpoint as complete, or use commands:",
                    color=discord.Color.blue()
                )
                
                embed.add_field(
                    name="🔧 Quick Commands",
                    value=(
                        "`!checkpoint add <res_id> <text>` - Add new checkpoint\n"
                        "`!checkpoint add_weekly <res_id> <task>` - Add 52 weekly checkpoints\n"
                        "`!checkpoint list <res_id>` - View checkpoint IDs\n"
                        "`!checkpoint complete <cp_id>` - Complete by ID"
                    ),
                    inline=False
                )
                
                embed.set_footer(text="Select from dropdown below, or use commands above!")
                
                view = CheckpointView(
                    checkpoints=checkpoints,
                    resolution_storage=self.resolution_storage
                )
                
                await ctx.send(embed=embed, view=view)
                
            except Exception as e:
                logger.error(f"Error showing checkpoint dropdown: {e}", exc_info=True)
                await ctx.send("❌ Failed to load checkpoints. Please try again.")
    
    @checkpoint.command(name="add", help="Add one or more checkpoints to a resolution")
    async def checkpoint_add(self, ctx, resolution_id: int, *, text: str):
        """
        Add checkpoint(s) to a resolution.
        
        Usage: 
        Single: !checkpoint add <resolution_id> <text>
        Multiple: !checkpoint add <resolution_id> <text1> | <text2> | <text3>
        
        The resolution_id can be your per-user ID (1, 2, 3...) or the global ID.
        
        Examples:
        !checkpoint add 1 "Join a gym"
        !checkpoint add 1 "Daily leetcode" | "Daily system design" | "March - application begins"
        """
        user_id = str(ctx.author.id)
        text = text.strip('"\'')
        
        # Check if multiple checkpoints (separated by |)
        checkpoint_texts = [t.strip().strip('"\'') for t in text.split('|')]
        
        # Validate all checkpoints
        for cp_text in checkpoint_texts:
            if len(cp_text) < 2:
                await ctx.send(f"❌ Checkpoint text is too short: \"{cp_text}\"")
                return
            
            if len(cp_text) > 200:
                await ctx.send(f"❌ Checkpoint text is too long (max 200 characters): \"{cp_text[:50]}...\"")
                return
        
        try:
            # Resolve ID (per-user or global)
            global_id = self._resolve_resolution_id(user_id, resolution_id)
            if not global_id:
                await ctx.send("❌ Resolution not found.")
                return
            
            # Verify resolution belongs to user
            resolution = self.resolution_storage.get_resolution(global_id, user_id=user_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            # Add all checkpoints
            added_checkpoints = []
            for cp_text in checkpoint_texts:
                checkpoint_id = self.resolution_storage.add_checkpoint(
                    resolution_id=global_id,
                    text=cp_text
                )
                added_checkpoints.append({
                    'id': checkpoint_id,
                    'text': cp_text
                })
            
            # Get updated progress
            resolution = self.resolution_storage.get_resolution(global_id, user_id=user_id)
            progress = resolution['checkpoint_progress']
            
            # Build response embed
            if len(added_checkpoints) == 1:
                embed = discord.Embed(
                    title="✅ Checkpoint Added",
                    description=f"**{added_checkpoints[0]['text']}**",
                    color=discord.Color.green()
                )
            else:
                embed = discord.Embed(
                    title=f"✅ {len(added_checkpoints)} Checkpoints Added",
                    description=f"Added to **{resolution['text']}**",
                    color=discord.Color.green()
                )
                
                checkpoint_list = "\n".join([f"• {cp['text']}" for cp in added_checkpoints])
                if len(checkpoint_list) > 1024:
                    checkpoint_list = "\n".join([f"• {cp['text'][:80]}..." if len(cp['text']) > 80 else f"• {cp['text']}" for cp in added_checkpoints])
                
                embed.add_field(
                    name="Added Checkpoints",
                    value=checkpoint_list,
                    inline=False
                )
            
            embed.add_field(
                name="Resolution",
                value=resolution['text'][:100],
                inline=False
            )
            embed.add_field(
                name="Total Checkpoints",
                value=str(progress['total']),
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error adding checkpoint: {e}", exc_info=True)
            await ctx.send("❌ Failed to add checkpoint. Please try again.")
    
    @checkpoint.command(name="add_weekly", aliases=["addweekly"], help="Add 52 weekly checkpoints for daily tasks")
    async def checkpoint_add_weekly(self, ctx, resolution_id: int, *, tasks: str):
        """
        Add 52 weekly checkpoints for one or more daily tasks.
        
        Usage: 
        Single: !checkpoint add_weekly <resolution_id> <task_name>
        Multiple: !checkpoint add_weekly <resolution_id> <task1> | <task2> | <task3>
        
        The resolution_id can be your per-user ID (1, 2, 3...) or the global ID.
        This creates 52 checkpoints per task (Week 1-52) for tracking
        daily habits on a weekly basis.
        
        Examples: 
        !checkpoint add_weekly 1 "Daily leetcode"
        !checkpoint add_weekly 2 "Run 3x per week" | "Daily pullup" | "Daily pushup"
        """
        user_id = str(ctx.author.id)
        tasks = tasks.strip().strip('"\'')
        
        # Check if multiple tasks (separated by |)
        task_names = [t.strip().strip('"\'') for t in tasks.split('|')]
        
        # Validate all tasks
        for task_name in task_names:
            if len(task_name) < 2:
                await ctx.send(f"❌ Task name is too short: \"{task_name}\"")
                return
            
            if len(task_name) > 150:
                await ctx.send(f"❌ Task name is too long (max 150 characters): \"{task_name[:50]}...\"")
                return
        
        try:
            # Resolve ID (per-user or global)
            global_id = self._resolve_resolution_id(user_id, resolution_id)
            if not global_id:
                await ctx.send("❌ Resolution not found.")
                return
            
            # Verify resolution belongs to user
            resolution = self.resolution_storage.get_resolution(global_id, user_id=user_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            # Send status message (warn if checkpoints already exist)
            existing_checkpoints = resolution['checkpoints']
            if len(existing_checkpoints) > 0:
                status_msg = await ctx.send(
                    f"⚠️ Note: This resolution already has {len(existing_checkpoints)} checkpoint(s).\n"
                    f"⏳ Creating {len(task_names) * 52} weekly checkpoints ({len(task_names)} task(s) × 52 weeks)..."
                )
            else:
                status_msg = await ctx.send(
                    f"⏳ Creating {len(task_names) * 52} weekly checkpoints ({len(task_names)} task(s) × 52 weeks)..."
                )
            
            # Create 52 weekly checkpoints for each task
            total_added = 0
            task_results = []
            
            for task_name in task_names:
                task_added = 0
                for week_num in range(1, 53):
                    checkpoint_text = f"Week {week_num}: {task_name}"
                    try:
                        self.resolution_storage.add_checkpoint(
                            resolution_id=global_id,
                            text=checkpoint_text
                        )
                        task_added += 1
                        total_added += 1
                    except Exception as e:
                        logger.error(f"Error adding checkpoint Week {week_num} for '{task_name}': {e}")
                        # Continue with other weeks
                
                task_results.append({
                    'name': task_name,
                    'count': task_added
                })
            
            # Get updated progress
            resolution = self.resolution_storage.get_resolution(global_id, user_id=user_id)
            progress = resolution['checkpoint_progress']
            
            embed = discord.Embed(
                title=f"✅ {len(task_names)} Weekly Task Set(s) Created!",
                description=f"Added to **{resolution['text']}**",
                color=discord.Color.green()
            )
            
            # Show task summary
            if len(task_results) == 1:
                embed.add_field(
                    name="Task",
                    value=task_results[0]['name'],
                    inline=False
                )
            else:
                task_list = "\n".join([f"• {tr['name']} ({tr['count']} checkpoints)" for tr in task_results])
                if len(task_list) > 1024:
                    task_list = "\n".join([f"• {tr['name'][:60]}..." if len(tr['name']) > 60 else f"• {tr['name']}" for tr in task_results[:10]])
                    if len(task_results) > 10:
                        task_list += f"\n... and {len(task_results) - 10} more"
                
                embed.add_field(
                    name="Tasks Created",
                    value=task_list,
                    inline=False
                )
            
            embed.add_field(
                name="Total Checkpoints Created",
                value=f"{total_added} weekly checkpoints (52 per task × {len(task_names)} tasks)",
                inline=True
            )
            
            embed.add_field(
                name="Total Checkpoints",
                value=str(progress['total']),
                inline=True
            )
            
            # Get user display ID for the resolution
            user_resolutions = self.resolution_storage.get_user_resolutions(user_id, include_completed=True)
            display_id = None
            for idx, res in enumerate(user_resolutions, 1):
                if res['id'] == global_id:
                    display_id = idx
                    break
            
            embed.add_field(
                name="💡 How to Use",
                value=(
                    f"Complete each week's checkpoint when you've done the daily task that week.\n"
                    f"Use `!checkpoint list {display_id or global_id}` to see all weekly checkpoints.\n"
                    f"Or use `!checkpoint` for the interactive dropdown."
                ),
                inline=False
            )
            
            await status_msg.edit(content=None, embed=embed)
            
        except Exception as e:
            logger.error(f"Error adding weekly checkpoints: {e}", exc_info=True)
            await ctx.send("❌ Failed to add weekly checkpoints. Please try again.")
    
    @checkpoint.command(name="complete", help="Mark a checkpoint as complete")
    async def checkpoint_complete(self, ctx, checkpoint_id: int):
        """
        Mark a checkpoint as completed.
        
        Usage: !checkpoint complete <checkpoint_id>
        """
        user_id = str(ctx.author.id)
        
        try:
            result = self.resolution_storage.complete_checkpoint(checkpoint_id)
            
            if 'error' in result:
                await ctx.send(f"❌ Error: {result['error']}")
                return
            
            # Verify ownership
            resolution = self.resolution_storage.get_resolution(result['resolution_id'])
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This checkpoint doesn't belong to you.")
                return
            
            progress = result['new_progress']
            
            embed = discord.Embed(
                title="✅ Checkpoint Completed!",
                color=discord.Color.green()
            )
            
            # Progress bar
            progress_bar = "█" * int(progress['percentage'] / 10) + "░" * (10 - int(progress['percentage'] / 10))
            embed.add_field(
                name="📊 Progress",
                value=f"{progress_bar} {progress['completed']}/{progress['total']} ({progress['percentage']}%)",
                inline=False
            )
            
            # Milestone celebration
            if result.get('milestone'):
                milestone_messages = {
                    25: "🌟 **Great start!** You've completed 25% of your checkpoints!",
                    50: "🌟 **Halfway there!** 50% of checkpoints complete!",
                    75: "🌟 **Almost there!** 75% done - the finish line is in sight!",
                    100: "🎉 **RESOLUTION COMPLETE!** You finished all checkpoints!"
                }
                embed.add_field(
                    name="🏆 Milestone",
                    value=milestone_messages.get(result['milestone'], ""),
                    inline=False
                )
            
            # All completed celebration
            if result.get('all_completed'):
                embed.color = discord.Color.gold()
                embed.set_footer(text="Congratulations on achieving your goal! 🎊")
                
                # Mark resolution as completed
                self.resolution_storage.mark_resolution_completed(result['resolution_id'])
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error completing checkpoint: {e}", exc_info=True)
            await ctx.send("❌ Failed to complete checkpoint. Please try again.")
    
    @checkpoint.command(name="list", help="List checkpoints for a resolution")
    async def checkpoint_list(self, ctx, resolution_id: int):
        """
        List all checkpoints for a resolution.
        
        Usage: !checkpoint list <resolution_id>
        """
        user_id = str(ctx.author.id)
        
        try:
            resolution = self.resolution_storage.get_resolution(resolution_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            checkpoints = resolution['checkpoints']
            progress = resolution['checkpoint_progress']
            
            embed = discord.Embed(
                title=f"📋 Checkpoints for Resolution #{display_id or global_id}",
                description=f"**{resolution['text']}**",
                color=discord.Color.blue()
            )
            
            if checkpoints:
                # Progress bar
                progress_bar = "█" * int(progress['percentage'] / 10) + "░" * (10 - int(progress['percentage'] / 10))
                embed.add_field(
                    name="Progress",
                    value=f"{progress_bar} {progress['completed']}/{progress['total']} ({progress['percentage']}%)",
                    inline=False
                )
                
                # Checkpoints list
                checkpoint_text = ""
                for cp in checkpoints:
                    emoji = "✅" if cp['is_completed'] else "⬜"
                    checkpoint_text += f"{emoji} #{cp['id']} {cp['text']}\n"
                
                embed.add_field(
                    name="Checkpoints",
                    value=checkpoint_text[:1024],
                    inline=False
                )
            else:
                embed.add_field(
                    name="No Checkpoints",
                    value=f"Add one with `!checkpoint add {resolution_id} \"Your task\"`",
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}", exc_info=True)
            await ctx.send("❌ Failed to list checkpoints. Please try again.")
    
    @checkpoint.command(name="delete", help="Delete a checkpoint")
    async def checkpoint_delete(self, ctx, checkpoint_id: int):
        """
        Delete a checkpoint.
        
        Usage: !checkpoint delete <checkpoint_id>
        """
        user_id = str(ctx.author.id)
        
        try:
            # Get checkpoint's resolution to verify ownership
            checkpoints = self.resolution_storage.get_user_incomplete_checkpoints(user_id)
            
            # Check if checkpoint belongs to user
            checkpoint_found = False
            for cp in checkpoints:
                if cp['id'] == checkpoint_id:
                    checkpoint_found = True
                    break
            
            # Also check completed checkpoints
            if not checkpoint_found:
                resolutions = self.resolution_storage.get_user_resolutions(user_id, include_completed=True)
                for res in resolutions:
                    for cp in res['checkpoints']:
                        if cp['id'] == checkpoint_id:
                            checkpoint_found = True
                            break
                    if checkpoint_found:
                        break
            
            if not checkpoint_found:
                await ctx.send("❌ Checkpoint not found or doesn't belong to you.")
                return
            
            success = self.resolution_storage.delete_checkpoint(checkpoint_id)
            
            if success:
                await ctx.send(f"✅ Checkpoint #{checkpoint_id} deleted.")
            else:
                await ctx.send("❌ Failed to delete checkpoint.")
            
        except Exception as e:
            logger.error(f"Error deleting checkpoint: {e}", exc_info=True)
            await ctx.send("❌ Failed to delete checkpoint. Please try again.")


async def setup(bot):
    """Load the cog."""
    await bot.add_cog(Resolutions(bot))

