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
            await ctx.send(
                "**Resolution Commands:**\n"
                "`!resolution set <frequency> <text>` - Create a resolution\n"
                "`!resolution list` - View your resolutions\n"
                "`!resolution check <id> <status>` - Record a check-in\n"
                "`!resolution summary` - Get AI analysis of your progress\n"
                "`!resolution delete <id>` - Delete a resolution\n\n"
                "**Frequencies:** weekly, biweekly, monthly\n"
                "**Statuses:** on_track, struggling"
            )
    
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
            
            # Get the created resolution to show check-in day
            resolution = self.resolution_storage.get_resolution(resolution_id)
            check_day_display = resolution.get('check_day_display', 'N/A')
            next_check = resolution.get('next_check_date', '')[:10]  # Get just date part
            
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
                value=f"#{resolution_id}",
                inline=True
            )
            embed.add_field(
                name="Next Step",
                value=f"Add checkpoints with:\n`!checkpoint add {resolution_id} \"Your sub-task\"`",
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
    
    @resolution.command(name="check", help="Record a check-in for a resolution")
    async def resolution_check(self, ctx, resolution_id: int, status: str, *, notes: Optional[str] = None):
        """
        Record a check-in response.
        
        Usage: !resolution check <id> <status> [notes]
        Statuses: on_track, struggling
        
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
            # Verify resolution belongs to user
            resolution = self.resolution_storage.get_resolution(resolution_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            # Record check-in
            result = self.resolution_storage.record_check_in(
                resolution_id=resolution_id,
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
        """
        user_id = str(ctx.author.id)
        new_text = new_text.strip('"\'')
        
        try:
            resolution = self.resolution_storage.get_resolution(resolution_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            success = self.resolution_storage.update_resolution(
                resolution_id=resolution_id,
                text=new_text
            )
            
            if success:
                await ctx.send(f"✅ Resolution updated: **{new_text}**")
            else:
                await ctx.send("❌ Failed to update resolution.")
            
        except Exception as e:
            logger.error(f"Error editing resolution: {e}", exc_info=True)
            await ctx.send("❌ Failed to edit resolution. Please try again.")
    
    @resolution.command(name="delete", help="Delete a resolution")
    async def resolution_delete(self, ctx, resolution_id: int):
        """
        Delete a resolution and all its checkpoints.
        
        Usage: !resolution delete <id>
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
            
            # Send confirmation with buttons
            view = ConfirmDeleteView(
                resolution_id=resolution_id,
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
                        await ctx.send(
                            "❌ You don't have any resolutions yet.\n"
                            "Create one with `!resolution set <frequency> <text>`"
                        )
                    else:
                        await ctx.send(
                            "🎉 All your checkpoints are complete! Great job!\n"
                            "Add more with `!checkpoint add <resolution_id> <text>`"
                        )
                    return
                
                view = CheckpointView(
                    checkpoints=checkpoints,
                    resolution_storage=self.resolution_storage
                )
                
                await ctx.send(
                    "**Select a checkpoint to mark as complete:**",
                    view=view
                )
                
            except Exception as e:
                logger.error(f"Error showing checkpoint dropdown: {e}", exc_info=True)
                await ctx.send("❌ Failed to load checkpoints. Please try again.")
    
    @checkpoint.command(name="add", help="Add a checkpoint to a resolution")
    async def checkpoint_add(self, ctx, resolution_id: int, *, text: str):
        """
        Add a checkpoint (sub-task) to a resolution.
        
        Usage: !checkpoint add <resolution_id> <text>
        Example: !checkpoint add 1 "Join a gym"
        """
        user_id = str(ctx.author.id)
        text = text.strip('"\'')
        
        if len(text) < 2:
            await ctx.send("❌ Checkpoint text is too short.")
            return
        
        if len(text) > 200:
            await ctx.send("❌ Checkpoint text is too long (max 200 characters).")
            return
        
        try:
            # Verify resolution belongs to user
            resolution = self.resolution_storage.get_resolution(resolution_id)
            if not resolution:
                await ctx.send("❌ Resolution not found.")
                return
            
            if resolution['user_id'] != user_id:
                await ctx.send("❌ This resolution doesn't belong to you.")
                return
            
            # Add checkpoint
            checkpoint_id = self.resolution_storage.add_checkpoint(
                resolution_id=resolution_id,
                text=text
            )
            
            # Get updated progress
            resolution = self.resolution_storage.get_resolution(resolution_id)
            progress = resolution['checkpoint_progress']
            
            embed = discord.Embed(
                title="✅ Checkpoint Added",
                description=f"**{text}**",
                color=discord.Color.green()
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
            embed.add_field(
                name="Checkpoint ID",
                value=f"#{checkpoint_id}",
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error adding checkpoint: {e}", exc_info=True)
            await ctx.send("❌ Failed to add checkpoint. Please try again.")
    
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
                title=f"📋 Checkpoints for Resolution #{resolution_id}",
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

