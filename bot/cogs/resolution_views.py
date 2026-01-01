"""
Discord UI components for resolution tracking.

Provides buttons, modals, and dropdowns for interactive check-ins and checkpoint management.
"""

import discord
from discord import ui
from discord.ui import View, Button, Modal, TextInput, Select
from typing import Optional, List, Dict, Callable
import logging

from storage.resolutions import ResolutionStorage

logger = logging.getLogger(__name__)


class NotesModal(Modal):
    """Modal for adding optional notes to a check-in."""
    
    def __init__(
        self,
        resolution_id: int,
        status: str,
        resolution_storage: ResolutionStorage,
        on_complete: Optional[Callable] = None
    ):
        super().__init__(title="Add Notes (Optional)")
        self.resolution_id = resolution_id
        self.status = status
        self.resolution_storage = resolution_storage
        self.on_complete = on_complete
        
        self.notes_input = TextInput(
            label="How's it going?",
            placeholder="Share any updates, wins, or challenges...",
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=500
        )
        self.add_item(self.notes_input)
    
    async def on_submit(self, interaction: discord.Interaction):
        """Handle modal submission."""
        notes = self.notes_input.value if self.notes_input.value else None
        
        # Record the check-in
        result = self.resolution_storage.record_check_in(
            resolution_id=self.resolution_id,
            status=self.status,
            notes=notes
        )
        
        if 'error' in result:
            await interaction.response.send_message(
                f"❌ Error recording check-in: {result['error']}",
                ephemeral=True
            )
            return
        
        # Build response message
        status_emoji = "✅" if self.status == "on_track" else "💪"
        status_text = "On Track" if self.status == "on_track" else "Struggling (but showing up!)"
        
        message = f"{status_emoji} Check-in recorded: **{status_text}**\n"
        
        if result['new_streak'] > 0:
            message += f"🔥 Current streak: **{result['new_streak']}** check-ins\n"
        
        if result.get('streak_milestone'):
            message += f"🎉 **Streak milestone reached: {result['streak_milestone']} weeks!**\n"
        
        next_date = result['next_check_date'][:10]  # Get just the date part
        message += f"📅 Next check-in: {next_date}"
        
        await interaction.response.send_message(message, ephemeral=True)
        
        # Call completion callback if provided
        if self.on_complete:
            await self.on_complete(interaction, result)


class CheckInButton(Button):
    """Button for check-in status response."""
    
    def __init__(
        self,
        resolution_id: int,
        status: str,
        resolution_storage: ResolutionStorage,
        label: str,
        style: discord.ButtonStyle,
        emoji: Optional[str] = None
    ):
        super().__init__(
            label=label,
            style=style,
            emoji=emoji,
            custom_id=f"checkin_{resolution_id}_{status}"
        )
        self.resolution_id = resolution_id
        self.status = status
        self.resolution_storage = resolution_storage
    
    async def callback(self, interaction: discord.Interaction):
        """Handle button click - open notes modal for on_track/struggling."""
        if self.status in ["on_track", "struggling"]:
            # Open modal for notes
            modal = NotesModal(
                resolution_id=self.resolution_id,
                status=self.status,
                resolution_storage=self.resolution_storage
            )
            await interaction.response.send_modal(modal)
        else:
            # Skip - record directly without modal
            result = self.resolution_storage.record_check_in(
                resolution_id=self.resolution_id,
                status="skipped",
                notes="User chose to skip"
            )
            
            if 'error' in result:
                await interaction.response.send_message(
                    f"❌ Error: {result['error']}",
                    ephemeral=True
                )
                return
            
            next_date = result['next_check_date'][:10]
            await interaction.response.send_message(
                f"⏭️ Check-in skipped. Your streak has been reset.\n"
                f"📅 Next check-in: {next_date}",
                ephemeral=True
            )


class CheckInView(View):
    """View containing check-in buttons for a single resolution."""
    
    def __init__(
        self,
        resolution_id: int,
        resolution_storage: ResolutionStorage,
        timeout: float = 86400  # 24 hours
    ):
        super().__init__(timeout=timeout)
        self.resolution_id = resolution_id
        self.resolution_storage = resolution_storage
        
        # Add check-in buttons
        self.add_item(CheckInButton(
            resolution_id=resolution_id,
            status="on_track",
            resolution_storage=resolution_storage,
            label="On Track",
            style=discord.ButtonStyle.success,
            emoji="✅"
        ))
        
        self.add_item(CheckInButton(
            resolution_id=resolution_id,
            status="struggling",
            resolution_storage=resolution_storage,
            label="Struggling",
            style=discord.ButtonStyle.primary,
            emoji="💪"
        ))
        
        self.add_item(CheckInButton(
            resolution_id=resolution_id,
            status="skipped",
            resolution_storage=resolution_storage,
            label="Skip",
            style=discord.ButtonStyle.secondary,
            emoji="⏭️"
        ))
    
    async def on_timeout(self):
        """Called when the view times out."""
        # Buttons will automatically become non-interactive
        pass


class MultiResolutionCheckInView(View):
    """View for checking in on multiple resolutions at once."""
    
    def __init__(
        self,
        resolutions: List[Dict],
        resolution_storage: ResolutionStorage,
        timeout: float = 86400
    ):
        super().__init__(timeout=timeout)
        self.resolutions = resolutions
        self.resolution_storage = resolution_storage
        
        # Add a row of buttons for each resolution (up to 5)
        for i, resolution in enumerate(resolutions[:5]):
            res_id = resolution['id']
            
            # Create buttons with row assignment
            on_track_btn = CheckInButton(
                resolution_id=res_id,
                status="on_track",
                resolution_storage=resolution_storage,
                label=f"#{i+1} ✅",
                style=discord.ButtonStyle.success
            )
            on_track_btn.row = i
            
            struggling_btn = CheckInButton(
                resolution_id=res_id,
                status="struggling",
                resolution_storage=resolution_storage,
                label=f"#{i+1} 💪",
                style=discord.ButtonStyle.primary
            )
            struggling_btn.row = i
            
            skip_btn = CheckInButton(
                resolution_id=res_id,
                status="skipped",
                resolution_storage=resolution_storage,
                label=f"#{i+1} ⏭️",
                style=discord.ButtonStyle.secondary
            )
            skip_btn.row = i
            
            self.add_item(on_track_btn)
            self.add_item(struggling_btn)
            self.add_item(skip_btn)


class CheckpointSelect(Select):
    """Dropdown for selecting a checkpoint to complete or add."""
    
    def __init__(
        self,
        checkpoints: List[Dict],
        resolution_storage: ResolutionStorage,
        action: str = "complete"  # "complete" or "add"
    ):
        self.resolution_storage = resolution_storage
        self.action = action
        
        options = []
        
        if action == "complete":
            # Show incomplete checkpoints (all checkpoints passed are already incomplete)
            for cp in checkpoints:
                options.append(discord.SelectOption(
                    label=cp['text'][:100],  # Discord limit
                    value=str(cp['id']),
                    description=f"Resolution: {cp.get('resolution_text', '')[:50]}",
                    emoji="⬜"
                ))
            
            if not options:
                options.append(discord.SelectOption(
                    label="No incomplete checkpoints",
                    value="none",
                    description="All checkpoints are complete!"
                ))
        
        super().__init__(
            placeholder="Select a checkpoint to complete..." if action == "complete" else "Select action...",
            min_values=1,
            max_values=1,
            options=options[:25]  # Discord limit
        )
    
    async def callback(self, interaction: discord.Interaction):
        """Handle checkpoint selection."""
        selected_value = self.values[0]
        
        if selected_value == "none":
            await interaction.response.send_message(
                "🎉 All your checkpoints are complete! Great job!",
                ephemeral=True
            )
            return
        
        checkpoint_id = int(selected_value)
        result = self.resolution_storage.complete_checkpoint(checkpoint_id)
        
        if 'error' in result:
            await interaction.response.send_message(
                f"❌ Error: {result['error']}",
                ephemeral=True
            )
            return
        
        # Build response message
        progress = result['new_progress']
        message = f"✅ Checkpoint completed!\n"
        message += f"📊 Progress: **{progress['completed']}/{progress['total']}** ({progress['percentage']}%)\n"
        
        # Check for milestone
        if result.get('milestone'):
            milestone_messages = {
                25: "🌟 **Great start!** You've completed 25% of your checkpoints!",
                50: "🌟 **Halfway there!** 50% of checkpoints complete!",
                75: "🌟 **Almost there!** 75% done - the finish line is in sight!",
                100: "🎉 **RESOLUTION COMPLETE!** You finished all checkpoints!"
            }
            message += f"\n{milestone_messages.get(result['milestone'], '')}"
        
        # Check if all completed
        if result.get('all_completed'):
            message += "\n\n🏆 Congratulations! Consider archiving this resolution."
        
        await interaction.response.send_message(message, ephemeral=True)


class CheckpointView(View):
    """View for checkpoint management dropdown."""
    
    def __init__(
        self,
        checkpoints: List[Dict],
        resolution_storage: ResolutionStorage,
        timeout: float = 300  # 5 minutes
    ):
        super().__init__(timeout=timeout)
        
        if checkpoints:
            self.add_item(CheckpointSelect(
                checkpoints=checkpoints,
                resolution_storage=resolution_storage,
                action="complete"
            ))


class AddCheckpointModal(Modal):
    """Modal for adding a new checkpoint to a resolution."""
    
    def __init__(
        self,
        resolution_id: int,
        resolution_storage: ResolutionStorage
    ):
        super().__init__(title="Add Checkpoint")
        self.resolution_id = resolution_id
        self.resolution_storage = resolution_storage
        
        self.checkpoint_input = TextInput(
            label="Checkpoint/Sub-task",
            placeholder="What's a step toward your goal?",
            style=discord.TextStyle.short,
            required=True,
            max_length=200
        )
        self.add_item(self.checkpoint_input)
    
    async def on_submit(self, interaction: discord.Interaction):
        """Handle checkpoint addition."""
        text = self.checkpoint_input.value.strip()
        
        if not text:
            await interaction.response.send_message(
                "❌ Checkpoint text cannot be empty.",
                ephemeral=True
            )
            return
        
        checkpoint_id = self.resolution_storage.add_checkpoint(
            resolution_id=self.resolution_id,
            text=text
        )
        
        # Get updated resolution with checkpoints
        resolution = self.resolution_storage.get_resolution(self.resolution_id)
        progress = resolution['checkpoint_progress']
        
        await interaction.response.send_message(
            f"✅ Checkpoint added: **{text}**\n"
            f"📊 Total checkpoints: {progress['total']}",
            ephemeral=True
        )


class ResolutionSelect(Select):
    """Dropdown for selecting a resolution to add checkpoints to."""
    
    def __init__(
        self,
        resolutions: List[Dict],
        resolution_storage: ResolutionStorage
    ):
        self.resolution_storage = resolution_storage
        self.resolutions = resolutions
        
        options = []
        for res in resolutions:
            progress = res['checkpoint_progress']
            options.append(discord.SelectOption(
                label=res['text'][:100],
                value=str(res['id']),
                description=f"{progress['completed']}/{progress['total']} checkpoints",
                emoji="🎯"
            ))
        
        if not options:
            options.append(discord.SelectOption(
                label="No resolutions",
                value="none",
                description="Create a resolution first with !resolution set"
            ))
        
        super().__init__(
            placeholder="Select a resolution to add checkpoint to...",
            min_values=1,
            max_values=1,
            options=options[:25]
        )
    
    async def callback(self, interaction: discord.Interaction):
        """Open modal to add checkpoint to selected resolution."""
        selected_value = self.values[0]
        
        if selected_value == "none":
            await interaction.response.send_message(
                "❌ You don't have any resolutions yet. "
                "Create one with `!resolution set <frequency> <text>`",
                ephemeral=True
            )
            return
        
        resolution_id = int(selected_value)
        modal = AddCheckpointModal(
            resolution_id=resolution_id,
            resolution_storage=self.resolution_storage
        )
        await interaction.response.send_modal(modal)


class AddCheckpointView(View):
    """View for adding checkpoints to resolutions."""
    
    def __init__(
        self,
        resolutions: List[Dict],
        resolution_storage: ResolutionStorage,
        timeout: float = 300
    ):
        super().__init__(timeout=timeout)
        
        if resolutions:
            self.add_item(ResolutionSelect(
                resolutions=resolutions,
                resolution_storage=resolution_storage
            ))


class SetResolutionModal(Modal):
    """Modal for creating a new resolution."""
    
    def __init__(self, resolution_storage: ResolutionStorage):
        super().__init__(title="Create New Resolution")
        self.resolution_storage = resolution_storage
        
        self.resolution_input = TextInput(
            label="Your Resolution",
            placeholder="What do you want to achieve?",
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=500
        )
        self.add_item(self.resolution_input)
    
    async def on_submit(self, interaction: discord.Interaction):
        """Handle resolution creation - will be overridden by command handler."""
        # This is a placeholder - the actual handling is done in the cog
        pass


class ConfirmDeleteView(View):
    """View for confirming resolution deletion."""
    
    def __init__(
        self,
        resolution_id: int,
        resolution_storage: ResolutionStorage,
        timeout: float = 60
    ):
        super().__init__(timeout=timeout)
        self.resolution_id = resolution_id
        self.resolution_storage = resolution_storage
    
    @ui.button(label="Yes, Delete", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def confirm_delete(self, interaction: discord.Interaction, button: Button):
        """Confirm deletion."""
        success = self.resolution_storage.delete_resolution(self.resolution_id)
        
        if success:
            await interaction.response.edit_message(
                content="✅ Resolution deleted.",
                view=None
            )
        else:
            await interaction.response.edit_message(
                content="❌ Failed to delete resolution.",
                view=None
            )
        self.stop()
    
    @ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_delete(self, interaction: discord.Interaction, button: Button):
        """Cancel deletion."""
        await interaction.response.edit_message(
            content="❌ Deletion cancelled.",
            view=None
        )
        self.stop()


class ConfirmDeleteAllView(View):
    """View for confirming deletion of all resolutions."""
    
    def __init__(
        self,
        user_id: str,
        resolution_storage: ResolutionStorage,
        timeout: float = 60
    ):
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.resolution_storage = resolution_storage
    
    @ui.button(label="Yes, Delete All", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def confirm_delete_all(self, interaction: discord.Interaction, button: Button):
        """Confirm deletion of all resolutions."""
        count = self.resolution_storage.delete_all_user_resolutions(self.user_id)
        
        if count > 0:
            await interaction.response.edit_message(
                content=f"✅ Deleted all {count} resolution(s) and all associated data.",
                view=None
            )
        else:
            await interaction.response.edit_message(
                content="❌ No resolutions found to delete.",
                view=None
            )
        self.stop()
    
    @ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_delete_all(self, interaction: discord.Interaction, button: Button):
        """Cancel deletion."""
        await interaction.response.edit_message(
            content="❌ Deletion cancelled.",
            view=None
        )
        self.stop()


def build_check_in_embed(
    resolutions: List[Dict],
    user_display_name: str
) -> discord.Embed:
    """
    Build an embed for check-in prompts.
    
    Args:
        resolutions: List of resolutions due for check-in
        user_display_name: User's display name
        
    Returns:
        Discord Embed for check-in prompt
    """
    if len(resolutions) == 1:
        res = resolutions[0]
        title = f"📋 Check-in Time!"
        
        # Build streak display
        streak_text = ""
        if res['current_streak'] > 0:
            streak_text = f" 🔥 {res['current_streak']}-{res['frequency']} streak!"
        
        embed = discord.Embed(
            title=title + streak_text,
            description=f"**Resolution:** {res['text']}",
            color=discord.Color.blue()
        )
        
        # Add checkpoint progress
        progress = res['checkpoint_progress']
        if progress['total'] > 0:
            progress_bar = _build_progress_bar(progress['percentage'])
            embed.add_field(
                name="📊 Checkpoint Progress",
                value=f"{progress_bar} {progress['completed']}/{progress['total']} ({progress['percentage']}%)",
                inline=False
            )
            
            # Show checkpoints
            checkpoint_text = ""
            for cp in res['checkpoints'][:8]:  # Limit to 8
                emoji = "✅" if cp['is_completed'] else "⬜"
                checkpoint_text += f"{emoji} {cp['text']}\n"
            
            if checkpoint_text:
                embed.add_field(
                    name="Checkpoints",
                    value=checkpoint_text,
                    inline=False
                )
        
        embed.set_footer(text="How's it going? Click a button to respond!")
        
    else:
        # Multiple resolutions
        embed = discord.Embed(
            title=f"📋 You have {len(resolutions)} check-ins due!",
            color=discord.Color.blue()
        )
        
        for res in resolutions[:5]:  # Limit to 5
            progress = res['checkpoint_progress']
            streak_text = f" 🔥{res['current_streak']}" if res['current_streak'] > 0 else ""
            
            # Use per-user display ID if available
            display_id = res.get('user_display_id', res['id'])
            
            field_value = f"Progress: {progress['completed']}/{progress['total']} checkpoints"
            if res['current_streak'] > 0:
                field_value += f"\nStreak: {res['current_streak']} {res['frequency']}s"
            
            embed.add_field(
                name=f"#{display_id} {res['text'][:50]}{'...' if len(res['text']) > 50 else ''}{streak_text}",
                value=field_value,
                inline=False
            )
        
        embed.set_footer(text="Use the buttons below to check in on each resolution!")
    
    return embed


def build_resolution_list_embed(
    resolutions: List[Dict],
    user_display_name: str
) -> discord.Embed:
    """Build an embed showing all user resolutions."""
    if not resolutions:
        embed = discord.Embed(
            title="📋 Your Resolutions",
            description="You don't have any active resolutions yet.\n"
                       "Create one with `!resolution set <frequency> <text>`",
            color=discord.Color.blue()
        )
        return embed
    
    embed = discord.Embed(
        title=f"📋 {user_display_name}'s Resolutions",
        color=discord.Color.blue()
    )
    
    for res in resolutions:
        progress = res['checkpoint_progress']
        streak_text = f" 🔥{res['current_streak']}" if res['current_streak'] > 0 else ""
        
        # Use per-user display ID if available, otherwise fall back to global ID
        display_id = res.get('user_display_id', res['id'])
        
        # Build field value
        lines = []
        
        # Progress bar
        if progress['total'] > 0:
            progress_bar = _build_progress_bar(progress['percentage'])
            lines.append(f"{progress_bar} {progress['completed']}/{progress['total']} ({progress['percentage']}%)")
        
        # Frequency with check-in day and next check date
        next_date = res['next_check_date'][:10] if res['next_check_date'] else "N/A"
        check_day = res.get('check_day_display', '')
        freq = res['frequency'].title()
        
        if check_day:
            if res['frequency'] in ['weekly', 'biweekly']:
                lines.append(f"📅 {freq} ({check_day}s) • Next: {next_date}")
            else:
                lines.append(f"📅 {freq} ({check_day}) • Next: {next_date}")
        else:
            lines.append(f"📅 {freq} • Next: {next_date}")
        
        # Streak
        if res['current_streak'] > 0:
            lines.append(f"🔥 Streak: {res['current_streak']} (Best: {res['longest_streak']})")
        
        # Checkpoints preview
        incomplete = [cp for cp in res['checkpoints'] if not cp['is_completed']]
        if incomplete:
            lines.append(f"⬜ Next: {incomplete[0]['text'][:40]}...")
        
        embed.add_field(
            name=f"#{display_id} {res['text'][:60]}{'...' if len(res['text']) > 60 else ''}{streak_text}",
            value="\n".join(lines),
            inline=False
        )
    
    embed.set_footer(text="Use !checkpoint to complete sub-tasks • !resolution summary for AI analysis")
    
    return embed


def build_completion_embed(resolution: Dict) -> discord.Embed:
    """Build an embed for a completed resolution."""
    embed = discord.Embed(
        title="🎉 Resolution Complete!",
        description=f"**{resolution['text']}**",
        color=discord.Color.gold()
    )
    
    # Show all checkpoints
    checkpoint_text = ""
    for cp in resolution['checkpoints']:
        checkpoint_text += f"✅ {cp['text']}\n"
    
    if checkpoint_text:
        embed.add_field(
            name="Completed Checkpoints",
            value=checkpoint_text,
            inline=False
        )
    
    # Stats
    stats_text = f"🔥 Final Streak: {resolution['current_streak']}\n"
    stats_text += f"🏆 Longest Streak: {resolution['longest_streak']}"
    
    embed.add_field(
        name="Stats",
        value=stats_text,
        inline=False
    )
    
    embed.set_footer(text="Congratulations on achieving your goal!")
    
    return embed


def _build_progress_bar(percentage: float, length: int = 10) -> str:
    """Build a text-based progress bar."""
    filled = int(percentage / 100 * length)
    empty = length - filled
    return "█" * filled + "░" * empty

