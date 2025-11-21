"""
Custom admin check decorator that applies social credit penalties.

This decorator wraps @commands.is_owner() and applies a -200 point penalty
to users who attempt to use admin commands without permission.
"""

from discord.ext import commands
from config import Config


def admin_with_penalty():
    """
    Custom check that requires owner status and applies penalty if not owner.
    
    Usage:
        @admin_with_penalty()
        async def my_admin_command(self, ctx):
            ...
    """
    async def predicate(ctx):
        # Check if user is owner
        is_owner = await ctx.bot.is_owner(ctx.author)
        
        if is_owner:
            return True
        
        # Not owner - apply penalty if social credit is enabled
        if Config.SOCIAL_CREDIT_ENABLED:
            try:
                from ai.social_credit import SocialCreditManager
                manager = SocialCreditManager()
                
                user_id = str(ctx.author.id)
                display_name = ctx.author.display_name
                
                # Apply penalty
                new_score = await manager.apply_penalty(
                    user_id,
                    "unauthorized_admin_command",
                    display_name
                )
                
                # Get old tier and new tier for notification
                old_tier = manager.get_tone_tier(new_score + 200)
                new_tier = manager.get_tone_tier(new_score)
                
                message = f"🚨 **SOCIAL CREDIT VIOLATION** 🚨\n"
                message += f"{ctx.author.mention} attempted to access admin command!\n"
                message += f"**-200 SOCIAL CREDIT**\n"
                message += f"New score: {new_score}"
                
                if old_tier != new_tier:
                    message += f"\n\nTier changed: {old_tier} → {new_tier}"
                
                await ctx.send(message)
            except Exception as e:
                # If penalty fails, still deny access
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to apply admin command penalty: {e}", exc_info=True)
                await ctx.send("🚫 **Access Denied!** You don't have permission to use admin commands.")
        
        # Raise NotOwner to trigger normal error handling
        raise commands.NotOwner("You are not the bot owner.")
    
    return commands.check(predicate)

