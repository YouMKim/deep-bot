"""
Social Credit Commands Cog

Admin commands for managing social credit scores.
"""

import discord
from discord.ext import commands
import logging
from typing import Optional, List
from ai.social_credit import SocialCreditManager
from config import Config

logger = logging.getLogger(__name__)


class SocialCreditCommands(commands.Cog):
    """Commands for managing social credit scores."""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config
        self.social_credit_manager = SocialCreditManager() if Config.SOCIAL_CREDIT_ENABLED else None
    
    async def cog_check(self, ctx):
        """Check if social credit is enabled."""
        if not self.config.SOCIAL_CREDIT_ENABLED:
            await ctx.send("❌ Social Credit system is currently disabled.")
            return False
        return True
    
    @commands.command(name="socialcredit", aliases=["sc", "credit"])
    async def social_credit_command(self, ctx, *args):
        """
        Social Credit management commands.
        
        Usage:
            !socialcredit view [@user] - View your or another user's score
            !socialcredit give @user <points> - Give points to a user (Admin only)
            !socialcredit take @user <points> - Take points from a user (Admin only)
            !socialcredit set @user <score> - Set user's score (Admin only)
            !socialcredit reset @user - Reset user's score to 0 (Admin only)
            !socialcredit leaderboard [top|bottom] [limit] - Show leaderboard (Admin only)
        """
        if not args:
            await ctx.send(self._get_admin_help())
            return
        
        command = args[0].lower()
        
        # Check if user is admin for admin commands
        is_admin = await self.bot.is_owner(ctx.author)
        
        if command == "view":
            await self._handle_view(ctx, args[1:])
        elif command in ["give", "take", "set", "reset", "leaderboard"]:
            if not is_admin:
                # Apply penalty for unauthorized admin command attempt
                if self.social_credit_manager:
                    try:
                        new_score = await self.social_credit_manager.apply_penalty(
                            str(ctx.author.id),
                            "unauthorized_admin_command",
                            ctx.author.display_name
                        )
                        message = f"🚨 YOU HAVE VIOLATED THE SOCIAL CREDIT SYSTEM 🚨\n"
                        message += f"{ctx.author.mention} attempted to access admin command!\n"
                        message += f"**{Config.SOCIAL_CREDIT_PENALTY_ADMIN_COMMAND} SOCIAL CREDIT**\n"
                        message += f"New score: {new_score}"
                        await ctx.send(message)
                        return
                    except Exception as e:
                        logger.error(f"Failed to apply penalty: {e}", exc_info=True)
                        await ctx.send("❌ Error applying penalty. Please contact an admin.")
                        return
                else:
                    await ctx.send("🚫 **Access Denied!** You are not an admin. This incident will be reported.")
                    return
            
            # Admin commands
            if command == "give":
                await self._handle_give(ctx, args[1:])
            elif command == "take":
                await self._handle_take(ctx, args[1:])
            elif command == "set":
                await self._handle_set(ctx, args[1:])
            elif command == "reset":
                await self._handle_reset(ctx, args[1:])
            elif command == "leaderboard":
                await self._handle_leaderboard(ctx, args[1:])
        else:
            await ctx.send(f"❌ Unknown command: `{command}`\n{self._get_admin_help()}")
    
    async def _handle_view(self, ctx, args):
        """Handle view command."""
        if not self.social_credit_manager:
            await ctx.send("❌ Social Credit system is disabled.")
            return
        
        # Get user from mention or use author
        if ctx.message.mentions:
            user = ctx.message.mentions[0]
            user_id = str(user.id)
            user_name = user.display_name
        elif args:
            # Try to lookup by display name
            display_name_lookup = " ".join(args)
            user_stats = await self.social_credit_manager.get_user_stats_by_display_name(display_name_lookup)
            if user_stats:
                user_id = user_stats['user_id']
                user_name = user_stats['user_display_name']
            else:
                await ctx.send(f"❌ User '{display_name_lookup}' not found.")
                return
        else:
            # View own score
            user_id = str(ctx.author.id)
            user_name = ctx.author.display_name
        
        # Auto-initialize if needed
        await self.social_credit_manager.get_or_initialize_score(user_id, user_name)
        stats = await self.social_credit_manager.get_user_stats(user_id)
        
        if not stats:
            await ctx.send(f"❌ Failed to retrieve user stats.")
            return
        
        embed = discord.Embed(
            title=f"Social Credit Score: {stats['user_display_name']}",
            color=discord.Color.blue()
        )
        embed.add_field(name="Score", value=str(stats['social_credit_score']), inline=True)
        embed.add_field(
            name="Last Interaction",
            value=str(stats['last_interaction']) if stats['last_interaction'] else "Never",
            inline=False
        )
        await ctx.send(embed=embed)
    
    async def _handle_give(self, ctx, args):
        """Handle give command."""
        if not ctx.message.mentions:
            await ctx.send("❌ Please mention a user: `!socialcredit give @user <points>`")
            return
        
        if len(args) < 2:
            await ctx.send("❌ Please specify points: `!socialcredit give @user <points>`")
            return
        
        user = ctx.message.mentions[0]
        user_id = str(user.id)
        user_name = user.display_name
        
        try:
            points = int(args[1])
            if points <= 0:
                await ctx.send("❌ Points must be positive.")
                return
        except ValueError:
            await ctx.send("❌ Points must be a number.")
            return
        
        # Auto-initialize if needed
        await self.social_credit_manager.get_or_initialize_score(user_id, user_name)
        new_score = await self.social_credit_manager.update_score(
            user_id, points, f"Admin adjustment by {ctx.author.display_name}", user_name
        )
        await ctx.send(f"✅ Gave {points} points to {user_name}. New score: {new_score}")
    
    async def _handle_take(self, ctx, args):
        """Handle take command."""
        if not ctx.message.mentions:
            await ctx.send("❌ Please mention a user: `!socialcredit take @user <points>`")
            return
        
        if len(args) < 2:
            await ctx.send("❌ Please specify points: `!socialcredit take @user <points>`")
            return
        
        user = ctx.message.mentions[0]
        user_id = str(user.id)
        user_name = user.display_name
        
        try:
            points = int(args[1])
            if points <= 0:
                await ctx.send("❌ Points must be positive.")
                return
        except ValueError:
            await ctx.send("❌ Points must be a number.")
            return
        
        # Auto-initialize if needed
        await self.social_credit_manager.get_or_initialize_score(user_id, user_name)
        new_score = await self.social_credit_manager.update_score(
            user_id, -points, f"Admin adjustment by {ctx.author.display_name}", user_name
        )
        await ctx.send(f"✅ Took {points} from {user_name}. New score: {new_score}")
    
    async def _handle_set(self, ctx, args):
        """Handle set command."""
        if not ctx.message.mentions:
            await ctx.send("❌ Please mention a user: `!socialcredit set @user <score>`")
            return
        
        if len(args) < 2:
            await ctx.send("❌ Please specify score: `!socialcredit set @user <score>`")
            return
        
        user = ctx.message.mentions[0]
        user_id = str(user.id)
        user_name = user.display_name
        
        try:
            target_score = int(args[1])
            if target_score < -1000 or target_score > 1000:
                await ctx.send("❌ Score must be between -1000 and 1000.")
                return
        except ValueError:
            await ctx.send("❌ Score must be a number.")
            return
        
        # Auto-initialize if needed
        await self.social_credit_manager.get_or_initialize_score(user_id, user_name)
        stats = await self.social_credit_manager.get_user_stats(user_id)
        current_score = stats['social_credit_score']
        delta = target_score - current_score
        new_score = await self.social_credit_manager.update_score(
            user_id, delta, f"Set to {target_score} by {ctx.author.display_name}", user_name
        )
        await ctx.send(f"✅ Set {user_name}'s score from {current_score} to {new_score}")
    
    async def _handle_reset(self, ctx, args):
        """Handle reset command."""
        if ctx.message.mentions:
            user = ctx.message.mentions[0]
            user_id = str(user.id)
            user_name = user.display_name
        elif args:
            display_name_lookup = " ".join(args)
            user_stats = await self.social_credit_manager.get_user_stats_by_display_name(display_name_lookup)
            if user_stats:
                user_id = user_stats['user_id']
                user_name = user_stats['user_display_name']
            else:
                await ctx.send(f"❌ User '{display_name_lookup}' not found.")
                return
        else:
            await ctx.send("❌ Please mention a user or provide display name: `!socialcredit reset @user`")
            return
        
        # Auto-initialize if needed
        await self.social_credit_manager.get_or_initialize_score(user_id, user_name)
        stats = await self.social_credit_manager.get_user_stats(user_id)
        current_score = stats['social_credit_score']
        new_score = await self.social_credit_manager.update_score(
            user_id, -current_score, f"Reset by {ctx.author.display_name}", user_name
        )
        await ctx.send(f"✅ Reset {user_name}'s score from {current_score} to {new_score}")
    
    async def _handle_leaderboard(self, ctx, args):
        """Handle leaderboard command."""
        limit = 10
        top = True
        
        if args:
            if args[0].lower() in ["top", "high", "highest"]:
                top = True
            elif args[0].lower() in ["bottom", "low", "lowest"]:
                top = False
            
            if len(args) > 1:
                try:
                    limit = int(args[1])
                    if limit < 1 or limit > 50:
                        await ctx.send("❌ Limit must be between 1 and 50.")
                        return
                except ValueError:
                    await ctx.send("❌ Limit must be a number.")
                    return
        
        leaderboard = await self.social_credit_manager.get_leaderboard(limit, ascending=not top)
        
        if not leaderboard:
            await ctx.send("❌ No users found in database.")
            return
        
        title = f"🏆 Top {limit} Social Credit Scores" if top else f"📉 Bottom {limit} Social Credit Scores"
        embed = discord.Embed(title=title, color=discord.Color.gold())
        
        for i, user in enumerate(leaderboard, 1):
            embed.add_field(
                name=f"{i}. {user['user_display_name']}",
                value=f"Score: {user['social_credit_score']}",
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    def _get_admin_help(self) -> str:
        """Get help text for admin commands."""
        return """**Social Credit Commands:**

`!socialcredit view [@user]` - View your or another user's score
`!socialcredit give @user <points>` - Give points to a user (Admin only)
`!socialcredit take @user <points>` - Take points from a user (Admin only)
`!socialcredit set @user <score>` - Set user's score (Admin only)
`!socialcredit reset @user` - Reset user's score to 0 (Admin only)
`!socialcredit leaderboard [top|bottom] [limit]` - Show leaderboard (Admin only)

**Note:** Non-admins attempting to use admin commands will receive a penalty."""


async def setup(bot):
    """Load the cog."""
    await bot.add_cog(SocialCreditCommands(bot))
