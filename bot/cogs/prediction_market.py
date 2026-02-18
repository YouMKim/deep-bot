"""
Prediction Market Cog

Commands and UI for the DeepCoin prediction market.
"""

import discord
from discord.ext import commands
import logging
from typing import Optional

from config import Config
from prediction_market.manager import PredictionMarketManager

logger = logging.getLogger(__name__)

MARKETS_PER_PAGE = 5


class BetAmountModal(discord.ui.Modal, title="Place Bet"):
    """Modal for entering bet amount."""

    amount_input = discord.ui.TextInput(
        label="Amount",
        placeholder="e.g. 500",
        min_length=1,
        max_length=7,
        required=True,
    )

    def __init__(self, market_id: int, side: str, manager: PredictionMarketManager, max_amount: int):
        super().__init__()
        self.market_id = market_id
        self.side = side
        self.manager = manager
        self.max_amount = max_amount
        self.amount_input.label = f"Amount (1–{max_amount} {manager.CURRENCY_SYMBOL})"

    async def on_submit(self, interaction: discord.Interaction):
        try:
            amount = int(self.amount_input.value.strip())
        except ValueError:
            await interaction.response.send_message(
                "Please enter a valid whole number.",
                ephemeral=True,
            )
            return

        if amount < 1:
            await interaction.response.send_message(
                "Amount must be at least 1.",
                ephemeral=True,
            )
            return

        if amount > self.max_amount:
            await interaction.response.send_message(
                f"Insufficient balance. You have {self.max_amount} {self.manager.CURRENCY_SYMBOL}.",
                ephemeral=True,
            )
            return

        success, err = await self.manager.place_bet(
            self.market_id,
            str(interaction.user.id),
            self.side,
            amount,
        )

        if success:
            side_label = "YES" if self.side == "yes" else "NO"
            await interaction.response.send_message(
                f"✅ Bet placed! **{amount}** {self.manager.CURRENCY_SYMBOL} on **{side_label}**.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"❌ {err}",
                ephemeral=True,
            )


class MarketDetailView(discord.ui.View):
    """View with Bet YES / Bet NO buttons for a market."""

    def __init__(self, market_id: int, manager: PredictionMarketManager, bot: commands.Bot, timeout: float = 900):
        super().__init__(timeout=timeout)
        self.market_id = market_id
        self.manager = manager
        self.bot = bot

    @discord.ui.button(label="Bet YES", style=discord.ButtonStyle.success, emoji="✅")
    async def bet_yes(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_bet_click(interaction, "yes")

    @discord.ui.button(label="Bet NO", style=discord.ButtonStyle.danger, emoji="❌")
    async def bet_no(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_bet_click(interaction, "no")

    async def _handle_bet_click(self, interaction: discord.Interaction, side: str):
        market = await self.manager.get_market(self.market_id)
        if not market:
            await interaction.response.send_message("Market not found.", ephemeral=True)
            return
        if market["status"] != "open":
            await interaction.response.send_message(
                f"Market is {market['status']}, no longer accepting bets.",
                ephemeral=True,
            )
            return

        balance = await self.manager.get_or_initialize_balance(str(interaction.user.id))
        if balance < 1:
            await interaction.response.send_message(
                f"You have no {self.manager.CURRENCY_NAME} to bet.",
                ephemeral=True,
            )
            return

        modal = BetAmountModal(
            market_id=self.market_id,
            side=side,
            manager=self.manager,
            max_amount=balance,
        )
        await interaction.response.send_modal(modal)


class MarketListView(discord.ui.View):
    """View with Prev/Next buttons for market list pagination."""

    def __init__(
        self,
        manager: PredictionMarketManager,
        status_filter: Optional[str],
        page: int,
        bot: commands.Bot,
        timeout: float = 900,
    ):
        super().__init__(timeout=timeout)
        self.manager = manager
        self.status_filter = status_filter
        self.page = page
        self.bot = bot

    @discord.ui.button(label="Prev", style=discord.ButtonStyle.secondary, emoji="⬅️")
    async def prev_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.page <= 0:
            await interaction.response.send_message("Already on first page.", ephemeral=True)
            return
        await self._send_page(interaction, self.page - 1)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.secondary, emoji="➡️")
    async def next_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._send_page(interaction, self.page + 1)

    async def _send_page(self, interaction: discord.Interaction, page: int):
        markets = await self.manager.list_markets(
            status=self.status_filter,
            limit=MARKETS_PER_PAGE,
            offset=page * MARKETS_PER_PAGE,
        )
        embed = self._build_embed(markets, page)
        view = MarketListView(
            manager=self.manager,
            status_filter=self.status_filter,
            page=page,
            bot=self.bot,
        )
        await interaction.response.edit_message(embed=embed, view=view)

    def _build_embed(self, markets: list, page: int) -> discord.Embed:
        status_label = self.status_filter or "all"
        embed = discord.Embed(
            title=f"📊 Open Markets ({status_label})",
            color=discord.Color.blue(),
        )
        if not markets:
            embed.description = "No markets found."
            return embed

        for m in markets:
            outcome_str = ""
            if m["status"] == "resolved":
                outcome_str = " ✓" if m["outcome"] else " ✗"
            embed.add_field(
                name=f"#{m['id']}{outcome_str}",
                value=f"{m['question'][:80]}{'...' if len(m['question']) > 80 else ''}\n"
                      f"YES: {m['yes_total']} DC | NO: {m['no_total']} DC",
                inline=False,
            )
        embed.set_footer(text=f"Page {page + 1} • Use !market info <id> for details")
        return embed


class PredictionMarket(commands.Cog):
    """Prediction market commands."""

    def __init__(self, bot):
        self.bot = bot
        self.manager = PredictionMarketManager(
            initial_balance=Config.PREDICTION_MARKET_INITIAL_BALANCE,
        )

    async def cog_check(self, ctx):
        if not Config.PREDICTION_MARKET_ENABLED:
            await ctx.send("❌ Prediction market is currently disabled.")
            return False
        return True

    @commands.command(name="market", aliases=["m"])
    async def market_command(self, ctx, *args):
        """
        Prediction market commands.

        !market list [open|resolved] - List markets
        !market info <id> - Market details + bet buttons
        !market create <question> - Create a market
        !market bet <id> yes|no <amount> - Place bet (text)
        !market balance [@user] - Your DeepCoin balance
        !market leaderboard - Top balances
        !market resolve <id> yes|no - Resolve market (admin)
        """
        if not args:
            await ctx.send(
                "**Prediction Market**\n"
                "`!market list [open|resolved]` - List markets\n"
                "`!market info <id>` - Market details + bet\n"
                "`!market create <question>` - Create market\n"
                "`!market bet <id> yes|no <amount>` - Place bet\n"
                "`!market balance [@user]` - Your balance\n"
                "`!market leaderboard` - Top balances\n"
                "`!market resolve <id> yes|no` - Resolve (admin)"
            )
            return

        sub = args[0].lower()
        if sub == "list":
            await self._cmd_list(ctx, args[1:])
        elif sub == "info":
            await self._cmd_info(ctx, args[1:])
        elif sub == "create":
            await self._cmd_create(ctx, args[1:])
        elif sub == "bet":
            await self._cmd_bet(ctx, args[1:])
        elif sub == "balance":
            await self._cmd_balance(ctx, args[1:])
        elif sub == "leaderboard":
            await self._cmd_leaderboard(ctx, args[1:])
        elif sub == "resolve":
            await self._cmd_resolve(ctx, args[1:])
        else:
            await ctx.send(f"❌ Unknown subcommand: `{sub}`")

    async def _cmd_list(self, ctx, args):
        status_filter = None
        if args and args[0].lower() in ("open", "resolved"):
            status_filter = args[0].lower()

        markets = await self.manager.list_markets(
            status=status_filter,
            limit=MARKETS_PER_PAGE,
            offset=0,
        )
        view = MarketListView(
            manager=self.manager,
            status_filter=status_filter,
            page=0,
            bot=self.bot,
        )
        embed = view._build_embed(markets, 0)
        await ctx.send(embed=embed, view=view)

    async def _cmd_info(self, ctx, args):
        if not args:
            await ctx.send("Usage: `!market info <id>`")
            return
        try:
            market_id = int(args[0])
        except ValueError:
            await ctx.send("Market ID must be a number.")
            return

        market = await self.manager.get_market(market_id)
        if not market:
            await ctx.send("Market not found.")
            return

        yes_total, no_total = await self.manager.get_market_totals(market_id)
        user_bet = await self.manager.get_user_bet(market_id, str(ctx.author.id))

        status_emoji = "🟢" if market["status"] == "open" else "🔴"
        outcome_str = ""
        if market["status"] == "resolved":
            outcome_str = " → **YES won** ✓" if market["outcome"] else " → **NO won** ✗"

        embed = discord.Embed(
            title=f"Market #{market_id} {status_emoji}",
            description=market["question"],
            color=discord.Color.green() if market["status"] == "open" else discord.Color.greyple(),
        )
        embed.add_field(name="YES pool", value=f"{yes_total} DC", inline=True)
        embed.add_field(name="NO pool", value=f"{no_total} DC", inline=True)
        if user_bet:
            embed.add_field(
                name="Your bet",
                value=f"{user_bet['amount']} DC on **{user_bet['side'].upper()}**",
                inline=False,
            )
        embed.add_field(
            name="Status",
            value=f"{market['status'].capitalize()}{outcome_str}",
            inline=False,
        )
        embed.set_footer(text="Click buttons below to place a bet")

        if market["status"] == "open":
            view = MarketDetailView(
                market_id=market_id,
                manager=self.manager,
                bot=self.bot,
            )
            await ctx.send(embed=embed, view=view)
        else:
            await ctx.send(embed=embed)

    async def _cmd_create(self, ctx, args):
        if not args:
            await ctx.send("Usage: `!market create <question>`")
            return

        question = " ".join(args).strip()
        if len(question) < 10:
            await ctx.send("Question must be at least 10 characters.")
            return
        if len(question) > 500:
            await ctx.send("Question must be 500 characters or less.")
            return

        market_id = await self.manager.create_market(str(ctx.author.id), question)
        await ctx.send(
            f"✅ Market **#{market_id}** created!\n"
            f"**{question}**\n"
            f"Use `!market info {market_id}` to view and bet."
        )

    async def _cmd_bet(self, ctx, args):
        if len(args) < 3:
            await ctx.send("Usage: `!market bet <id> yes|no <amount>`")
            return
        try:
            market_id = int(args[0])
            amount = int(args[2])
        except ValueError:
            await ctx.send("ID and amount must be numbers.")
            return

        side = args[1].lower()
        if side not in ("yes", "no"):
            await ctx.send("Side must be `yes` or `no`.")
            return

        success, err = await self.manager.place_bet(
            market_id,
            str(ctx.author.id),
            side,
            amount,
        )
        if success:
            side_label = "YES" if side == "yes" else "NO"
            await ctx.send(f"✅ Bet placed! **{amount}** DC on **{side_label}**.")
        else:
            await ctx.send(f"❌ {err}")

    async def _cmd_balance(self, ctx, args):
        if ctx.message.mentions:
            user = ctx.message.mentions[0]
            user_id = str(user.id)
            display_name = user.display_name
        else:
            user_id = str(ctx.author.id)
            display_name = ctx.author.display_name

        balance = await self.manager.get_or_initialize_balance(user_id)
        embed = discord.Embed(
            title=f"💰 {display_name}'s DeepCoin Balance",
            color=discord.Color.gold(),
        )
        embed.add_field(name="Balance", value=f"{balance} DC", inline=True)
        await ctx.send(embed=embed)

    async def _cmd_leaderboard(self, ctx, args):
        limit = 10
        if args:
            try:
                limit = min(int(args[0]), 25)
            except ValueError:
                pass

        entries = await self.manager.get_leaderboard(limit=limit)
        if not entries:
            await ctx.send("No one has joined the market yet.")
            return

        embed = discord.Embed(
            title="🏆 DeepCoin Leaderboard",
            color=discord.Color.gold(),
        )
        for i, e in enumerate(entries, 1):
            user = self.bot.get_user(int(e["user_id"]))
            name = user.display_name if user else f"User {e['user_id']}"
            embed.add_field(
                name=f"{i}. {name}",
                value=f"{e['balance']} DC",
                inline=False,
            )
        await ctx.send(embed=embed)

    async def _cmd_resolve(self, ctx, args):
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("❌ Only the bot owner can resolve markets.")
            return

        if len(args) < 2:
            await ctx.send("Usage: `!market resolve <id> yes|no`")
            return
        try:
            market_id = int(args[0])
        except ValueError:
            await ctx.send("Market ID must be a number.")
            return

        outcome_str = args[1].lower()
        if outcome_str not in ("yes", "no"):
            await ctx.send("Outcome must be `yes` or `no`.")
            return
        outcome = outcome_str == "yes"

        success, err = await self.manager.resolve_market(market_id, outcome)
        if success:
            result = "YES" if outcome else "NO"
            await ctx.send(f"✅ Market **#{market_id}** resolved: **{result}** won. Payouts distributed.")
        else:
            await ctx.send(f"❌ {err}")


async def setup(bot):
    await bot.add_cog(PredictionMarket(bot))
