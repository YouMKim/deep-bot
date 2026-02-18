"""
Prediction Market Manager for Deep-Bot.

Manages DeepCoin balances, binary prediction markets, bets, and pool-split payouts.
"""

import sqlite3
import logging
import asyncio
from typing import Optional, Dict, List, Tuple
from storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)


class PredictionMarketManager(SQLiteStorage):
    """Manages prediction markets, balances, and bets."""

    CURRENCY_NAME = "DeepCoin"
    CURRENCY_SYMBOL = "DC"

    def __init__(self, db_path: str = "data/prediction_market.db", initial_balance: int = 5000):
        super().__init__(db_path)
        self.initial_balance = initial_balance
        self._init_database()

    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_balances (
                    user_id TEXT PRIMARY KEY,
                    balance INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS markets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    creator_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    outcome INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (market_id) REFERENCES markets(id)
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_market ON bets(market_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_user ON bets(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status)")

            conn.commit()

    async def get_or_initialize_balance(self, user_id: str) -> int:
        """Get user's balance, initializing with initial_balance if new."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_or_initialize_balance_sync, user_id
        )

    def _get_or_initialize_balance_sync(self, user_id: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT balance FROM prediction_balances WHERE user_id = ?",
                (user_id,),
            )
            result = cursor.fetchone()
            if result:
                return result[0]

            cursor.execute(
                """INSERT INTO prediction_balances (user_id, balance, created_at, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                (user_id, self.initial_balance),
            )
            conn.commit()
            logger.info(f"Initialized balance for user {user_id}: {self.initial_balance} {self.CURRENCY_SYMBOL}")
            return self.initial_balance

    async def get_balance(self, user_id: str) -> Optional[int]:
        """Get balance without initializing. Returns None if user has never joined."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_balance_sync, user_id)

    def _get_balance_sync(self, user_id: str) -> Optional[int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT balance FROM prediction_balances WHERE user_id = ?",
                (user_id,),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    async def create_market(self, creator_id: str, question: str) -> int:
        """Create a new market. Returns market ID."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._create_market_sync, creator_id, question
        )

    def _create_market_sync(self, creator_id: str, question: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO markets (creator_id, question, status, outcome)
                   VALUES (?, ?, 'open', NULL)""",
                (creator_id, question.strip()),
            )
            market_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Created market #{market_id}: {question[:50]}...")
            return market_id

    async def get_market(self, market_id: int) -> Optional[Dict]:
        """Get market by ID."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_market_sync, market_id)

    def _get_market_sync(self, market_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, creator_id, question, status, outcome, created_at, resolved_at FROM markets WHERE id = ?",
                (market_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "creator_id": row[1],
                "question": row[2],
                "status": row[3],
                "outcome": row[4],
                "created_at": row[5],
                "resolved_at": row[6],
            }

    async def get_market_totals(self, market_id: int) -> Tuple[int, int]:
        """Get (yes_total, no_total) for a market."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_market_totals_sync, market_id)

    def _get_market_totals_sync(self, market_id: int) -> Tuple[int, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT side, COALESCE(SUM(amount), 0) FROM bets WHERE market_id = ? GROUP BY side",
                (market_id,),
            )
            totals = {"yes": 0, "no": 0}
            for row in cursor.fetchall():
                totals[row[0]] = row[1]
            return totals["yes"], totals["no"]

    async def get_user_bet(self, market_id: int, user_id: str) -> Optional[Dict]:
        """Get user's total bet on a market (aggregated if multiple bets)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_user_bet_sync, market_id, user_id)

    def _get_user_bet_sync(self, market_id: int, user_id: str) -> Optional[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT side, SUM(amount) FROM bets
                   WHERE market_id = ? AND user_id = ?
                   GROUP BY side""",
                (market_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {"side": row[0], "amount": row[1]}

    async def place_bet(self, market_id: int, user_id: str, side: str, amount: int) -> Tuple[bool, str]:
        """
        Place a bet. Returns (success, error_message).
        side must be 'yes' or 'no'.
        """
        if side not in ("yes", "no"):
            return False, "Side must be 'yes' or 'no'."
        if amount < 1:
            return False, "Amount must be at least 1."

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._place_bet_sync, market_id, user_id, side, amount
        )

    def _place_bet_sync(self, market_id: int, user_id: str, side: str, amount: int) -> Tuple[bool, str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            market = self._get_market_sync(market_id)
            if not market:
                return False, "Market not found."
            if market["status"] != "open":
                return False, f"Market is {market['status']}, no longer accepting bets."

            balance = self._get_balance_sync(user_id)
            if balance is None:
                self._get_or_initialize_balance_sync(user_id)
                balance = self.initial_balance
            if amount > balance:
                return False, f"Insufficient balance. You have {balance} {self.CURRENCY_SYMBOL}."

            cursor.execute(
                """INSERT INTO bets (market_id, user_id, side, amount)
                   VALUES (?, ?, ?, ?)""",
                (market_id, user_id, side, amount),
            )
            cursor.execute(
                """UPDATE prediction_balances
                   SET balance = balance - ?, updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = ?""",
                (amount, user_id),
            )
            conn.commit()
            logger.info(f"Bet placed: user {user_id} bet {amount} on {side} for market #{market_id}")
            return True, ""

    async def resolve_market(self, market_id: int, outcome: bool) -> Tuple[bool, str]:
        """
        Resolve market. outcome=True means YES won, False means NO won.
        Returns (success, error_message).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._resolve_market_sync, market_id, outcome)

    def _resolve_market_sync(self, market_id: int, outcome: bool) -> Tuple[bool, str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            market = self._get_market_sync(market_id)
            if not market:
                return False, "Market not found."
            if market["status"] != "open":
                return False, f"Market is already {market['status']}."

            yes_total, no_total = self._get_market_totals_sync(market_id)
            total_pot = yes_total + no_total

            winning_side = "yes" if outcome else "no"
            losing_side = "no" if outcome else "yes"
            winning_total = yes_total if outcome else no_total

            if winning_total == 0:
                cursor.execute(
                    """UPDATE markets SET status = 'resolved', outcome = ?, resolved_at = CURRENT_TIMESTAMP
                       WHERE id = ?""",
                    (1 if outcome else 0, market_id),
                )
                conn.commit()
                return True, "Market resolved. No winners (no one bet on winning side)."

            cursor.execute(
                """SELECT user_id, SUM(amount) as stake
                   FROM bets WHERE market_id = ? AND side = ?
                   GROUP BY user_id""",
                (market_id, winning_side),
            )
            winners = cursor.fetchall()

            for user_id, stake in winners:
                share = int((stake / winning_total) * total_pot)
                cursor.execute(
                    """INSERT OR IGNORE INTO prediction_balances (user_id, balance, created_at, updated_at)
                       VALUES (?, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                    (user_id,),
                )
                cursor.execute(
                    """UPDATE prediction_balances
                       SET balance = balance + ?, updated_at = CURRENT_TIMESTAMP
                       WHERE user_id = ?""",
                    (share, user_id),
                )

            cursor.execute(
                """UPDATE markets SET status = 'resolved', outcome = ?, resolved_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (1 if outcome else 0, market_id),
            )
            conn.commit()
            logger.info(f"Resolved market #{market_id}: {'YES' if outcome else 'NO'} won. Payouts distributed.")
            return True, ""

    async def list_markets(
        self, status: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> List[Dict]:
        """List markets, optionally filtered by status (open, resolved)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._list_markets_sync, status, limit, offset
        )

    def _list_markets_sync(
        self, status: Optional[str], limit: int, offset: int
    ) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    """SELECT id, creator_id, question, status, outcome, created_at
                       FROM markets WHERE status = ?
                       ORDER BY id DESC LIMIT ? OFFSET ?""",
                    (status, limit, offset),
                )
            else:
                cursor.execute(
                    """SELECT id, creator_id, question, status, outcome, created_at
                       FROM markets ORDER BY id DESC LIMIT ? OFFSET ?""",
                    (limit, offset),
                )
            rows = cursor.fetchall()
            result = []
            for row in rows:
                yes_total, no_total = self._get_market_totals_sync(row[0])
                result.append({
                    "id": row[0],
                    "creator_id": row[1],
                    "question": row[2],
                    "status": row[3],
                    "outcome": row[4],
                    "created_at": row[5],
                    "yes_total": yes_total,
                    "no_total": no_total,
                })
            return result

    async def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top balances."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_leaderboard_sync, limit)

    def _get_leaderboard_sync(self, limit: int) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT user_id, balance FROM prediction_balances
                   ORDER BY balance DESC LIMIT ?""",
                (limit,),
            )
            return [{"user_id": row[0], "balance": row[1]} for row in cursor.fetchall()]
