from typing import Dict, Optional, List
from data.sqlite_storage import SQLiteStorage


class UserAITracker(SQLiteStorage):
    def __init__(self, db_path: str = "data/ai_usage.db"):
        super().__init__(db_path)
        self._init_database()
    
    def _init_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_ai_stats (
                    user_display_name TEXT PRIMARY KEY,
                    lifetime_cost REAL DEFAULT 0,
                    lifetime_tokens INTEGER DEFAULT 0,
                    lifetime_credit REAL DEFAULT 0
                )
            """)
            conn.commit()
    
    def log_ai_usage(
        self,
        user_display_name: str,
        cost: float = 0.0,
        tokens_total: int = 0
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            credit_earned = self._calculate_credit(cost, tokens_total)
            cursor.execute("""
                SELECT lifetime_cost, lifetime_tokens, lifetime_credit
                FROM user_ai_stats
                WHERE user_display_name = ?
            """, (user_display_name,))
            
            result = cursor.fetchone()
            
            if result:
                old_cost, old_tokens, old_credit = result
                new_cost = old_cost + cost
                new_tokens = old_tokens + tokens_total
                new_credit = old_credit + credit_earned
                
                cursor.execute("""
                    UPDATE user_ai_stats
                    SET lifetime_cost = ?,
                        lifetime_tokens = ?,
                        lifetime_credit = ?
                    WHERE user_display_name = ?
                """, (new_cost, new_tokens, new_credit, user_display_name))
            else:
                cursor.execute("""
                    INSERT INTO user_ai_stats 
                    (user_display_name, lifetime_cost, lifetime_tokens, lifetime_credit)
                    VALUES (?, ?, ?, ?)
                """, (user_display_name, cost, tokens_total, credit_earned))
            
            conn.commit()
        self.logger.debug(f"Updated stats for {user_display_name}")
    
    def _calculate_credit(self, cost: float, tokens: int) -> float:
        """
        Calculate social credit for a single use.
        
        Simple formula:
        - Base: 100 points
        - Token efficiency bonus
        - Cost efficiency bonus
        """
        base_credit = 100
        token_bonus = max(0, 1000 - tokens) / 10
        cost_bonus = max(0, (0.001 - cost) * 10000)
        
        return base_credit + token_bonus + cost_bonus
    
    def get_user_stats(self, user_display_name: str) -> Optional[Dict]:
        """
        Get statistics for a specific user.
        
        Returns:
            Dict with user stats or None if user not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT lifetime_cost, lifetime_tokens, lifetime_credit
                FROM user_ai_stats
                WHERE user_display_name = ?
            """, (user_display_name,))
            
            result = cursor.fetchone()
        
        if not result:
            return None
        
        return {
            "user_display_name": user_display_name,
            "lifetime_cost": result[0],
            "lifetime_tokens": result[1],
            "lifetime_credit": result[2]
        }
    
    def get_top_users(self, limit: int = 10) -> List[Dict]:
        """
        Get top users by social credit.
        
        Args:
            limit: Number of users to return
            
        Returns:
            List of user dicts sorted by credit
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_display_name, lifetime_credit, lifetime_cost, lifetime_tokens
                FROM user_ai_stats
                ORDER BY lifetime_credit DESC
                LIMIT ?
            """, (limit,))
            
            results = cursor.fetchall()
        
        return [
            {
                "user_display_name": name,
                "lifetime_credit": credit,
                "lifetime_cost": cost,
                "lifetime_tokens": tokens
            }
            for name, credit, cost, tokens in results
        ]
    
    def get_global_stats(self) -> Dict:
        """Get global statistics across all users."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT user_display_name),
                    SUM(lifetime_cost),
                    SUM(lifetime_tokens)
                FROM user_ai_stats
            """)
            
            result = cursor.fetchone()
        
        return {
            "unique_users": result[0] or 0,
            "total_cost": result[1] or 0,
            "total_tokens": result[2] or 0
        }
