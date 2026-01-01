"""
SQLite storage for New Year's Resolutions tracking.

Handles resolutions, checkpoints (sub-tasks), and check-ins with streak tracking.
"""

import sqlite3
import logging
import calendar
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from enum import Enum

from storage.sqlite_storage import SQLiteStorage


class CheckFrequency(Enum):
    """Check-in frequency options."""
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


class CheckInStatus(Enum):
    """Check-in response status."""
    ON_TRACK = "on_track"
    STRUGGLING = "struggling"
    SKIPPED = "skipped"


def get_next_weekday(from_date: date, target_weekday: int, weeks: int = 1) -> date:
    """
    Get the next occurrence of a specific weekday.
    
    Args:
        from_date: Starting date
        target_weekday: Target day of week (0=Monday, 6=Sunday)
        weeks: Number of weeks to add (1 for weekly, 2 for biweekly)
    
    Returns:
        Next date that falls on target_weekday
    """
    days_ahead = target_weekday - from_date.weekday()
    if days_ahead <= 0:  # Target day already happened this week or is today
        days_ahead += 7 * weeks
    else:
        days_ahead += 7 * (weeks - 1)
    return from_date + timedelta(days=days_ahead)


def get_next_month_day(from_date: date, target_day: int) -> date:
    """
    Get the next occurrence of a specific day of the month.
    
    Handles month boundaries:
    - If target_day is 29, 30, or 31 and month doesn't have that day,
      uses the last day of that month.
    
    Args:
        from_date: Starting date
        target_day: Target day of month (1-31)
    
    Returns:
        Next date that falls on or near target_day
    """
    # Start by looking at next month
    year = from_date.year
    month = from_date.month + 1
    
    if month > 12:
        month = 1
        year += 1
    
    # Get the last day of the target month
    last_day_of_month = calendar.monthrange(year, month)[1]
    
    # Use the minimum of target_day and last_day_of_month
    actual_day = min(target_day, last_day_of_month)
    
    return date(year, month, actual_day)


SCHEMA_SQL = """
-- Main resolutions table
CREATE TABLE IF NOT EXISTS resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    user_display_name TEXT,
    text TEXT NOT NULL,
    frequency TEXT NOT NULL DEFAULT 'weekly',
    next_check_date TEXT NOT NULL,
    check_day_of_week INTEGER,
    check_day_of_month INTEGER,
    current_streak INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,
    is_completed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    updated_at TEXT NOT NULL
);

-- Checkpoints (sub-tasks) for resolutions
CREATE TABLE IF NOT EXISTS resolution_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resolution_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    is_completed INTEGER DEFAULT 0,
    completed_at TEXT,
    display_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (resolution_id) REFERENCES resolutions(id) ON DELETE CASCADE
);

-- Check-in records
CREATE TABLE IF NOT EXISTS resolution_check_ins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resolution_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    reminder_sent INTEGER DEFAULT 0,
    channel_message_id TEXT,
    dm_message_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (resolution_id) REFERENCES resolutions(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_resolutions_user ON resolutions(user_id);
CREATE INDEX IF NOT EXISTS idx_resolutions_next_check ON resolutions(next_check_date, is_active);
CREATE INDEX IF NOT EXISTS idx_checkpoints_resolution ON resolution_checkpoints(resolution_id);
CREATE INDEX IF NOT EXISTS idx_check_ins_resolution ON resolution_check_ins(resolution_id);
CREATE INDEX IF NOT EXISTS idx_check_ins_created ON resolution_check_ins(created_at);
"""


class ResolutionStorage(SQLiteStorage):
    """Storage class for resolution tracking with streak management."""
    
    def __init__(self, db_path: str = "data/resolutions.db"):
        super().__init__(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            try:
                conn.executescript(SCHEMA_SQL)
                
                # Add new columns if they don't exist (migration)
                try:
                    conn.execute("ALTER TABLE resolutions ADD COLUMN check_day_of_week INTEGER")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                try:
                    conn.execute("ALTER TABLE resolutions ADD COLUMN check_day_of_month INTEGER")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                conn.commit()
            except Exception as e:
                self.logger.error(f"Failed to initialize resolutions database: {e}")
                raise
    
    # ==================== Date Calculation Helpers ====================
    
    def _calculate_next_check_date(
        self,
        frequency: str,
        check_day_of_week: Optional[int] = None,
        check_day_of_month: Optional[int] = None,
        from_date: Optional[date] = None
    ) -> str:
        """
        Calculate the next check-in date based on frequency and stored day.
        
        Args:
            frequency: 'weekly', 'biweekly', or 'monthly'
            check_day_of_week: Day of week (0=Monday, 6=Sunday) for weekly/biweekly
            check_day_of_month: Day of month (1-31) for monthly
            from_date: Starting date (defaults to today)
        
        Returns:
            ISO format date string
        """
        if from_date is None:
            from_date = date.today()
        
        freq = CheckFrequency(frequency.lower())
        
        if freq == CheckFrequency.WEEKLY:
            if check_day_of_week is not None:
                next_date = get_next_weekday(from_date, check_day_of_week, weeks=1)
            else:
                next_date = from_date + timedelta(days=7)
        
        elif freq == CheckFrequency.BIWEEKLY:
            if check_day_of_week is not None:
                next_date = get_next_weekday(from_date, check_day_of_week, weeks=2)
            else:
                next_date = from_date + timedelta(days=14)
        
        elif freq == CheckFrequency.MONTHLY:
            if check_day_of_month is not None:
                next_date = get_next_month_day(from_date, check_day_of_month)
            else:
                next_date = from_date + timedelta(days=30)
        
        else:
            next_date = from_date + timedelta(days=7)
        
        return datetime.combine(next_date, datetime.min.time()).isoformat()
    
    def _get_check_day_display(
        self,
        frequency: str,
        check_day_of_week: Optional[int],
        check_day_of_month: Optional[int]
    ) -> str:
        """Get a human-readable string for the check-in day."""
        freq = CheckFrequency(frequency.lower())
        
        if freq in [CheckFrequency.WEEKLY, CheckFrequency.BIWEEKLY]:
            if check_day_of_week is not None:
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                return days[check_day_of_week]
            return "N/A"
        
        elif freq == CheckFrequency.MONTHLY:
            if check_day_of_month is not None:
                suffix = 'th'
                if check_day_of_month in [1, 21, 31]:
                    suffix = 'st'
                elif check_day_of_month in [2, 22]:
                    suffix = 'nd'
                elif check_day_of_month in [3, 23]:
                    suffix = 'rd'
                return f"{check_day_of_month}{suffix} of each month"
            return "N/A"
        
        return "N/A"
    
    # ==================== Resolution CRUD ====================
    
    def create_resolution(
        self,
        user_id: str,
        text: str,
        frequency: str = "weekly",
        user_display_name: Optional[str] = None
    ) -> int:
        """
        Create a new resolution for a user.
        
        The check-in day is set based on the current day:
        - Weekly/Biweekly: Same day of the week as creation
        - Monthly: Same day of the month as creation
        
        Args:
            user_id: Discord user ID
            text: Resolution text
            frequency: Check-in frequency (weekly, biweekly, monthly)
            user_display_name: User's display name
            
        Returns:
            ID of the created resolution
        """
        now = datetime.now()
        now_str = now.isoformat()
        today = now.date()
        
        freq = CheckFrequency(frequency.lower())
        
        # Determine check day based on frequency
        check_day_of_week = None
        check_day_of_month = None
        
        if freq in [CheckFrequency.WEEKLY, CheckFrequency.BIWEEKLY]:
            # Store the day of the week (0=Monday, 6=Sunday)
            check_day_of_week = today.weekday()
        elif freq == CheckFrequency.MONTHLY:
            # Store the day of the month (1-31)
            check_day_of_month = today.day
        
        # Calculate first check-in date
        next_check = self._calculate_next_check_date(
            frequency=frequency.lower(),
            check_day_of_week=check_day_of_week,
            check_day_of_month=check_day_of_month,
            from_date=today
        )
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO resolutions (
                    user_id, user_display_name, text, frequency,
                    next_check_date, check_day_of_week, check_day_of_month,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, user_display_name, text, frequency.lower(),
                next_check, check_day_of_week, check_day_of_month,
                now_str, now_str
            ))
            conn.commit()
            
            resolution_id = cursor.lastrowid
            self.logger.info(f"Created resolution {resolution_id} for user {user_id}")
            return resolution_id
    
    def get_resolution(self, resolution_id: int) -> Optional[Dict]:
        """Get a resolution by ID with checkpoint progress."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, user_display_name, text, frequency,
                       next_check_date, check_day_of_week, check_day_of_month,
                       current_streak, longest_streak,
                       is_active, is_completed, created_at, completed_at
                FROM resolutions
                WHERE id = ?
            """, (resolution_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            frequency = row[4]
            check_day_of_week = row[6]
            check_day_of_month = row[7]
            
            resolution = {
                'id': row[0],
                'user_id': row[1],
                'user_display_name': row[2],
                'text': row[3],
                'frequency': frequency,
                'next_check_date': row[5],
                'check_day_of_week': check_day_of_week,
                'check_day_of_month': check_day_of_month,
                'check_day_display': self._get_check_day_display(frequency, check_day_of_week, check_day_of_month),
                'current_streak': row[8],
                'longest_streak': row[9],
                'is_active': bool(row[10]),
                'is_completed': bool(row[11]),
                'created_at': row[12],
                'completed_at': row[13]
            }
            
            # Get checkpoint progress
            checkpoints = self.get_checkpoints(resolution_id)
            resolution['checkpoints'] = checkpoints
            resolution['checkpoint_progress'] = self._calculate_checkpoint_progress(checkpoints)
            
            return resolution
    
    def get_user_resolutions(
        self,
        user_id: str,
        include_completed: bool = False
    ) -> List[Dict]:
        """Get all resolutions for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if include_completed:
                cursor.execute("""
                    SELECT id, user_id, user_display_name, text, frequency,
                           next_check_date, check_day_of_week, check_day_of_month,
                           current_streak, longest_streak,
                           is_active, is_completed, created_at, completed_at
                    FROM resolutions
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT id, user_id, user_display_name, text, frequency,
                           next_check_date, check_day_of_week, check_day_of_month,
                           current_streak, longest_streak,
                           is_active, is_completed, created_at, completed_at
                    FROM resolutions
                    WHERE user_id = ? AND is_active = 1 AND is_completed = 0
                    ORDER BY created_at DESC
                """, (user_id,))
            
            rows = cursor.fetchall()
            resolutions = []
            
            for row in rows:
                frequency = row[4]
                check_day_of_week = row[6]
                check_day_of_month = row[7]
                
                resolution = {
                    'id': row[0],
                    'user_id': row[1],
                    'user_display_name': row[2],
                    'text': row[3],
                    'frequency': frequency,
                    'next_check_date': row[5],
                    'check_day_of_week': check_day_of_week,
                    'check_day_of_month': check_day_of_month,
                    'check_day_display': self._get_check_day_display(frequency, check_day_of_week, check_day_of_month),
                    'current_streak': row[8],
                    'longest_streak': row[9],
                    'is_active': bool(row[10]),
                    'is_completed': bool(row[11]),
                    'created_at': row[12],
                    'completed_at': row[13]
                }
                
                # Get checkpoint progress
                checkpoints = self.get_checkpoints(row[0])
                resolution['checkpoints'] = checkpoints
                resolution['checkpoint_progress'] = self._calculate_checkpoint_progress(checkpoints)
                
                resolutions.append(resolution)
            
            return resolutions
    
    def update_resolution(
        self,
        resolution_id: int,
        text: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> bool:
        """
        Update a resolution's text or frequency.
        
        If frequency changes, the check day is updated to the current day.
        """
        updates = []
        params = []
        
        if text is not None:
            updates.append("text = ?")
            params.append(text)
        
        if frequency is not None:
            freq = CheckFrequency(frequency.lower())
            today = date.today()
            
            updates.append("frequency = ?")
            params.append(frequency.lower())
            
            # Update check day based on new frequency
            if freq in [CheckFrequency.WEEKLY, CheckFrequency.BIWEEKLY]:
                check_day_of_week = today.weekday()
                check_day_of_month = None
                updates.append("check_day_of_week = ?")
                params.append(check_day_of_week)
                updates.append("check_day_of_month = ?")
                params.append(check_day_of_month)
            else:  # MONTHLY
                check_day_of_week = None
                check_day_of_month = today.day
                updates.append("check_day_of_week = ?")
                params.append(check_day_of_week)
                updates.append("check_day_of_month = ?")
                params.append(check_day_of_month)
            
            # Recalculate next check date
            next_check = self._calculate_next_check_date(
                frequency=frequency.lower(),
                check_day_of_week=check_day_of_week,
                check_day_of_month=check_day_of_month,
                from_date=today
            )
            updates.append("next_check_date = ?")
            params.append(next_check)
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(resolution_id)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE resolutions
                SET {', '.join(updates)}
                WHERE id = ?
            """, params)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_resolution(self, resolution_id: int) -> bool:
        """Delete a resolution and all its checkpoints/check-ins."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Cascading deletes handle checkpoints and check-ins
            cursor.execute("DELETE FROM resolutions WHERE id = ?", (resolution_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_all_user_resolutions(self, user_id: str) -> int:
        """
        Delete all resolutions for a user.
        
        Returns:
            Number of resolutions deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Cascading deletes handle checkpoints and check-ins
            cursor.execute("DELETE FROM resolutions WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount
    
    def mark_resolution_completed(self, resolution_id: int) -> bool:
        """Mark a resolution as completed."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE resolutions
                SET is_completed = 1, completed_at = ?, updated_at = ?
                WHERE id = ?
            """, (now, now, resolution_id))
            conn.commit()
            return cursor.rowcount > 0
    
    # ==================== Checkpoint CRUD ====================
    
    def add_checkpoint(self, resolution_id: int, text: str) -> int:
        """Add a checkpoint to a resolution."""
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current max display_order
            cursor.execute("""
                SELECT COALESCE(MAX(display_order), 0) + 1
                FROM resolution_checkpoints
                WHERE resolution_id = ?
            """, (resolution_id,))
            next_order = cursor.fetchone()[0]
            
            cursor.execute("""
                INSERT INTO resolution_checkpoints (
                    resolution_id, text, display_order, created_at
                ) VALUES (?, ?, ?, ?)
            """, (resolution_id, text, next_order, now))
            conn.commit()
            
            return cursor.lastrowid
    
    def get_checkpoints(self, resolution_id: int) -> List[Dict]:
        """Get all checkpoints for a resolution."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, resolution_id, text, is_completed, completed_at, display_order
                FROM resolution_checkpoints
                WHERE resolution_id = ?
                ORDER BY display_order ASC
            """, (resolution_id,))
            
            rows = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'resolution_id': row[1],
                    'text': row[2],
                    'is_completed': bool(row[3]),
                    'completed_at': row[4],
                    'display_order': row[5]
                }
                for row in rows
            ]
    
    def get_user_incomplete_checkpoints(self, user_id: str) -> List[Dict]:
        """Get all incomplete checkpoints for a user across all resolutions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, c.resolution_id, c.text, r.text as resolution_text
                FROM resolution_checkpoints c
                JOIN resolutions r ON c.resolution_id = r.id
                WHERE r.user_id = ? AND r.is_active = 1 AND r.is_completed = 0
                  AND c.is_completed = 0
                ORDER BY r.id, c.display_order
            """, (user_id,))
            
            rows = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'resolution_id': row[1],
                    'text': row[2],
                    'resolution_text': row[3]
                }
                for row in rows
            ]
    
    def complete_checkpoint(self, checkpoint_id: int) -> Dict:
        """
        Mark a checkpoint as completed.
        
        Returns:
            Dict with milestone info (previous_pct, new_pct, resolution_id)
        """
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get the resolution_id for this checkpoint
            cursor.execute("""
                SELECT resolution_id FROM resolution_checkpoints WHERE id = ?
            """, (checkpoint_id,))
            row = cursor.fetchone()
            if not row:
                return {'error': 'Checkpoint not found'}
            
            resolution_id = row[0]
            
            # Get checkpoint progress before completion
            checkpoints_before = self.get_checkpoints(resolution_id)
            prev_progress = self._calculate_checkpoint_progress(checkpoints_before)
            
            # Mark checkpoint as completed
            cursor.execute("""
                UPDATE resolution_checkpoints
                SET is_completed = 1, completed_at = ?
                WHERE id = ?
            """, (now, checkpoint_id))
            conn.commit()
            
            # Get checkpoint progress after completion
            checkpoints_after = self.get_checkpoints(resolution_id)
            new_progress = self._calculate_checkpoint_progress(checkpoints_after)
            
            # Check for milestone
            milestone = self._check_milestone(prev_progress['percentage'], new_progress['percentage'])
            
            # Check if all checkpoints completed
            all_completed = new_progress['completed'] == new_progress['total'] and new_progress['total'] > 0
            
            return {
                'resolution_id': resolution_id,
                'previous_progress': prev_progress,
                'new_progress': new_progress,
                'milestone': milestone,
                'all_completed': all_completed
            }
    
    def delete_checkpoint(self, checkpoint_id: int) -> bool:
        """Delete a checkpoint."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM resolution_checkpoints WHERE id = ?", (checkpoint_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def _calculate_checkpoint_progress(self, checkpoints: List[Dict]) -> Dict:
        """Calculate checkpoint completion progress."""
        total = len(checkpoints)
        completed = sum(1 for c in checkpoints if c['is_completed'])
        percentage = (completed / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'completed': completed,
            'percentage': round(percentage, 1)
        }
    
    def _check_milestone(self, prev_pct: float, new_pct: float) -> Optional[int]:
        """Check if a milestone was crossed (25, 50, 75, 100)."""
        milestones = [25, 50, 75, 100]
        for m in milestones:
            if prev_pct < m <= new_pct:
                return m
        return None
    
    # ==================== Check-in Management ====================
    
    def record_check_in(
        self,
        resolution_id: int,
        status: str,
        notes: Optional[str] = None,
        channel_message_id: Optional[str] = None
    ) -> Dict:
        """
        Record a check-in and update streak.
        
        Args:
            resolution_id: Resolution ID
            status: Check-in status (on_track, struggling, skipped)
            notes: Optional notes from user
            channel_message_id: Message ID of the check-in prompt
            
        Returns:
            Dict with streak info
        """
        now = datetime.now()
        now_str = now.isoformat()
        today = now.date()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current resolution data including check day info
            cursor.execute("""
                SELECT current_streak, longest_streak, frequency,
                       check_day_of_week, check_day_of_month
                FROM resolutions
                WHERE id = ?
            """, (resolution_id,))
            row = cursor.fetchone()
            if not row:
                return {'error': 'Resolution not found'}
            
            current_streak, longest_streak, frequency, check_day_of_week, check_day_of_month = row
            
            # Record the check-in
            cursor.execute("""
                INSERT INTO resolution_check_ins (
                    resolution_id, status, notes, channel_message_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (resolution_id, status, notes, channel_message_id, now_str))
            
            # Update streak based on status
            if status in [CheckInStatus.ON_TRACK.value, CheckInStatus.STRUGGLING.value]:
                # Keep or increment streak
                new_streak = current_streak + 1
                new_longest = max(longest_streak, new_streak)
            else:
                # Skipped - reset streak
                new_streak = 0
                new_longest = longest_streak
            
            # Calculate next check date using stored check day
            next_check = self._calculate_next_check_date(
                frequency=frequency,
                check_day_of_week=check_day_of_week,
                check_day_of_month=check_day_of_month,
                from_date=today
            )
            
            # Update resolution
            cursor.execute("""
                UPDATE resolutions
                SET current_streak = ?, longest_streak = ?, 
                    next_check_date = ?, updated_at = ?
                WHERE id = ?
            """, (new_streak, new_longest, next_check, now_str, resolution_id))
            
            conn.commit()
            
            # Check for streak milestone
            streak_milestone = self._check_streak_milestone(current_streak, new_streak)
            
            return {
                'previous_streak': current_streak,
                'new_streak': new_streak,
                'longest_streak': new_longest,
                'streak_milestone': streak_milestone,
                'next_check_date': next_check
            }
    
    def _check_streak_milestone(self, prev: int, new: int) -> Optional[int]:
        """Check if a streak milestone was reached (4, 8, 12, 26, 52 weeks)."""
        milestones = [4, 8, 12, 26, 52]
        for m in milestones:
            if prev < m <= new:
                return m
        return None
    
    def get_check_in_history(
        self,
        resolution_id: int,
        limit: int = 10
    ) -> List[Dict]:
        """Get check-in history for a resolution."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, status, notes, created_at
                FROM resolution_check_ins
                WHERE resolution_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (resolution_id, limit))
            
            rows = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'status': row[1],
                    'notes': row[2],
                    'created_at': row[3]
                }
                for row in rows
            ]
    
    def get_user_check_in_stats(self, user_id: str) -> Dict:
        """Get aggregate check-in stats for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN ci.status = 'on_track' THEN 1 ELSE 0 END) as on_track,
                    SUM(CASE WHEN ci.status = 'struggling' THEN 1 ELSE 0 END) as struggling,
                    SUM(CASE WHEN ci.status = 'skipped' THEN 1 ELSE 0 END) as skipped
                FROM resolution_check_ins ci
                JOIN resolutions r ON ci.resolution_id = r.id
                WHERE r.user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            return {
                'total': row[0] or 0,
                'on_track': row[1] or 0,
                'struggling': row[2] or 0,
                'skipped': row[3] or 0
            }
    
    # ==================== Scheduled Check-in Queries ====================
    
    def get_due_check_ins(self) -> List[Dict]:
        """
        Get all resolutions due for check-in today.
        
        Returns:
            List of resolutions with user info, due for check-in
        """
        today = datetime.now().date().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.id, r.user_id, r.user_display_name, r.text, r.frequency,
                       r.current_streak, r.next_check_date
                FROM resolutions r
                WHERE r.is_active = 1 
                  AND r.is_completed = 0
                  AND date(r.next_check_date) <= date(?)
                ORDER BY r.user_id, r.id
            """, (today,))
            
            rows = cursor.fetchall()
            resolutions = []
            
            for row in rows:
                resolution = {
                    'id': row[0],
                    'user_id': row[1],
                    'user_display_name': row[2],
                    'text': row[3],
                    'frequency': row[4],
                    'current_streak': row[5],
                    'next_check_date': row[6]
                }
                
                # Get checkpoint progress
                checkpoints = self.get_checkpoints(row[0])
                resolution['checkpoints'] = checkpoints
                resolution['checkpoint_progress'] = self._calculate_checkpoint_progress(checkpoints)
                
                resolutions.append(resolution)
            
            return resolutions
    
    def get_pending_reminders(self, hours_threshold: int = 24) -> List[Dict]:
        """
        Get resolutions that need DM reminders (check-in sent but no response).
        
        Args:
            hours_threshold: Hours since check-in prompt was sent
            
        Returns:
            List of resolutions needing reminders
        """
        threshold_time = (datetime.now() - timedelta(hours=hours_threshold)).isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Find resolutions where:
            # - Check-in is overdue (next_check_date is in the past)
            # - No check-in recorded since the due date
            # - Reminder not yet sent
            cursor.execute("""
                SELECT r.id, r.user_id, r.user_display_name, r.text, r.frequency,
                       r.current_streak, r.next_check_date
                FROM resolutions r
                WHERE r.is_active = 1 
                  AND r.is_completed = 0
                  AND datetime(r.next_check_date) <= datetime(?)
                  AND NOT EXISTS (
                      SELECT 1 FROM resolution_check_ins ci
                      WHERE ci.resolution_id = r.id
                        AND datetime(ci.created_at) >= datetime(r.next_check_date)
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM resolution_check_ins ci
                      WHERE ci.resolution_id = r.id
                        AND ci.reminder_sent = 1
                        AND datetime(ci.created_at) >= datetime(r.next_check_date)
                  )
                ORDER BY r.user_id
            """, (threshold_time,))
            
            rows = cursor.fetchall()
            resolutions = []
            
            for row in rows:
                resolution = {
                    'id': row[0],
                    'user_id': row[1],
                    'user_display_name': row[2],
                    'text': row[3],
                    'frequency': row[4],
                    'current_streak': row[5],
                    'next_check_date': row[6]
                }
                
                checkpoints = self.get_checkpoints(row[0])
                resolution['checkpoints'] = checkpoints
                resolution['checkpoint_progress'] = self._calculate_checkpoint_progress(checkpoints)
                
                resolutions.append(resolution)
            
            return resolutions
    
    def mark_reminder_sent(self, resolution_id: int, dm_message_id: str) -> None:
        """Record that a DM reminder was sent for a resolution."""
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO resolution_check_ins (
                    resolution_id, status, reminder_sent, dm_message_id, created_at
                ) VALUES (?, 'pending', 1, ?, ?)
            """, (resolution_id, dm_message_id, now))
            conn.commit()
    
    def get_auto_skip_candidates(self, hours_threshold: int = 48) -> List[Dict]:
        """
        Get resolutions that should be auto-skipped (no response after threshold).
        
        Args:
            hours_threshold: Hours since reminder was sent
            
        Returns:
            List of resolutions to auto-skip
        """
        threshold_time = (datetime.now() - timedelta(hours=hours_threshold)).isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Find resolutions where:
            # - Reminder was sent (pending status with reminder_sent=1)
            # - No actual check-in response since reminder
            # - Reminder was sent more than threshold hours ago
            cursor.execute("""
                SELECT DISTINCT r.id, r.user_id, r.user_display_name, r.text,
                       r.current_streak, r.next_check_date
                FROM resolutions r
                JOIN resolution_check_ins ci ON ci.resolution_id = r.id
                WHERE r.is_active = 1 
                  AND r.is_completed = 0
                  AND ci.reminder_sent = 1
                  AND ci.status = 'pending'
                  AND datetime(ci.created_at) <= datetime(?)
                  AND NOT EXISTS (
                      SELECT 1 FROM resolution_check_ins ci2
                      WHERE ci2.resolution_id = r.id
                        AND ci2.status IN ('on_track', 'struggling', 'skipped')
                        AND datetime(ci2.created_at) > datetime(ci.created_at)
                  )
            """, (threshold_time,))
            
            rows = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'user_id': row[1],
                    'user_display_name': row[2],
                    'text': row[3],
                    'current_streak': row[4],
                    'next_check_date': row[5]
                }
                for row in rows
            ]
    
    def auto_skip_check_in(self, resolution_id: int) -> Dict:
        """
        Auto-skip a check-in for a resolution that didn't respond.
        
        Returns:
            Dict with updated streak info
        """
        return self.record_check_in(
            resolution_id=resolution_id,
            status=CheckInStatus.SKIPPED.value,
            notes="Auto-skipped due to no response"
        )
    
    def store_check_in_message_id(
        self,
        resolution_id: int,
        channel_message_id: str
    ) -> None:
        """Store the channel message ID for a check-in prompt."""
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO resolution_check_ins (
                    resolution_id, status, channel_message_id, created_at
                ) VALUES (?, 'pending', ?, ?)
            """, (resolution_id, channel_message_id, now))
            conn.commit()

