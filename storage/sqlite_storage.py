"""
SQLite storage base class.

This is a minimal base class that provides SQLite connection management
for subclasses like MessageStorage and UserAITracker. It's intentionally
minimal - subclasses implement the actual storage methods.
"""
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

class SQLiteStorage:
    """
    Base class for SQLite-based storage implementations.
    
    Provides connection management via context manager. Subclasses should
    implement their own storage methods (save, retrieve, etc.) using
    the _get_connection() context manager.
    
    Example:
        class MyStorage(SQLiteStorage):
            def save_data(self, data):
                with self._get_connection() as conn:
                    conn.execute("INSERT INTO ...", data)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path 
        self.logger = logging.getLogger(__name__)
        self._ensure_database_directory_exists()

    def _ensure_database_directory_exists(self):
        """Ensure the database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """
        Get a SQLite database connection as a context manager.
        
        Usage:
            with self._get_connection() as conn:
                conn.execute("SELECT ...")
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    

