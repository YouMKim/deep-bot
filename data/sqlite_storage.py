import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

class SQLiteStorage:

    def __init__(self, db_path: str):
        self.db_path = db_path 
        self.logger = logging.getLogger(__name__)
        self._ensure_database_directory_exists()

    def _ensure_database_directory_exists(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close() 
    

    