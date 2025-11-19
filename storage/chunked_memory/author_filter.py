"""
Author filtering service for chunked memory.

Handles filtering of documents based on author blacklist and whitelist.
"""
import logging
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config


class AuthorFilter:
    """Service for filtering documents based on author criteria."""

    def __init__(self, config: Optional['Config'] = None):
        """
        Initialize AuthorFilter.

        Args:
            config: Configuration instance (defaults to Config class)
        """
        from config import Config as ConfigClass
        
        self.config = config or ConfigClass
        self.logger = logging.getLogger(__name__)

    def should_include(
        self,
        author: str,
        exclude_blacklisted: bool,
        filter_authors: Optional[List[str]],
        author_id: Optional[str] = None
    ) -> bool:
        """
        Determine if a document should be included based on author.

        Checks:
        1. Blacklist filtering (if enabled) - uses author_id if provided
        2. Author whitelist filtering (if provided) - uses author name

        Args:
            author: Author name from document metadata
            exclude_blacklisted: Whether to filter out blacklisted authors
            filter_authors: Specific authors to include (None = include all)
            author_id: Author ID (Discord user ID) for blacklist checking

        Returns:
            True if document should be INCLUDED, False if should be FILTERED OUT
        """
        # Check blacklist - use author_id if available (more reliable)
        if exclude_blacklisted:
            # Try author_id first (Discord user ID)
            if author_id:
                try:
                    author_id_int = int(author_id)
                    if author_id_int in self.config.BLACKLIST_IDS:
                        self.logger.debug(f"Filtered out blacklisted author ID: {author_id} ({author})")
                        return False
                except (ValueError, TypeError):
                    pass
            
            # Fallback: check author name/ID as string
            if author in self.config.BLACKLIST_IDS or \
               str(author) in [str(bid) for bid in self.config.BLACKLIST_IDS]:
                self.logger.debug(f"Filtered out blacklisted author: {author}")
                return False

        # Check author whitelist
        if filter_authors:
            author_lower = author.lower()
            # Check if author matches any in the filter list (case-insensitive, partial match)
            matches = any(
                fa.lower() in author_lower or author_lower in fa.lower()
                for fa in filter_authors
            )
            if not matches:
                self.logger.debug(
                    f"Filtered out author {author} "
                    f"(not in whitelist: {filter_authors})"
                )
                return False

        return True

