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
        filter_authors: Optional[List[str]]
    ) -> bool:
        """
        Determine if a document should be included based on author.

        Checks:
        1. Blacklist filtering (if enabled)
        2. Author whitelist filtering (if provided)

        Args:
            author: Author name/ID from document metadata
            exclude_blacklisted: Whether to filter out blacklisted authors
            filter_authors: Specific authors to include (None = include all)

        Returns:
            True if document should be INCLUDED, False if should be FILTERED OUT
        """
        # Check blacklist
        if exclude_blacklisted:
            # Check both string and int representations
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

