import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QueryValidator:
    """Validates and sanitizes user queries."""

    # Configuration
    MAX_QUERY_LENGTH = 1000  # characters
    MIN_QUERY_LENGTH = 3     # characters

    # Patterns that suggest prompt injection
    INJECTION_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"ignore\s+above",
        r"disregard\s+previous",
        r"you\s+are\s+now",
        r"new\s+instructions",
        r"system\s+prompt",
        r"<\s*system\s*>",  # System tags
        r"<\s*\|.*?\|\s*>",  # Special tokens
    ]

    @classmethod
    def validate(cls, query: str) -> str:
        """
        Validate and sanitize a user query.

        Args:
            query: Raw user input

        Returns:
            Sanitized query string

        Raises:
            ValueError: If query fails validation
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be string, got {type(query)}")

        # Strip whitespace
        query = query.strip()

        # Check minimum length
        if len(query) < cls.MIN_QUERY_LENGTH:
            raise ValueError(
                f"Query too short (minimum {cls.MIN_QUERY_LENGTH} characters)"
            )

        # Check maximum length
        if len(query) > cls.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long (maximum {cls.MAX_QUERY_LENGTH} characters). "
                f"Please shorten your question."
            )

        # Check for prompt injection attempts
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(
                    f"Potential prompt injection detected: {query[:100]}... "
                    f"(matched pattern: {pattern})"
                )
                # Option 1: Reject the query
                raise ValueError(
                    "Query contains suspicious content. "
                    "Please rephrase your question."
                )

                # Option 2: Sanitize by removing the pattern
                # query = re.sub(pattern, '', query, flags=re.IGNORECASE)

        # Basic sanitization - remove excessive whitespace
        query = re.sub(r'\s+', ' ', query)

        # Remove null bytes (can cause issues)
        query = query.replace('\x00', '')

        return query

    @classmethod
    def is_safe(cls, query: str) -> bool:
        """
        Check if query is safe without raising exceptions.

        Returns:
            True if safe, False otherwise
        """
        try:
            cls.validate(query)
            return True
        except ValueError:
            return False

