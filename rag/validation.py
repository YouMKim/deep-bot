import re
import logging
from typing import Optional, Tuple

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
    async def validate(
        cls, 
        query: str,
        user_id: Optional[str] = None,
        user_display_name: Optional[str] = None,
        social_credit_manager=None
    ) -> str:
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
                
                # Apply penalty if social credit manager is available
                if user_id and social_credit_manager:
                    try:
                        new_score = await social_credit_manager.apply_penalty(
                            user_id,
                            "query_filter_violation",
                            user_display_name
                        )
                        error_msg = (
                            f"Query contains prohibited content. "
                            f"-150 SOCIAL CREDIT. New score: {new_score}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to apply penalty: {e}", exc_info=True)
                        error_msg = "Query contains prohibited content. Please rephrase your question."
                else:
                    error_msg = "Query contains prohibited content. Please rephrase your question."
                
                raise ValueError(error_msg)

        # Basic sanitization - remove excessive whitespace
        query = re.sub(r'\s+', ' ', query)

        # Remove null bytes (can cause issues)
        query = query.replace('\x00', '')

        return query

    @classmethod
    async def is_safe(
        cls, 
        query: str,
        user_id: Optional[str] = None,
        user_display_name: Optional[str] = None,
        social_credit_manager=None
    ) -> bool:
        """
        Check if query is safe without raising exceptions.

        Returns:
            True if safe, False otherwise
        """
        try:
            await cls.validate(query, user_id, user_display_name, social_credit_manager)
            return True
        except ValueError:
            return False

