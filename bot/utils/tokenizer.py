"""
Shared tokenizer utility for consistent token counting across the application.

Provides a singleton tiktoken encoder to avoid multiple initializations.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global singleton tokenizer
_tokenizer = None
_tokenizer_initialized = False

# Fallback estimation constant (chars per token)
CHARS_PER_TOKEN_ESTIMATE = 4


def get_tokenizer():
    """
    Get the shared tiktoken encoder instance.
    
    Returns a singleton tiktoken encoder, initializing it on first call.
    If tiktoken is not available, returns None.
    
    Returns:
        tiktoken.Encoding or None if not available
    """
    global _tokenizer, _tokenizer_initialized
    
    if _tokenizer_initialized:
        return _tokenizer
    
    try:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.debug("Initialized shared tiktoken encoder")
    except ImportError:
        logger.warning(
            "tiktoken not available, token counting will use fallback estimation. "
            "Install tiktoken for accurate counts: pip install tiktoken"
        )
        _tokenizer = None
    except Exception as e:
        logger.warning(f"Failed to initialize tiktoken: {e}")
        _tokenizer = None
    
    _tokenizer_initialized = True
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken if available, fallback to estimation.
    
    Uses the cl100k_base encoding (used by GPT-4, GPT-3.5-turbo).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens (estimated if tiktoken not available)
    """
    tokenizer = get_tokenizer()
    
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using fallback")
    
    # Fallback: rough estimation (~4 characters per token)
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def count_tokens_batch(texts: list) -> list:
    """
    Count tokens for multiple texts efficiently.
    
    Args:
        texts: List of texts to count tokens for
        
    Returns:
        List of token counts
    """
    return [count_tokens(text) for text in texts]


def estimate_tokens_from_chars(char_count: int) -> int:
    """
    Estimate token count from character count.
    
    Useful for quick estimations without processing the actual text.
    
    Args:
        char_count: Number of characters
        
    Returns:
        Estimated token count
    """
    return char_count // CHARS_PER_TOKEN_ESTIMATE


def estimate_chars_from_tokens(token_count: int) -> int:
    """
    Estimate character count from token count.
    
    Useful for setting character limits based on token budgets.
    
    Args:
        token_count: Number of tokens
        
    Returns:
        Estimated character count
    """
    return token_count * CHARS_PER_TOKEN_ESTIMATE

