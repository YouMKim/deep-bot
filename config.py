"""
Configuration management for the Discord bot.
Handles loading environment variables securely.
"""

import os
from discord import Intents
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()



class Config:
    """Configuration class to manage all bot settings."""

    # Discord Configuration
    DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
    DISCORD_CLIENT_ID: str = os.getenv("DISCORD_CLIENT_ID", "")
    DISCORD_GUILD_ID: Optional[str] = os.getenv("DISCORD_GUILD_ID")

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Bot Configuration
    BOT_PREFIX: str = os.getenv("BOT_PREFIX", "!")
    BOT_OWNER_ID: Optional[str] = os.getenv("BOT_OWNER_ID")

    # Security Settings
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    TEST_MODE: bool = os.getenv("TEST_MODE", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Blacklist from AI usage
    BLACKLIST_IDS: list = []

    #Discord Message rate limits
    MESSAGE_FETCH_DELAY: float = float(os.getenv("MESSAGE_FETCH_DELAY", "0.2"))
    MESSAGE_FETCH_BATCH_SIZE: int = int(os.getenv("MESSAGE_FETCH_BATCH_SIZE", "100"))
    MESSAGE_FETCH_PROGRESS_INTERVAL: int = int(os.getenv("MESSAGE_FETCH_PROGRESS_INTERVAL", "100"))
    MESSAGE_FETCH_MAX_RETRIES: int = int(os.getenv("MESSAGE_FETCH_MAX_RETRIES", "5"))
    
    #Chunking Configuration
    CHUNKING_TEMPORAL_WINDOW: int = int(os.getenv("CHUNKING_TEMPORAL_WINDOW", "3600"))  # 1 hour in seconds
    CHUNKING_CONVERSATION_GAP: int = int(os.getenv("CHUNKING_CONVERSATION_GAP", "1800"))  # 30 minutes
    CHUNKING_WINDOW_SIZE: int = int(os.getenv("CHUNKING_WINDOW_SIZE", "10"))  # messages per window
    CHUNKING_OVERLAP: int = int(os.getenv("CHUNKING_OVERLAP", "2"))  # overlapping messages
    CHUNKING_MAX_TOKENS: int = int(os.getenv("CHUNKING_MAX_TOKENS", "512"))  # max tokens per chunk
    CHUNKING_MIN_CHUNK_SIZE: int = int(os.getenv("CHUNKING_MIN_CHUNK_SIZE", "3"))  # min messages per chunk
    
    # Default chunking strategies to use (comma-separated)
    # Options: single, temporal, conversation, sliding_window, author, tokens
    # Default: single,tokens,author (covers most use cases: fast, token-aware, and author-grouped)
    CHUNKING_DEFAULT_STRATEGIES: str = os.getenv("CHUNKING_DEFAULT_STRATEGIES", "single,tokens,author")

    # Vector Store Configuration
    VECTOR_STORE_PROVIDER: str = os.getenv("VECTOR_STORE_PROVIDER", "chroma")  # chroma, pinecone, etc.

    # Embedding batch size (tune based on your embedder)
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))  # Process 100 documents at a time
    EMBEDDING_BATCH_DELAY: float = float(os.getenv("EMBEDDING_BATCH_DELAY", "0.1"))  # Seconds to wait between batches (rate limiting)

    #RAG configs 
    RAG_DEFAULT_TOP_K: int = int(os.getenv("RAG_DEFAULT_TOP_K", "10"))
    RAG_DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_DEFAULT_SIMILARITY_THRESHOLD", "0.35"))
    RAG_DEFAULT_MAX_CONTEXT_TOKENS: int = int(os.getenv("RAG_DEFAULT_MAX_CONTEXT_TOKENS", "4000"))
    RAG_DEFAULT_TEMPERATURE: float = float(os.getenv("RAG_DEFAULT_TEMPERATURE", "0.7"))
    RAG_DEFAULT_STRATEGY: str = os.getenv("RAG_DEFAULT_STRATEGY", "tokens")
    
    # RAG Technique Settings
    RAG_USE_HYBRID_SEARCH: bool = os.getenv("RAG_USE_HYBRID_SEARCH", "False").lower() == "true"
    RAG_USE_MULTI_QUERY: bool = os.getenv("RAG_USE_MULTI_QUERY", "False").lower() == "true"
    RAG_USE_HYDE: bool = os.getenv("RAG_USE_HYDE", "False").lower() == "true"
    RAG_USE_RERANKING: bool = os.getenv("RAG_USE_RERANKING", "False").lower() == "true"
    RAG_MAX_OUTPUT_TOKENS: int = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "1000"))

    @classmethod
    def load_blacklist(cls):
        """Load blacklisted user IDs from environment variable."""
        import logging
        logger = logging.getLogger(__name__)
        
        blacklist_str = os.getenv("BLACKLIST_IDS", "")
        if blacklist_str:
            # Split by comma and convert to integers
            try:
                cls.BLACKLIST_IDS = [int(id_str.strip()) for id_str in blacklist_str.split(",") if id_str.strip()]
                print(f"✅ Loaded {len(cls.BLACKLIST_IDS)} blacklisted user IDs")
            except ValueError as e:
                print(f"❌ Error parsing BLACKLIST_IDS: {e}")
                cls.BLACKLIST_IDS = []
        else:
            print("No blacklisted user IDs configured")
    
    @classmethod
    def is_blacklisted(cls, user_id: int) -> bool:
        """Check if a user ID is blacklisted."""
        return user_id in cls.BLACKLIST_IDS

    @classmethod
    def validate(cls) -> bool:
        """Validate that all required environment variables are set."""
        required_vars = ["DISCORD_TOKEN", "OPENAI_API_KEY"]

        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)

        if missing_vars:
            print(
                f"❌ Missing required environment variables: {', '.join(missing_vars)}"
            )
            print(
                "Please check your .env file and ensure all required variables are set."
            )
            return False

        print("All required environment variables are set.")
        return True

    @classmethod
    def get_discord_intents(cls):
        """Get Discord intents configuration."""
        intents = Intents.default()
        intents.message_content = True  # Required for message content access
        intents.members = True  # If you need member information
        intents.guilds = True  # If you need guild information

        return intents
    
    @classmethod
    def update_rag_setting(cls, key: str, value) -> None:
        """
        Update a RAG setting in Config (in-memory only).
        Settings reset to .env defaults on bot restart.
        
        Args:
            key: Setting name (e.g., 'RAG_USE_HYBRID_SEARCH', 'RAG_DEFAULT_STRATEGY')
            value: New value (bool for flags, int for tokens, str for strategy)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Validate key
        valid_keys = [
            'RAG_USE_HYBRID_SEARCH', 'RAG_USE_MULTI_QUERY', 'RAG_USE_HYDE',
            'RAG_USE_RERANKING', 'RAG_MAX_OUTPUT_TOKENS', 'RAG_DEFAULT_STRATEGY'
        ]
        if key not in valid_keys:
            raise ValueError(f"Invalid RAG setting key: {key}. Valid keys: {valid_keys}")
        
        # Convert value to proper type
        if key == 'RAG_MAX_OUTPUT_TOKENS':
            typed_value = int(value)
        elif key == 'RAG_DEFAULT_STRATEGY':
            # Validate strategy name
            from chunking.constants import ChunkStrategy
            try:
                ChunkStrategy(str(value).lower())
                typed_value = str(value).lower()
            except ValueError:
                valid_strategies = ", ".join([s.value for s in ChunkStrategy])
                raise ValueError(f"Invalid strategy '{value}'. Valid strategies: {valid_strategies}")
        else:
            typed_value = bool(value) if isinstance(value, str) else value
        
        # Update Config attribute (in-memory only)
        setattr(cls, key, typed_value)
        
        logger.info(f"Updated {key} = {typed_value} (in-memory, resets to .env defaults on restart)")
    
    @classmethod
    def reset_rag_settings(cls) -> None:
        """Reset all RAG settings to .env defaults (in-memory only)."""
        # Reload from environment variables
        cls.RAG_USE_HYBRID_SEARCH = os.getenv("RAG_USE_HYBRID_SEARCH", "False").lower() == "true"
        cls.RAG_USE_MULTI_QUERY = os.getenv("RAG_USE_MULTI_QUERY", "False").lower() == "true"
        cls.RAG_USE_HYDE = os.getenv("RAG_USE_HYDE", "False").lower() == "true"
        cls.RAG_USE_RERANKING = os.getenv("RAG_USE_RERANKING", "False").lower() == "true"
        cls.RAG_MAX_OUTPUT_TOKENS = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "1000"))
        cls.RAG_DEFAULT_STRATEGY = os.getenv("RAG_DEFAULT_STRATEGY", "tokens")
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Reset all RAG settings to .env defaults")
