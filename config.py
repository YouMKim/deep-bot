"""
Configuration management for the Discord bot.
Handles loading environment variables securely.
"""

import os
import logging
from discord import Intents
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)



class Config:
    """Configuration class to manage all bot settings."""

    # Discord Configuration
    DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
    DISCORD_CLIENT_ID: str = os.getenv("DISCORD_CLIENT_ID", "")
    DISCORD_GUILD_ID: Optional[str] = os.getenv("DISCORD_GUILD_ID")

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_DEFAULT_MODEL: str = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini-2025-08-07")
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_DEFAULT_MODEL: str = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-haiku-4-5")
    
    # Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_DEFAULT_MODEL: str = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")

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

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")  # "sentence-transformers" or "openai"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")  # Auto-selects: "all-MiniLM-L6-v2" for sentence-transformers, "text-embedding-3-small" for OpenAI
    
    # Embedding batch size (tune based on your embedder)
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))  # Process 100 documents at a time
    EMBEDDING_BATCH_DELAY: float = float(os.getenv("EMBEDDING_BATCH_DELAY", "0.1"))  # Seconds to wait between batches (rate limiting)

    #RAG configs 
    RAG_DEFAULT_TOP_K: int = int(os.getenv("RAG_DEFAULT_TOP_K", "8"))  # Reduced from 15 to reduce context size
    RAG_DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_DEFAULT_SIMILARITY_THRESHOLD", "0.01"))
    RAG_DEFAULT_MAX_CONTEXT_TOKENS: int = int(os.getenv("RAG_DEFAULT_MAX_CONTEXT_TOKENS", "4000"))
    RAG_FETCH_MULTIPLIER: int = int(os.getenv("RAG_FETCH_MULTIPLIER", "2"))  # Multiplier for reranking (reduced from 3)
    RAG_MULTI_QUERY_MULTIPLIER: float = float(os.getenv("RAG_MULTI_QUERY_MULTIPLIER", "1.5"))  # Multiplier for multi-query (reduced from 2)
    RAG_DEFAULT_TEMPERATURE: float = float(os.getenv("RAG_DEFAULT_TEMPERATURE", "0.7"))
    RAG_DEFAULT_STRATEGY: str = os.getenv("RAG_DEFAULT_STRATEGY", "author")
    
    # RAG Technique Settings (all enabled by default)
    RAG_USE_HYBRID_SEARCH: bool = os.getenv("RAG_USE_HYBRID_SEARCH", "True").lower() == "true"
    RAG_USE_MULTI_QUERY: bool = os.getenv("RAG_USE_MULTI_QUERY", "True").lower() == "true"
    RAG_USE_HYDE: bool = os.getenv("RAG_USE_HYDE", "True").lower() == "true"
    RAG_USE_RERANKING: bool = os.getenv("RAG_USE_RERANKING", "True").lower() == "true"
    RAG_MAX_OUTPUT_TOKENS: int = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "1200"))  # Increased to 1200 for more detailed responses

    # Chatbot Configuration
    CHATBOT_CHANNEL_ID: int = int(os.getenv("CHATBOT_CHANNEL_ID", "0"))
    CHATBOT_MAX_HISTORY: int = int(os.getenv("CHATBOT_MAX_HISTORY", "15"))
    CHATBOT_SESSION_TIMEOUT: int = int(os.getenv("CHATBOT_SESSION_TIMEOUT", "1800"))  # 30 minutes
    CHATBOT_MAX_TOKENS: int = int(os.getenv("CHATBOT_MAX_TOKENS", "400"))  # Default/fallback
    CHATBOT_CHAT_MAX_TOKENS: int = int(os.getenv("CHATBOT_CHAT_MAX_TOKENS", "250"))  # Conversational chat
    CHATBOT_RAG_MAX_TOKENS: int = int(os.getenv("CHATBOT_RAG_MAX_TOKENS", "1500"))  # RAG responses (can split if needed)
    CHATBOT_TEMPERATURE: float = float(os.getenv("CHATBOT_TEMPERATURE", "0.8"))
    CHATBOT_USE_RAG: bool = os.getenv("CHATBOT_USE_RAG", "True").lower() == "true"
    CHATBOT_RAG_THRESHOLD: float = float(os.getenv("CHATBOT_RAG_THRESHOLD", "0.01"))
    CHATBOT_RATE_LIMIT_MESSAGES: int = int(os.getenv("CHATBOT_RATE_LIMIT_MESSAGES", "10"))
    CHATBOT_RATE_LIMIT_WINDOW: int = int(os.getenv("CHATBOT_RATE_LIMIT_WINDOW", "60"))
    CHATBOT_INCLUDE_CONTEXT_MESSAGES: int = int(os.getenv("CHATBOT_INCLUDE_CONTEXT_MESSAGES", "5"))
    AI_DEFAULT_PROVIDER: str = os.getenv("AI_DEFAULT_PROVIDER", "openai")

    # Chatbot system prompt (defines personality)
    CHATBOT_SYSTEM_PROMPT: str = os.getenv(
        "CHATBOT_SYSTEM_PROMPT",
        """You are a helpful Discord chatbot assistant named Deep-Bot. 
You have access to past conversation history from this Discord server and can answer questions about what was discussed.
You can also answer questions about Deep-Bot's capabilities, commands, and how it works.

Guidelines:
- Be conversational, concise, and engaging (like a Discord message)
- When answering factual questions, rely on the provided context from past messages
- If you don't have enough context, say so politely
- Keep responses under 400 tokens (roughly 300 words)
- Use natural language, not bullet points or lists
- When asked about Deep-Bot itself, use information from the bot documentation if available
- Reference people by their Discord display names when mentioning past conversations
- Be friendly but not overly casual"""
    )
    
    # Social Credit Configuration
    SOCIAL_CREDIT_ENABLED: bool = os.getenv("SOCIAL_CREDIT_ENABLED", "True").lower() == "true"
    SOCIAL_CREDIT_INITIAL_MEAN: int = int(os.getenv("SOCIAL_CREDIT_INITIAL_MEAN", "0"))
    SOCIAL_CREDIT_INITIAL_STD: int = int(os.getenv("SOCIAL_CREDIT_INITIAL_STD", "200"))
    SOCIAL_CREDIT_PENALTY_ADMIN_COMMAND: int = int(os.getenv("SOCIAL_CREDIT_PENALTY_ADMIN_COMMAND", "-500"))
    SOCIAL_CREDIT_PENALTY_QUERY_FILTER: int = int(os.getenv("SOCIAL_CREDIT_PENALTY_QUERY_FILTER", "-500"))
    SOCIAL_CREDIT_DECAY_NEGATIVE: int = int(os.getenv("SOCIAL_CREDIT_DECAY_NEGATIVE", "-10"))
    SOCIAL_CREDIT_GROWTH_POSITIVE: int = int(os.getenv("SOCIAL_CREDIT_GROWTH_POSITIVE", "10"))

    # Evaluate Command Configuration
    EVALUATE_ENABLED: bool = os.getenv("EVALUATE_ENABLED", "True").lower() == "true"
    EVALUATE_MAX_TOKENS: int = int(os.getenv("EVALUATE_MAX_TOKENS", "800"))  # Reduced from 1500 for more concise responses
    EVALUATE_TEMPERATURE: float = float(os.getenv("EVALUATE_TEMPERATURE", "0.3"))
    EVALUATE_PROVIDER: Optional[str] = os.getenv("EVALUATE_PROVIDER")  # Defaults to AI_DEFAULT_PROVIDER if None

    # Snapshot Configuration
    SNAPSHOT_CHANNEL_ID: int = int(os.getenv("SNAPSHOT_CHANNEL_ID", "0"))  # Channel to post snapshots to

    # Year in Review Configuration
    YEAR_IN_REVIEW_ENABLED: bool = os.getenv("YEAR_IN_REVIEW_ENABLED", "True").lower() == "true"
    YEAR_IN_REVIEW_CHANNEL_ID: int = int(os.getenv("YEAR_IN_REVIEW_CHANNEL_ID", "0"))  # Channel to post reviews to (defaults to SNAPSHOT_CHANNEL_ID if 0)

    # Cronjob Configuration
    CRONJOB_SCHEDULE_TIME: str = os.getenv("CRONJOB_SCHEDULE_TIME", "14:00")  # Time to run cronjob daily (HH:MM format, UTC). Default 14:00 UTC = 6 AM Pacific
    CRONJOB_ENABLED: bool = os.getenv("CRONJOB_ENABLED", "True").lower() == "true"  # Enable/disable cronjob scheduler

    # Resolution Tracking Configuration
    RESOLUTION_ENABLED: bool = os.getenv("RESOLUTION_ENABLED", "True").lower() == "true"  # Enable/disable resolution tracking
    RESOLUTION_CHANNEL_ID: int = int(os.getenv("RESOLUTION_CHANNEL_ID", "0"))  # Channel to post check-in prompts
    RESOLUTION_REMINDER_HOURS: int = int(os.getenv("RESOLUTION_REMINDER_HOURS", "24"))  # Hours before sending DM reminder
    RESOLUTION_AUTO_SKIP_HOURS: int = int(os.getenv("RESOLUTION_AUTO_SKIP_HOURS", "48"))  # Hours before auto-skipping check-in

    @classmethod
    def load_blacklist(cls):
        """Load blacklisted user IDs from environment variable."""
        import logging
        logger = logging.getLogger(__name__)
        
        blacklist_str = os.getenv("BLACKLIST_IDS", "")
        if not blacklist_str:
            logger.info("No blacklisted user IDs configured")
            cls.BLACKLIST_IDS = []
            return
        
        try:
            ids = []
            for id_str in blacklist_str.split(","):
                id_str = id_str.strip()
                if id_str:
                    user_id = int(id_str)
                    if user_id <= 0:
                        logger.warning(f"Invalid blacklist ID (must be positive): {user_id}")
                        continue
                    ids.append(user_id)
            
            cls.BLACKLIST_IDS = ids
            logger.info(f"Loaded {len(cls.BLACKLIST_IDS)} blacklisted user IDs")
        except ValueError as e:
            logger.error(f"Error parsing BLACKLIST_IDS: {e}")
            print(f"❌ Error parsing BLACKLIST_IDS: {e}")
            raise ValueError(f"Invalid BLACKLIST_IDS format. Expected comma-separated integers. Error: {e}")
    
    @classmethod
    def is_blacklisted(cls, user_id: int) -> bool:
        """Check if a user ID is blacklisted."""
        return user_id in cls.BLACKLIST_IDS

    @classmethod
    def validate(cls) -> bool:
        """Validate that all required environment variables are set."""
        import logging
        logger = logging.getLogger(__name__)
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

        logger.debug("All required environment variables are set.")
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
        cls.RAG_USE_HYBRID_SEARCH = os.getenv("RAG_USE_HYBRID_SEARCH", "True").lower() == "true"
        cls.RAG_USE_MULTI_QUERY = os.getenv("RAG_USE_MULTI_QUERY", "True").lower() == "true"
        cls.RAG_USE_HYDE = os.getenv("RAG_USE_HYDE", "True").lower() == "true"
        cls.RAG_USE_RERANKING = os.getenv("RAG_USE_RERANKING", "True").lower() == "true"
        cls.RAG_MAX_OUTPUT_TOKENS = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "1200"))
        cls.RAG_DEFAULT_STRATEGY = os.getenv("RAG_DEFAULT_STRATEGY", "author")
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Reset all RAG settings to .env defaults")
    
    @classmethod
    def validate_chatbot_config(cls) -> bool:
        """
        Validate chatbot configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)
        
        errors = []
        
        # Validate channel ID - must be explicitly configured
        if not cls.CHATBOT_CHANNEL_ID or cls.CHATBOT_CHANNEL_ID <= 0:
            errors.append("CHATBOT_CHANNEL_ID must be configured in .env file (must be a positive integer)")
        
        # Validate max history
        if cls.CHATBOT_MAX_HISTORY < 1:
            errors.append("CHATBOT_MAX_HISTORY must be >= 1")
        
        # Validate session timeout
        if cls.CHATBOT_SESSION_TIMEOUT < 60:
            errors.append("CHATBOT_SESSION_TIMEOUT must be >= 60 seconds")
        
        # Validate max tokens
        if cls.CHATBOT_MAX_TOKENS < 50 or cls.CHATBOT_MAX_TOKENS > 2000:
            errors.append("CHATBOT_MAX_TOKENS must be between 50 and 2000")
        if cls.CHATBOT_CHAT_MAX_TOKENS < 50 or cls.CHATBOT_CHAT_MAX_TOKENS > 1000:
            errors.append("CHATBOT_CHAT_MAX_TOKENS must be between 50 and 1000")
        # RAG max tokens can be higher since we support splitting long responses
        if cls.CHATBOT_RAG_MAX_TOKENS < 100 or cls.CHATBOT_RAG_MAX_TOKENS > 5000:
            errors.append("CHATBOT_RAG_MAX_TOKENS must be between 100 and 5000 (responses will be split if needed)")
        
        # Validate temperature
        if not 0.0 <= cls.CHATBOT_TEMPERATURE <= 2.0:
            errors.append("CHATBOT_TEMPERATURE must be between 0.0 and 2.0")
        
        # Validate RAG threshold
        if not 0.0 <= cls.CHATBOT_RAG_THRESHOLD <= 1.0:
            errors.append("CHATBOT_RAG_THRESHOLD must be between 0.0 and 1.0")
        
        # Validate rate limit
        if cls.CHATBOT_RATE_LIMIT_MESSAGES < 1:
            errors.append("CHATBOT_RATE_LIMIT_MESSAGES must be >= 1")
        
        if cls.CHATBOT_RATE_LIMIT_WINDOW < 1:
            errors.append("CHATBOT_RATE_LIMIT_WINDOW must be >= 1")
        
        # Validate context messages
        if cls.CHATBOT_INCLUDE_CONTEXT_MESSAGES < 0:
            errors.append("CHATBOT_INCLUDE_CONTEXT_MESSAGES must be >= 0")
        
        # Validate AI provider
        valid_providers = ["openai", "anthropic", "gemini"]
        if cls.AI_DEFAULT_PROVIDER not in valid_providers:
            errors.append(f"AI_DEFAULT_PROVIDER must be one of: {', '.join(valid_providers)}")
        
        if errors:
            for error in errors:
                logger.error(f"Chatbot config validation error: {error}")
            return False
        
        logger.info("Chatbot configuration validated successfully")
        return True
    
    @classmethod
    def create_rag_config(cls, **overrides) -> 'RAGConfig':
        """
        Create RAGConfig with defaults from environment variables.
        
        This provides a single source of truth for RAG configuration creation,
        eliminating duplication between chatbot.py and rag.py.
        
        Args:
            **overrides: Any config values to override defaults
            
        Returns:
            RAGConfig instance with defaults applied
        """
        from rag.models import RAGConfig
        
        config_dict = {
            'top_k': cls.RAG_DEFAULT_TOP_K,
            'similarity_threshold': cls.RAG_DEFAULT_SIMILARITY_THRESHOLD,
            'max_context_tokens': cls.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
            'temperature': cls.RAG_DEFAULT_TEMPERATURE,
            'strategy': cls.RAG_DEFAULT_STRATEGY,
            'use_hybrid_search': cls.RAG_USE_HYBRID_SEARCH,
            'use_multi_query': cls.RAG_USE_MULTI_QUERY,
            'use_hyde': cls.RAG_USE_HYDE,
            'use_reranking': cls.RAG_USE_RERANKING,
            'max_output_tokens': cls.RAG_MAX_OUTPUT_TOKENS,
        }
        
        # Apply overrides
        config_dict.update(overrides)
        
        return RAGConfig(**config_dict)
