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
