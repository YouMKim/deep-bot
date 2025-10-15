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
    DISCORD_TOKEN: str = os.getenv('DISCORD_TOKEN', '')
    DISCORD_CLIENT_ID: str = os.getenv('DISCORD_CLIENT_ID', '')
    DISCORD_GUILD_ID: Optional[str] = os.getenv('DISCORD_GUILD_ID')
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    
    
    # Bot Configuration
    BOT_PREFIX: str = os.getenv('BOT_PREFIX', '!')
    BOT_OWNER_ID: Optional[str] = os.getenv('BOT_OWNER_ID')
    
    # Security Settings
    DEBUG_MODE: bool = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required environment variables are set."""
        required_vars = [
            'DISCORD_TOKEN',
            'OPENAI_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("Please check your .env file and ensure all required variables are set.")
            return False
        
        print("✅ All required environment variables are set.")
        return True
    
    @classmethod
    def get_discord_intents(cls):
        """Get Discord intents configuration."""
        intents = Intents.default()
        intents.message_content = True  # Required for message content access
        intents.members = True  # If you need member information
        intents.guilds = True   # If you need guild information
        
        return intents
