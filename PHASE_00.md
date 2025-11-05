# Phase 0: Development Environment Setup

## Overview

Before diving into building your Discord RAG chatbot, let's set up a proper development environment. This phase ensures everyone starts with the same setup, reducing "it works on my machine" issues and making debugging easier.

**Learning Objectives:**
- Set up Python virtual environment
- Install and configure required dependencies
- Create and configure a Discord bot
- Set up environment variables securely
- Verify your development environment works
- Learn Git basics for version control

**Prerequisites:**
- Computer with internet connection
- Basic command line knowledge
- Enthusiasm to learn! 🚀

**Estimated Time:** 30-60 minutes

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Python Setup](#2-python-setup)
3. [Project Setup](#3-project-setup)
4. [Discord Bot Setup](#4-discord-bot-setup)
5. [Environment Configuration](#5-environment-configuration)
6. [Verify Installation](#6-verify-installation)
7. [Optional: Local LLM Setup](#7-optional-local-llm-setup)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. System Requirements

### Minimum Requirements
- **Operating System:** Windows 10+, macOS 10.15+, or Linux
- **Python:** 3.11 or higher (3.12 recommended)
- **RAM:** 4GB minimum, 8GB+ recommended
- **Storage:** 5GB free space (more if using local LLMs)
- **Internet:** Required for Discord API and cloud services

### Recommended Tools
- **Code Editor:** VS Code (with Python extension)
- **Terminal:** Built-in terminal or Windows Terminal
- **Git:** For version control
- **Discord Account:** For bot development and testing

---

## 2. Python Setup

### Check Python Version

```bash
# Check if Python is installed
python --version
# or
python3 --version

# Should show: Python 3.11.x or higher
```

### Install Python (if needed)

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer, **CHECK "Add Python to PATH"**
3. Verify: `python --version`

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.12

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

### Create Virtual Environment

Virtual environments keep your project dependencies isolated:

```bash
# Navigate to your project directory
cd /path/to/deep-bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# You should see (venv) in your prompt
```

**Why Virtual Environments?**
- ✅ Isolate project dependencies
- ✅ Avoid version conflicts
- ✅ Easy to recreate on other machines
- ✅ Keep global Python clean

---

## 3. Project Setup

### Clone or Create Project

**Option A: Clone from GitHub**
```bash
git clone https://github.com/yourusername/deep-bot.git
cd deep-bot
```

**Option B: Create Fresh**
```bash
mkdir deep-bot
cd deep-bot
git init
```

### Create Project Structure

```bash
# Create directory structure
mkdir -p data/{raw_messages,chroma_db,security}
mkdir -p src/{services,cogs,utils}
mkdir -p tests/{unit,integration}
mkdir -p examples
mkdir -p docs

# Create initial files
touch src/__init__.py
touch src/bot.py
touch src/config.py
touch README.md
touch .gitignore
touch .env.example
```

**Project Structure:**
```
deep-bot/
├── data/                  # Data storage (gitignored)
│   ├── raw_messages/      # SQLite databases
│   ├── chroma_db/         # Vector embeddings
│   └── security/          # Security logs
├── src/                   # Source code
│   ├── services/          # Business logic
│   ├── cogs/              # Discord command groups
│   ├── utils/             # Helper functions
│   ├── bot.py             # Main bot file
│   └── config.py          # Configuration
├── tests/                 # Tests
├── examples/              # Learning examples
├── docs/                  # Documentation
├── .env                   # Environment variables (gitignored)
├── .env.example           # Template for .env
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

### Create .gitignore

Create `.gitignore` file:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Environment variables
.env
.env.local

# Data directories
data/
*.db
*.db-journal

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# ChromaDB
chroma_db/
```

### Install Core Dependencies

Create `requirements.txt`:

```txt
# Discord
discord.py==2.3.2

# Database
aiosqlite==0.19.0

# Embeddings
sentence-transformers==2.2.2
openai==1.12.0

# Vector Store
chromadb==0.4.22

# Utilities
python-dotenv==1.0.0
tiktoken==0.5.2
rank-bm25==0.2.2

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
```

Install dependencies:

```bash
# Make sure virtual environment is activated
pip install --upgrade pip
pip install -r requirements.txt

# This might take 5-10 minutes (downloading models)
```

### Create Minimal Requirements (for quick start)

Create `requirements-minimal.txt`:

```txt
# Minimal dependencies for quick start
discord.py==2.3.2
sentence-transformers==2.2.2
chromadb==0.4.22
python-dotenv==1.0.0
tiktoken==0.5.2
```

---

## 4. Discord Bot Setup

### Create Discord Application

1. **Go to Discord Developer Portal**
   - Visit: https://discord.com/developers/applications
   - Click "New Application"
   - Name it: "My RAG Bot" (or your choice)
   - Click "Create"

2. **Create Bot User**
   - Click "Bot" in left sidebar
   - Click "Add Bot" → "Yes, do it!"
   - Under "Token", click "Copy" (save this!)
   - **IMPORTANT:** Never share your bot token publicly!

3. **Configure Bot Settings**
   - Under "Privileged Gateway Intents", enable:
     - ✅ Presence Intent
     - ✅ Server Members Intent
     - ✅ Message Content Intent (CRITICAL!)
   - Click "Save Changes"

4. **Invite Bot to Test Server**
   - Click "OAuth2" → "URL Generator"
   - Under "Scopes", check:
     - ✅ bot
     - ✅ applications.commands
   - Under "Bot Permissions", check:
     - ✅ Read Messages/View Channels
     - ✅ Send Messages
     - ✅ Read Message History
     - ✅ Add Reactions
   - Copy the generated URL
   - Open URL in browser → Select your test server → "Authorize"

### Create Test Server (if needed)

1. Open Discord
2. Click "+" on server list
3. "Create My Own" → "For me and my friends"
4. Name it "RAG Bot Testing"
5. Create a channel: "#bot-testing"

---

## 5. Environment Configuration

### Create .env.example Template

Create `.env.example`:

```bash
# Discord Configuration
DISCORD_TOKEN=your_discord_bot_token_here
BOT_PREFIX=!
BOT_OWNER_ID=your_discord_user_id

# OpenAI (Optional - for cloud embeddings/LLM)
OPENAI_API_KEY=your_openai_api_key_here

# Embedding Configuration
EMBEDDING_PROVIDER=sentence-transformers  # or "openai"
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# LLM Configuration
LLM_PROVIDER=local  # or "openai"
LLM_MODEL=gpt-3.5-turbo  # if using OpenAI

# Vector Store
VECTOR_STORE_PROVIDER=chroma
CHROMA_PERSIST_DIRECTORY=data/chroma_db

# Database
RAW_MESSAGES_DIR=data/raw_messages

# Rate Limiting
MESSAGE_FETCH_DELAY=1.0
MESSAGE_FETCH_BATCH_SIZE=100

# Security
SECURITY_ENABLED=True
RATE_LIMIT_QUERIES_PER_MINUTE=5
RATE_LIMIT_QUERIES_PER_HOUR=50

# Features
SUMMARY_USE_STORED_MESSAGES=True
```

### Create Your .env File

```bash
# Copy template
cp .env.example .env

# Edit .env with your values
# On macOS/Linux:
nano .env

# On Windows:
notepad .env
```

**Fill in:**
1. `DISCORD_TOKEN`: Paste your bot token
2. `BOT_OWNER_ID`: Your Discord user ID
   - Enable Developer Mode: Settings → Advanced → Developer Mode
   - Right-click your name → Copy ID
3. `OPENAI_API_KEY`: (Optional) If using OpenAI

**Example .env:**
```bash
DISCORD_TOKEN=MTIzNDU2Nzg5MDEyMzQ1Njc4OQ.GaBcDe.FgHiJkLmNoPqRsTuVwXyZ123456789
BOT_PREFIX=!
BOT_OWNER_ID=123456789012345678
EMBEDDING_PROVIDER=sentence-transformers
```

### Create config.py

Create `src/config.py`:

```python
"""
Configuration management for the bot.
Loads environment variables and provides typed config.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load .env file
load_dotenv()

class Config:
    """Central configuration class."""

    # Discord
    DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
    BOT_PREFIX: str = os.getenv("BOT_PREFIX", "!")
    BOT_OWNER_ID: Optional[str] = os.getenv("BOT_OWNER_ID")

    # OpenAI (Optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Embedding
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    # Vector Store
    VECTOR_STORE_PROVIDER: str = os.getenv("VECTOR_STORE_PROVIDER", "chroma")
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")

    # Database
    RAW_MESSAGES_DIR: str = os.getenv("RAW_MESSAGES_DIR", "data/raw_messages")

    # Rate Limiting
    MESSAGE_FETCH_DELAY: float = float(os.getenv("MESSAGE_FETCH_DELAY", "1.0"))
    MESSAGE_FETCH_BATCH_SIZE: int = int(os.getenv("MESSAGE_FETCH_BATCH_SIZE", "100"))

    # Security
    SECURITY_ENABLED: bool = os.getenv("SECURITY_ENABLED", "True").lower() == "true"
    RATE_LIMIT_QUERIES_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_QUERIES_PER_MINUTE", "5"))
    RATE_LIMIT_QUERIES_PER_HOUR: int = int(os.getenv("RATE_LIMIT_QUERIES_PER_HOUR", "50"))

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        if not cls.DISCORD_TOKEN:
            print("❌ ERROR: DISCORD_TOKEN not set in .env file")
            return False

        if not cls.BOT_OWNER_ID:
            print("⚠️  WARNING: BOT_OWNER_ID not set (some commands won't work)")

        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            print("⚠️  WARNING: OPENAI_API_KEY not set but OpenAI embeddings selected")

        if cls.MESSAGE_FETCH_DELAY < 0.5:
            print("⚠️  WARNING: MESSAGE_FETCH_DELAY very low, may get rate limited")

        return True


# Validate on import
if __name__ != "__main__":
    Config.validate()
```

---

## 6. Verify Installation

### Create Test Bot

Create `src/bot.py`:

```python
"""
Simple test bot to verify setup.
"""

import discord
from discord.ext import commands
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config

# Create bot instance
intents = discord.Intents.default()
intents.message_content = True  # Required for reading messages
intents.members = True

bot = commands.Bot(
    command_prefix=Config.BOT_PREFIX,
    intents=intents,
    help_command=None  # We'll create custom help later
)


@bot.event
async def on_ready():
    """Called when bot successfully connects."""
    print(f"✅ Bot is ready!")
    print(f"   Logged in as: {bot.user.name}")
    print(f"   Bot ID: {bot.user.id}")
    print(f"   Prefix: {Config.BOT_PREFIX}")
    print(f"   Servers: {len(bot.guilds)}")
    print("---")
    print("Try: !ping in Discord")


@bot.command(name="ping")
async def ping(ctx):
    """Test command to verify bot responds."""
    latency_ms = round(bot.latency * 1000)
    await ctx.send(f"🏓 Pong! Latency: {latency_ms}ms")


@bot.command(name="info")
async def info(ctx):
    """Show bot information."""
    embed = discord.Embed(
        title="🤖 Bot Information",
        color=discord.Color.blue()
    )
    embed.add_field(name="Prefix", value=Config.BOT_PREFIX, inline=True)
    embed.add_field(name="Servers", value=len(bot.guilds), inline=True)
    embed.add_field(name="Python", value=sys.version.split()[0], inline=True)
    embed.add_field(name="discord.py", value=discord.__version__, inline=True)

    await ctx.send(embed=embed)


def main():
    """Run the bot."""
    # Validate config
    if not Config.validate():
        print("\n❌ Configuration validation failed!")
        print("   Please check your .env file")
        return

    # Run bot
    try:
        print("🚀 Starting bot...")
        bot.run(Config.DISCORD_TOKEN)
    except discord.LoginFailure:
        print("\n❌ Invalid bot token!")
        print("   Check DISCORD_TOKEN in .env file")
    except Exception as e:
        print(f"\n❌ Error starting bot: {e}")


if __name__ == "__main__":
    main()
```

### Run Your Bot

```bash
# Make sure virtual environment is activated
# You should see (venv) in your prompt

# Run the bot
python src/bot.py

# Expected output:
# 🚀 Starting bot...
# ✅ Bot is ready!
#    Logged in as: My RAG Bot
#    Bot ID: 1234567890
#    Prefix: !
#    Servers: 1
# ---
# Try: !ping in Discord
```

### Test in Discord

1. Go to your test server
2. In #bot-testing channel, type: `!ping`
3. Bot should respond: "🏓 Pong! Latency: XXms"
4. Try: `!info` to see bot information

**If it works: 🎉 Congratulations! Your environment is set up correctly!**

---

## 7. Optional: Local LLM Setup

Want to run everything locally for FREE? Set up Ollama:

### Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from: https://ollama.com/download

### Download Model

```bash
# Download Llama 3.2 (3B parameters, ~2GB)
ollama pull llama3.2:3b

# Or for better quality (7B parameters, ~4GB)
ollama pull llama3.2:7b

# Test it
ollama run llama3.2:3b "Hello, what is RAG?"
```

### Configure for Local LLM

Update `.env`:
```bash
LLM_PROVIDER=local
LLM_MODEL=llama3.2:3b
LOCAL_LLM_URL=http://localhost:11434
```

**Benefits:**
- ✅ FREE (no API costs)
- ✅ Privacy (data stays local)
- ✅ Offline usage
- ✅ Great for learning

**Trade-offs:**
- ❌ Slower than cloud APIs
- ❌ Lower quality than GPT-4
- ❌ Requires more RAM

---

## 8. Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'discord'"

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### "discord.LoginFailure: Improper token has been passed."

**Solution:**
1. Check `.env` file has correct `DISCORD_TOKEN`
2. No spaces around the token
3. Token hasn't been regenerated (get new one from Discord portal)
4. File is named `.env` not `env.txt` or `.env.txt`

#### Bot doesn't respond to commands

**Solution:**
1. Check "Message Content Intent" is enabled in Discord Developer Portal
2. Bot has "Read Messages" permission in your server
3. Prefix is correct (try `!ping`)
4. Bot is online (green dot in Discord)

#### "ImportError: cannot import name 'Config'"

**Solution:**
```bash
# Make sure you're running from project root
cd /path/to/deep-bot
python src/bot.py

# Not from src directory
```

#### Virtual environment not activating

**Windows PowerShell:**
```powershell
# If you get execution policy error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
# Make sure venv was created
ls venv/bin/activate

# If missing, recreate
python3 -m venv venv
```

### Getting Help

If you're still stuck:

1. **Check logs:** Look at the error message carefully
2. **Google the error:** Copy exact error message
3. **Discord.py docs:** https://discordpy.readthedocs.io/
4. **GitHub Issues:** Check if others had same problem
5. **Ask for help:** Include:
   - Error message (full traceback)
   - Python version (`python --version`)
   - OS (Windows/Mac/Linux)
   - Steps you've tried

---

## ✅ Phase 0 Checklist

Before moving to Phase 1, verify:

- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows discord.py, etc.)
- [ ] Discord bot created and token saved
- [ ] `.env` file configured with valid token
- [ ] Project structure created
- [ ] Test bot runs without errors
- [ ] Bot responds to `!ping` in Discord
- [ ] `!info` command works

**All checked? Great! You're ready for Phase 1!** 🚀

---

## Next Steps

Now that your environment is set up, you have two paths:

### Path A: Quick Start (Instant Gratification)
→ Go to [QUICKSTART.md](../QUICKSTART.md)
→ Get a working RAG bot in 5 minutes!

### Path B: Learn by Building (Recommended)
→ Go to [Phase 1: Project Foundation](./PHASE_01.md)
→ Build the bot from scratch, learning every component

Choose based on your learning style. Quick Start is great for seeing what's possible, then come back and learn how it works!

---

## Additional Resources

- **Discord.py Documentation:** https://discordpy.readthedocs.io/
- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **Git Basics:** https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
- **VS Code Python Setup:** https://code.visualstudio.com/docs/python/python-tutorial
- **Ollama Documentation:** https://ollama.com/docs

---

**🎉 Congratulations on completing Phase 0!**

You now have a solid development environment and a working Discord bot. This foundation will make all future phases much smoother.

**Time invested:** ~1 hour
**Value gained:** Countless hours saved from setup issues!
