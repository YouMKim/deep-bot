# Quick Start Guide 🚀

**Get a working Discord AI chatbot in 5 minutes!**

This guide gets you up and running with a minimal demo bot so you can see what you're building before diving into the full learning path.

---

## Prerequisites

- Python 3.8+ installed
- A Discord account
- 5 minutes of your time

**No coding experience needed for this quick demo!**

---

## Step 1: Clone the Repository (1 minute)

```bash
# Clone the repository
git clone https://github.com/YouMKim/deep-bot.git
cd deep-bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install minimal dependencies
pip install discord.py python-dotenv
```

---

## Step 2: Create a Discord Bot (2 minutes)

1. **Go to Discord Developer Portal:**
   - Visit: https://discord.com/developers/applications
   - Click "New Application"
   - Name it "DeepBot Demo" (or whatever you like)
   - Click "Create"

2. **Create the bot:**
   - Click "Bot" in the left sidebar
   - Click "Add Bot" → "Yes, do it!"
   - Under "Token", click "Reset Token" and copy it
   - **Save this token!** You'll need it in Step 3

3. **Set bot permissions:**
   - Scroll down to "Privileged Gateway Intents"
   - Enable:
     - ✅ MESSAGE CONTENT INTENT
     - ✅ SERVER MEMBERS INTENT
     - ✅ PRESENCE INTENT

4. **Invite bot to your server:**
   - Click "OAuth2" → "URL Generator"
   - Select scopes:
     - ✅ `bot`
   - Select permissions:
     - ✅ Read Messages/View Channels
     - ✅ Send Messages
     - ✅ Read Message History
     - ✅ Use Slash Commands
   - Copy the generated URL
   - Open it in your browser and invite the bot to your server

---

## Step 3: Configure the Bot (1 minute)

Create a `.env` file in the `deep-bot` directory:

```bash
# .env
DISCORD_TOKEN=your_token_from_step_2_here
BOT_PREFIX=!
LOG_LEVEL=INFO
```

**Replace `your_token_from_step_2_here` with your actual bot token!**

---

## Step 4: Run the Minimal Demo Bot (1 minute)

Create a file called `quickstart_bot.py`:

```python
"""
Minimal demo bot - No AI, just proves Discord connectivity works.
"""

import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create bot
bot = commands.Bot(
    command_prefix="!",
    intents=discord.Intents.all()
)


@bot.event
async def on_ready():
    """Called when bot connects to Discord."""
    print(f"✅ Bot is online as {bot.user.name}")
    print(f"📊 Connected to {len(bot.guilds)} server(s)")
    print(f"🎯 Try these commands in Discord:")
    print("   !ping - Test bot response")
    print("   !echo <text> - Bot repeats your text")
    print("   !help - Show all commands")


@bot.command(name="ping")
async def ping(ctx):
    """Test command - bot responds with latency."""
    latency_ms = round(bot.latency * 1000)
    await ctx.send(f"🏓 Pong! Latency: {latency_ms}ms")


@bot.command(name="echo")
async def echo(ctx, *, text: str):
    """Bot repeats what you say."""
    await ctx.send(f"You said: {text}")


@bot.command(name="recent")
async def recent(ctx, count: int = 5):
    """Show recent messages in channel."""
    messages = []
    async for message in ctx.channel.history(limit=count):
        if message.author.bot or not message.content.strip():
            continue
        messages.append(f"{message.author.name}: {message.content[:50]}")

    if messages:
        messages_text = "\n".join(reversed(messages))
        await ctx.send(f"📜 **Recent messages:**\n```\n{messages_text}\n```")
    else:
        await ctx.send("No recent messages found.")


# Run bot
if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("❌ DISCORD_TOKEN not found in .env file!")
        exit(1)

    print("🚀 Starting bot...")
    bot.run(token)
```

Run it:

```bash
python quickstart_bot.py
```

You should see:
```
✅ Bot is online as DeepBot Demo
📊 Connected to 1 server(s)
🎯 Try these commands in Discord:
   !ping - Test bot response
   !echo <text> - Bot repeats your text
   !help - Show all commands
```

---

## Step 5: Test in Discord (30 seconds)

Go to your Discord server and try:

```
You: !ping
Bot: 🏓 Pong! Latency: 45ms

You: !echo Hello World!
Bot: You said: Hello World!

You: !recent 3
Bot: 📜 Recent messages:
     Alice: Hey everyone!
     Bob: What's up?
     You: Testing the bot
```

**Congratulations! Your bot is working!** 🎉

---

## What Just Happened?

You now have a **basic Discord bot** that:
- ✅ Connects to Discord
- ✅ Responds to commands
- ✅ Reads channel messages
- ✅ Shows bot latency

**But it's not smart yet!** It can't:
- ❌ Answer questions about chat history
- ❌ Use AI to understand context
- ❌ Search through old messages
- ❌ Remember conversations

---

## Next Steps: Add AI Intelligence

To turn this into a real AI chatbot, you have two paths:

### Path 1: Quick Learning (Recommended for beginners)

Follow the structured learning path:

1. **Phase 0: Development Environment Setup** (30-60 min)
   - Set up proper project structure
   - Install all dependencies
   - Optional: Set up local AI (Ollama)
   - [Read Phase 0](PHASE_00.md)

2. **Phase 2: Simple MVP Chatbot** (1 hour)
   - Add AI-powered question answering
   - Uses OpenAI API or free local Ollama
   - Answers questions about recent messages
   - [Read Phase 2](PHASE_02_MVP.md)

3. **Phase 3: Security Fundamentals** (2-3 hours)
   - Input validation
   - Rate limiting
   - Error handling
   - [Read Phase 3](PHASE_03_SECURITY.md)

4. **Continue with full path:**
   - Phase 4-6: RAG system basics
   - Phase 7-10: Advanced RAG techniques
   - Phase 11-13: Production features
   - [See Full Guide](IMPLEMENTATION_GUIDE.md)

### Path 2: Fast Track (If you're experienced)

If you already know Python and Discord bots:

1. Jump to **Phase 2** to add AI immediately
2. Implement **Phase 6** RAG system for semantic search
3. Add **Phase 18** security for production

---

## Cost Considerations

### Option 1: Free (Local AI with Ollama)

- **Cost:** $0
- **Speed:** Slower (runs on your computer)
- **Quality:** Good for learning
- **Setup:** [Phase 0](PHASE_00.md) shows how to install Ollama

### Option 2: Paid (OpenAI API)

- **Cost:** ~$0.0004 per query (0.04¢)
- **Speed:** Fast (cloud API)
- **Quality:** Excellent
- **Budget:** $5/month = ~12,500 queries

**Recommendation:** Start with Ollama (free) for learning, switch to OpenAI for production.

---

## Troubleshooting

### Bot doesn't connect

**Error:** `discord.errors.LoginFailure: Improper token has been passed`

**Fix:**
1. Check your `.env` file has the correct token
2. Make sure there are no spaces around the token
3. Try regenerating the token in Discord Developer Portal

### Bot is online but doesn't respond

**Fix:**
1. Make sure bot has "Read Messages" permission in your server
2. Check that "MESSAGE CONTENT INTENT" is enabled in bot settings
3. Try mentioning the bot: `@DeepBot !ping`

### "Module not found" error

**Fix:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install discord.py python-dotenv
```

### Bot can't read messages

**Fix:**
1. In Discord Developer Portal → Your App → Bot
2. Enable "MESSAGE CONTENT INTENT"
3. Restart your bot

---

## Project Structure Overview

Once you complete Phase 0, your project will look like this:

```
deep-bot/
├── bot.py                 # Main bot file
├── config.py              # Configuration management
├── .env                   # Your secrets (never commit!)
├── .env.example           # Template for .env
├── requirements.txt       # Python dependencies
│
├── cogs/                  # Bot commands (organized features)
│   ├── basic.py          # Basic commands (!ping, !help)
│   ├── mvp_chatbot.py    # Simple AI chatbot (Phase 2)
│   ├── summary.py        # Message summarization
│   └── admin.py          # Admin commands
│
├── services/              # Business logic
│   ├── ai_service.py     # AI/LLM integration
│   ├── memory_service.py # Vector database
│   └── message_storage.py # Message persistence
│
├── utils/                 # Utilities
│   ├── input_validator.py
│   ├── rate_limiter.py
│   └── error_handler.py
│
└── data/                  # Data storage
    ├── chroma/           # Vector database
    └── raw_messages/     # SQLite database
```

---

## Learning Resources

- **Discord.py Documentation:** https://discordpy.readthedocs.io/
- **OpenAI API Docs:** https://platform.openai.com/docs
- **Ollama (Free Local AI):** https://ollama.com/
- **Full Implementation Guide:** [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)

---

## Common Questions

**Q: Do I need to pay for anything?**
A: No! You can use Ollama (local AI) for completely free learning. Only pay if you want faster responses with OpenAI.

**Q: Can I use this bot in multiple servers?**
A: Yes! But the chatbot will only answer questions about messages in the channel where you ask.

**Q: How do I deploy this to run 24/7?**
A: See Phase 0 for deployment options (Heroku, Railway, AWS, etc.)

**Q: Is this beginner-friendly?**
A: Yes! The guide assumes no prior knowledge and teaches you step-by-step.

**Q: How long does the full project take?**
A:
- Quick demo: 5 minutes ✅ (you just did it!)
- MVP with AI: 3-4 hours
- Full RAG system: 20-30 hours
- Production-ready: 40-60 hours

**Q: Can I customize the bot's personality?**
A: Yes! In Phase 2+, you'll learn how to modify the AI's system prompt to change its behavior.

---

## Get Help

- **Issues:** https://github.com/YouMKim/deep-bot/issues
- **Discussions:** https://github.com/YouMKim/deep-bot/discussions
- **Discord:** [Join our community] (if you have a Discord server)

---

## What's Next?

You have three options:

1. **Learn step-by-step** → Start with [Phase 0: Development Environment Setup](PHASE_00.md)
2. **Add AI now** → Jump to [Phase 2: MVP Chatbot](PHASE_02_MVP.md)
3. **Understand the full picture** → Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)

**Most people should start with Phase 0!** It sets up everything properly.

---

**Ready to build something amazing?** Let's go! 🚀
