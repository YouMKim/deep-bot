# Frequently Asked Questions (FAQ) ❓

Quick answers to common questions about building your Discord AI chatbot.

---

## General Questions

### Q: What is this project?

**A:** This is a comprehensive learning guide to build a Discord bot that can answer questions about your server's chat history using RAG (Retrieval-Augmented Generation) and AI.

Think of it as giving your Discord server a smart assistant that remembers everything said and can answer questions like "What did we discuss about Python last week?"

### Q: Do I need coding experience?

**A:** Some Python knowledge is helpful, but the guide is beginner-friendly:
- **No experience:** Follow Phase 0 carefully, expect 60+ hours total
- **Basic Python:** You'll be comfortable, ~30-40 hours total
- **Experienced:** You can skip to Phase 2-6, ~10-20 hours

### Q: Do I need to pay for anything?

**A:** No! You can build everything for free using:
- **Discord:** Free
- **Ollama:** Free local AI (instead of OpenAI)
- **sentence-transformers:** Free local embeddings
- **ChromaDB:** Free local vector database

Only pay if you want faster responses with OpenAI (~$5-20/month).

### Q: How long does it take to complete?

**A:**
- **Quick demo:** 5 minutes ([QUICKSTART.md](QUICKSTART.md))
- **Basic chatbot (Phase 0-2):** 3-4 hours
- **Full RAG system (Phase 0-10):** 20-30 hours
- **Production-ready (Phase 0-18):** 40-60 hours

You can stop at any phase—each adds new capabilities!

---

## Technical Questions

### Q: What's the difference between a basic chatbot and RAG?

**A:**

**Basic chatbot (Phase 2):**
- Only knows about recent 50 messages
- Limited by token window (~4,000 tokens)
- Fast but forgetful

**RAG chatbot (Phase 6+):**
- Searches through thousands of messages
- Uses semantic search (understands meaning, not just keywords)
- Retrieves relevant context from entire history
- Can answer questions about months-old conversations

### Q: What is RAG?

**A:** RAG = Retrieval-Augmented Generation

Traditional AI has no knowledge of your chat history. RAG adds a memory system:

1. **Store** messages in a vector database (converts text to numbers)
2. **Search** when someone asks a question (finds relevant messages)
3. **Generate** an answer using AI + retrieved context

It's like giving the AI access to a search engine for your chat history!

### Q: What programming language is used?

**A:** Python 3.8+

Why Python?
- ✅ Easiest for AI/ML projects
- ✅ Best libraries (discord.py, OpenAI, transformers)
- ✅ Great for beginners
- ✅ Industry standard for AI

### Q: What's a vector database?

**A:** A special database that stores "embeddings" (vectors) instead of just text.

**Example:**
```
Text: "I love Python programming"
Embedding: [0.23, -0.45, 0.67, ..., 0.12]  (384 numbers)
```

Vector databases can find similar messages by comparing these numbers, even if they use different words!

**"best Python tutorials"** matches **"great resources for learning Python"** (semantic similarity)

### Q: Do I need a powerful computer?

**A:**

**Minimum (cloud APIs only):**
- 4GB RAM
- Any processor
- 5GB disk space

**Recommended (with local AI):**
- 8GB+ RAM
- Modern processor (last 5 years)
- 20GB disk space

**For local LLMs:**
- 16GB+ RAM (for 7B models)
- 32GB+ RAM (for 13B+ models)

### Q: Can I use GPT-4 instead of GPT-4o-mini?

**A:** Yes! Just change the model:

```python
# In your .env file:
MVP_AI_PROVIDER=openai

# In code:
self.model = "gpt-4"  # Instead of "gpt-4o-mini"
```

**But be aware:**
- GPT-4 costs 65x more than GPT-4o-mini
- $0.026 per query vs $0.0004
- Budget: $20/month = ~770 GPT-4 queries vs 50,000 gpt-4o-mini queries

**Recommendation:** Use gpt-4o-mini for 95% of queries, GPT-4 for complex questions only.

---

## Setup & Installation Questions

### Q: How do I install Python?

**A:** See [Phase 0: Step 1](PHASE_00.md#step-1-install-python)

**Quick links:**
- Windows: https://python.org/downloads/
- macOS: Pre-installed or `brew install python3`
- Linux: `sudo apt install python3 python3-pip`

Verify: `python --version` (should show 3.8 or higher)

### Q: What's a virtual environment and why do I need it?

**A:** A virtual environment is an isolated Python installation for your project.

**Why?**
- ✅ Keeps your project dependencies separate
- ✅ Prevents version conflicts
- ✅ Easy to reproduce on other computers
- ✅ Won't mess up your system Python

**Create one:**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Q: How do I get a Discord bot token?

**A:** See [Phase 0: Step 3](PHASE_00.md#step-3-create-discord-bot-application) or [QUICKSTART.md: Step 2](QUICKSTART.md#step-2-create-a-discord-bot-2-minutes)

**Quick summary:**
1. Go to https://discord.com/developers/applications
2. Create New Application
3. Go to "Bot" → "Add Bot"
4. Click "Reset Token" and copy it
5. Enable "MESSAGE CONTENT INTENT"
6. Paste token in `.env` file

### Q: I keep getting "discord.errors.LoginFailure: Improper token"

**A:** Common fixes:

1. **Check `.env` format:**
   ```bash
   # ✅ Correct:
   DISCORD_TOKEN=MTA1234567890.GH-abc.xyz123_def456

   # ❌ Wrong:
   DISCORD_TOKEN = MTA1234567890.GH-abc.xyz123_def456  # Extra spaces!
   DISCORD_TOKEN="MTA1234567890..."  # Quotes not needed!
   ```

2. **Regenerate token:**
   - Discord Developer Portal → Your App → Bot
   - Click "Reset Token"
   - Copy new token to `.env`

3. **Check `.env` is in project root:**
   ```bash
   deep-bot/
   ├── .env          # ✅ Here
   ├── bot.py
   └── ...
   ```

### Q: Bot is online but doesn't respond to commands

**A:** Check these permissions:

1. **In Discord Developer Portal:**
   - Bot → Privileged Gateway Intents
   - Enable "MESSAGE CONTENT INTENT" ✅

2. **In your Discord server:**
   - Bot needs "Read Messages" permission
   - Bot needs "Send Messages" permission
   - Bot can see the channel you're testing in

3. **Restart bot after enabling intents!**

### Q: How do I install Ollama for free local AI?

**A:** See [Phase 0: Optional Setup](PHASE_00.md#optional-setup-ollama-for-local-ai)

**Quick install:**
1. Download: https://ollama.com/
2. Install and run Ollama
3. Download a model: `ollama pull llama3.2:3b`
4. Test: `ollama run llama3.2:3b "Hello!"`
5. Update `.env`: `MVP_AI_PROVIDER=ollama`

**Recommended models:**
- `llama3.2:3b` (2GB, fast, decent quality)
- `mistral:7b` (4GB, slower, better quality)
- `phi3:mini` (2GB, very fast)

---

## Usage Questions

### Q: How much does it cost to run?

**A:** See [COST_MANAGEMENT.md](COST_MANAGEMENT.md)

**Quick summary:**

| Setup | Monthly Cost | Queries/Day |
|-------|--------------|-------------|
| Ollama (local) | $0 | Unlimited |
| gpt-4o-mini | $1-5 | 50-300 |
| gpt-4o-mini | $10-20 | 500-1000 |
| GPT-4 | $50+ | 500-1000 |

**Most users:** $0-5/month is plenty!

### Q: Can I use this in multiple Discord servers?

**A:** Yes! Invite your bot to multiple servers using the OAuth2 URL.

**Each server:**
- Has separate message history
- Can have different settings
- Independent rate limits

**To scale:**
- Use Phase 26: Multi-Guild Support
- Configure per-guild settings
- Separate vector databases per guild (optional)

### Q: How do I make the bot run 24/7?

**A:** Your bot only runs when your computer is on. For 24/7:

**Free options:**
- Railway.app (free tier)
- Render.com (free tier)
- Oracle Cloud (always free tier)

**Paid options:**
- Heroku ($7/month)
- DigitalOcean ($6/month)
- AWS EC2 ($5-10/month)

**Local option:**
- Raspberry Pi (one-time ~$50)
- Old laptop always running

See deployment guides in Phase 0.

### Q: Can the bot read messages from before it joined?

**A:** No, Discord doesn't allow bots to fetch messages sent before they joined.

**Solution:** Run `!load_messages` command after inviting the bot to load available history (limited to recent messages by Discord's API).

### Q: How many messages can it remember?

**A:**

**Technical limit:** Millions (vector databases scale well)

**Practical limits:**
- **Discord API:** Can fetch ~millions with pagination
- **Embedding cost:** $0.02 per 1M tokens (OpenAI) or free (local)
- **Storage:** ~1KB per message

**Recommendation:**
- Small server: Store everything (10,000-100,000 messages)
- Large server: Store last 6-12 months
- Archive older messages if needed

### Q: Can it search through images/files?

**A:** Not in the basic implementation (Phase 0-10).

For multimodal search, see:
- **Phase 23:** Multimodal RAG
  - OCR for images with text
  - Image embeddings (CLIP)
  - File parsing (PDFs, docs)

### Q: How accurate is the chatbot?

**A:** Depends on several factors:

**Model quality:**
- Ollama (local): 70-80% accurate
- gpt-4o-mini: 85-90% accurate
- GPT-4: 90-95% accurate

**Retrieval quality:**
- Good embeddings: Finds relevant messages
- Poor embeddings: May miss context

**Context quality:**
- Clear questions: Better answers
- Vague questions: May hallucinate

**Improvements:**
- Phase 6: Better retrieval (RAG)
- Phase 8: Hybrid search
- Phase 15: Reranking
- Phase 18: Prompt injection defense

---

## Troubleshooting

### Q: "ModuleNotFoundError: No module named 'discord'"

**A:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-minimal.txt
```

### Q: "RateLimitError: Rate limit exceeded"

**A:** You're making too many API calls.

**Fixes:**
1. Add rate limiting to commands (see Phase 3)
2. Increase cooldown between requests
3. Upgrade OpenAI plan for higher limits
4. Switch to local Ollama (no limits)

### Q: Bot is slow to respond

**A:**

**If using Ollama:**
- Normal for local AI (5-30 seconds)
- Use smaller models (llama3.2:3b)
- Upgrade computer (more RAM/CPU)

**If using OpenAI:**
- Should be 1-3 seconds
- Check internet connection
- Check OpenAI status page

**To improve:**
- Use streaming responses
- Show "thinking..." message
- Reduce context window size

### Q: Bot crashes with "out of memory"

**A:**

**If loading messages:**
- Process in smaller batches
- Reduce `MESSAGE_FETCH_BATCH_SIZE`

**If using local embeddings:**
- Reduce `EMBEDDING_BATCH_SIZE`
- Use smaller model (all-MiniLM-L6-v2)

**If using Ollama:**
- Use smaller model (3B instead of 7B)
- Close other applications
- Upgrade RAM

### Q: "Error: Could not find embeddings model"

**A:**
```bash
# First time only: download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Model will download (~80MB) to `~/.cache/huggingface/`

---

## Phase-Specific Questions

### Q: Can I skip phases?

**A:**

**Required phases:**
- Phase 0 (environment setup)
- Phase 1 (message loading)
- Phase 4 (chunking)
- Phase 5 (embeddings)
- Phase 6 (RAG basics)

**Recommended but optional:**
- Phase 2 (MVP chatbot - good quick win)
- Phase 3 (security - important but can come later)
- Phase 7-10 (improvements)

**Advanced/optional:**
- Phase 11-17 (advanced features)
- Phase 18 (advanced security)
- Phase 19+ (production features)

### Q: I finished Phase 2, what's the difference in Phase 6?

**A:**

**Phase 2 (MVP):**
- Uses last 50 messages only
- Sends all messages to AI every time
- No search capability
- Limited by token window

**Phase 6 (RAG):**
- Searches through entire message history
- Only sends relevant messages to AI
- Semantic search (understands meaning)
- Scales to millions of messages
- Much more accurate

**When to upgrade:** When you need to search older messages or have >100 messages.

### Q: What's the difference between Phase 18 and Phase 3 security?

**A:**

**Phase 3 (Security Fundamentals):**
- Input validation
- Rate limiting
- Error handling
- Basic prompt injection detection

**Phase 18 (Advanced Security):**
- Multi-layer defense (8 layers)
- ML-based injection detection
- Adversarial testing
- Security audit logging
- Production-grade protection

**Recommendation:** Complete Phase 3 early, add Phase 18 before production deployment.

---

## Contribution Questions

### Q: Can I contribute to this project?

**A:** Yes! Contributions welcome:

1. **Bug reports:** Open an issue
2. **Feature suggestions:** Open a discussion
3. **Code improvements:** Submit a pull request
4. **Documentation:** Fix typos, add examples
5. **Examples:** Share your customizations

See [CONTRIBUTING.md](CONTRIBUTING.md) (if exists)

### Q: Can I use this for commercial projects?

**A:** Yes! This project is open-source (check LICENSE for specific terms).

**Requirements:**
- Follow OpenAI/Anthropic terms of service
- Follow Discord terms of service & bot policies
- Consider security implications (Phase 18)
- Handle user data responsibly (GDPR, privacy)

### Q: Can I customize the bot's personality?

**A:** Yes! Modify the system prompt:

```python
# In your AI service
system_prompt = """You are a helpful Discord assistant.
You are friendly, concise, and helpful.
You answer questions about chat history."""

# Want pirate bot?
system_prompt = """You are a pirate Discord assistant.
Speak like a pirate! Say "Arrr!" a lot.
You answer questions about chat history, matey!"""
```

See Phase 11 for advanced personality customization.

---

## Support & Community

### Q: Where can I get help?

**A:**

1. **Check this FAQ first!**
2. **Read the relevant phase documentation**
3. **Search existing issues:** https://github.com/YouMKim/deep-bot/issues
4. **Open a new issue:** Describe your problem with error messages
5. **Join Discord community:** [Link if available]

When asking for help, include:
- Which phase you're on
- Error message (full traceback)
- What you've tried
- Your Python version
- Your OS (Windows/macOS/Linux)

### Q: Is there a Discord server for this project?

**A:** [Add Discord server link if available, or:]

Not yet! If there's interest, we'll create one. Star the repo and watch for updates!

### Q: How often is this guide updated?

**A:** Actively maintained! Updates include:
- New phases for emerging techniques
- Bug fixes
- Dependency updates
- Better examples

Check the [CHANGELOG.md](CHANGELOG.md) (if exists) for recent updates.

---

## Still Have Questions?

**Didn't find your answer?**

1. Search existing issues: https://github.com/YouMKim/deep-bot/issues
2. Open a new issue with the `question` label
3. Be specific and include context

**Want to chat with others building this?**

Check if there's a Discord community link in the README!

---

**Ready to get started?** → [QUICKSTART.md](QUICKSTART.md) or [Phase 0](PHASE_00.md)
