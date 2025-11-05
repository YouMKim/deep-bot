# Phase 2: Simple MVP Chatbot (Quick Win) 🚀

**Estimated Time:** 1 hour
**Difficulty:** ⭐ Beginner
**Prerequisites:** Phase 0 (Development Environment Setup)

---

## Overview

This phase gets you a **working chatbot in 1 hour**. No vector databases, no embeddings, no complex RAG system yet—just a simple bot that can answer questions about recent chat history.

**Why Phase 2?**
- ✅ Provides instant gratification (working bot quickly)
- ✅ Teaches core concepts (Discord commands, AI API, context management)
- ✅ Builds confidence before complexity
- ✅ You can show friends immediately!

**What You'll Build:**
```
User: !ask What did we talk about earlier?
Bot: You discussed Python tutorials and debugging tips.
     Alice mentioned a helpful resource at...
```

---

## Learning Objectives

By the end of this phase, you will:
1. Implement a basic `!ask` command
2. Fetch recent messages from Discord channel
3. Send chat history to an AI model
4. Parse and display AI responses
5. Understand token limits and context windows
6. Handle basic errors gracefully

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Simple MVP Chatbot                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  User: !ask <question>                                       │
│     │                                                        │
│     ├──> Fetch last 50 messages from channel                │
│     │                                                        │
│     ├──> Format messages as context                         │
│     │                                                        │
│     ├──> Send to AI (OpenAI/Ollama)                        │
│     │    "Answer this question based on:                    │
│     │     [chat history]                                     │
│     │     Question: <question>"                              │
│     │                                                        │
│     └──> Display AI response to user                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**No vector database yet!** Just recent messages sent directly to AI.

---

## Step 1: Install Minimal Dependencies

Create a `requirements-mvp.txt` file:

```txt
# requirements-mvp.txt
discord.py==2.3.2
python-dotenv==1.0.0
openai==1.12.0
tiktoken==0.6.0
```

Install:
```bash
pip install -r requirements-mvp.txt
```

---

## Step 2: Update Your `.env` File

Make sure you have these variables set:

```bash
# Discord Configuration
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CLIENT_ID=your_client_id_here

# OpenAI Configuration (or skip for Ollama)
OPENAI_API_KEY=sk-your-openai-key-here

# Bot Configuration
BOT_PREFIX=!
LOG_LEVEL=INFO

# MVP Settings
MVP_MESSAGE_LIMIT=50           # How many recent messages to fetch
MVP_MAX_CONTEXT_TOKENS=3000    # Token limit for context
MVP_AI_PROVIDER=openai         # or 'ollama' for local
```

**Using Ollama (Free Local Option):**
If you set up Ollama in Phase 0, set:
```bash
MVP_AI_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b       # or mistral, phi3, etc.
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Step 3: Create the MVP Chatbot Cog

Create `cogs/mvp_chatbot.py`:

```python
"""
Simple MVP Chatbot Cog
Answers questions based on recent chat history without vector database.
"""

import discord
from discord.ext import commands
import openai
import tiktoken
import os
from typing import List, Dict
from datetime import datetime


class MVPChatbot(commands.Cog):
    """Simple chatbot using recent messages as context."""

    def __init__(self, bot):
        self.bot = bot
        self.provider = os.getenv("MVP_AI_PROVIDER", "openai")
        self.message_limit = int(os.getenv("MVP_MESSAGE_LIMIT", "50"))
        self.max_tokens = int(os.getenv("MVP_MAX_CONTEXT_TOKENS", "3000"))

        # Initialize AI client
        if self.provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = "gpt-4o-mini"  # Cheapest GPT-4 model
            self.encoding = tiktoken.encoding_for_model(self.model)
        elif self.provider == "ollama":
            openai.api_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            openai.api_key = "dummy"  # Ollama doesn't need a real key
            self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            self.encoding = tiktoken.get_encoding("cl100k_base")  # Approximation

    @commands.command(
        name="ask",
        help="Ask a question about recent chat history",
        usage="!ask <your question>"
    )
    @commands.cooldown(1, 10, commands.BucketType.user)  # 1 per 10 seconds
    async def ask(self, ctx, *, question: str):
        """Answer questions based on recent chat history."""

        # Input validation
        if len(question) < 3:
            await ctx.send("❌ Question too short. Please be more specific.")
            return

        if len(question) > 500:
            await ctx.send("❌ Question too long. Keep it under 500 characters.")
            return

        # Send "thinking" message
        thinking_msg = await ctx.send("🤔 Analyzing recent messages...")

        try:
            # Step 1: Fetch recent messages
            messages = await self._fetch_recent_messages(ctx.channel)

            if not messages:
                await thinking_msg.edit(content="❌ No messages found to analyze.")
                return

            await thinking_msg.edit(content=f"📚 Found {len(messages)} messages. Thinking...")

            # Step 2: Build context from messages
            context = self._build_context(messages)

            # Step 3: Get AI response
            answer = await self._get_ai_response(question, context)

            # Step 4: Send response
            await thinking_msg.delete()

            # Create embed for nice formatting
            embed = discord.Embed(
                title="💡 Answer",
                description=answer,
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
            embed.set_footer(text=f"Question by {ctx.author.name}")
            embed.add_field(
                name="📊 Context",
                value=f"Analyzed {len(messages)} recent messages",
                inline=False
            )

            await ctx.send(embed=embed)

        except Exception as e:
            await thinking_msg.edit(content=f"❌ Error: {str(e)}")
            print(f"Error in ask command: {e}")

    async def _fetch_recent_messages(self, channel) -> List[Dict]:
        """Fetch recent messages from channel."""
        messages = []

        async for message in channel.history(limit=self.message_limit):
            # Skip bot messages and empty messages
            if message.author.bot or not message.content.strip():
                continue

            # Skip commands
            if message.content.startswith("!"):
                continue

            messages.append({
                "author": message.author.display_name,
                "content": message.content,
                "timestamp": message.created_at.strftime("%Y-%m-%d %H:%M")
            })

        # Reverse to chronological order (oldest first)
        return list(reversed(messages))

    def _build_context(self, messages: List[Dict]) -> str:
        """Build context string from messages, respecting token limit."""
        context_parts = []
        token_count = 0

        for msg in messages:
            # Format: "2024-01-15 10:30 - Alice: Hello everyone!"
            line = f"{msg['timestamp']} - {msg['author']}: {msg['content']}"

            # Count tokens in this line
            line_tokens = len(self.encoding.encode(line))

            # Stop if we exceed token limit
            if token_count + line_tokens > self.max_tokens:
                break

            context_parts.append(line)
            token_count += line_tokens

        return "\n".join(context_parts)

    async def _get_ai_response(self, question: str, context: str) -> str:
        """Get response from AI model."""

        system_prompt = """You are a helpful Discord chat assistant.
Answer questions based ONLY on the chat history provided.
If you cannot find relevant information in the chat history, say so.
Keep answers concise (2-3 sentences max)."""

        user_prompt = f"""Chat History:
{context}

Question: {question}

Answer the question based on the chat history above."""

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"AI request failed: {str(e)}")

    @ask.error
    async def ask_error(self, ctx, error):
        """Handle command errors."""
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(
                f"⏰ Slow down! Try again in {error.retry_after:.1f} seconds."
            )
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("❌ Please provide a question. Usage: `!ask <question>`")
        else:
            await ctx.send(f"❌ An error occurred: {str(error)}")


async def setup(bot):
    """Load the cog."""
    await bot.add_cog(MVPChatbot(bot))
```

---

## Step 4: Register the Cog in Your Bot

Update `bot.py` to load the MVP chatbot cog:

```python
# In bot.py, in the setup_hook method:

async def setup_hook(self):
    """Called when the bot is starting up."""
    # Load configuration
    Config.load_blacklist()

    # Load cogs
    try:
        await self.load_extension("cogs.basic")
        logger.info("Loaded basic cog")
    except Exception as e:
        logger.error(f"Failed to load basic cog: {e}")

    # ADD THIS:
    try:
        await self.load_extension("cogs.mvp_chatbot")
        logger.info("Loaded MVP chatbot cog")
    except Exception as e:
        logger.error(f"Failed to load MVP chatbot cog: {e}")
```

---

## Step 5: Test Your MVP Chatbot

1. **Start your bot:**
   ```bash
   python bot.py
   ```

2. **In Discord, have a conversation:**
   ```
   Alice: Hey everyone! I'm learning Python.
   Bob: Cool! Check out the Real Python tutorials.
   Alice: Thanks! I'll look into that.
   ```

3. **Ask the bot:**
   ```
   You: !ask What resources did Bob recommend?
   Bot: 💡 Answer
        Bob recommended checking out the Real Python tutorials
        for learning Python.

        📊 Context: Analyzed 3 recent messages
   ```

4. **Try another question:**
   ```
   You: !ask Who is learning Python?
   Bot: 💡 Answer
        Alice is learning Python.

        📊 Context: Analyzed 3 recent messages
   ```

---

## Understanding the Code

### Key Concepts

**1. Message Fetching**
```python
async for message in channel.history(limit=50):
    # Process each message
```
- Fetches messages asynchronously
- `limit=50` gets last 50 messages
- Filters out bots and commands

**2. Token Counting**
```python
line_tokens = len(self.encoding.encode(line))
```
- Counts how many tokens are in text
- Prevents exceeding AI model limits
- Different models have different limits

**3. Context Window Management**
```python
if token_count + line_tokens > self.max_tokens:
    break  # Stop adding messages
```
- AI models have token limits (e.g., 8,000 tokens)
- We limit context to 3,000 tokens
- Leaves room for question + response

**4. Rate Limiting**
```python
@commands.cooldown(1, 10, commands.BucketType.user)
```
- Prevents spam
- 1 use per 10 seconds per user
- Saves API costs

---

## Limitations of This MVP

This simple approach has limitations:

❌ **Only uses recent messages** - Can't search older conversations
❌ **Limited by token window** - Can't process very long discussions
❌ **No semantic search** - Just sends all recent messages
❌ **Inefficient** - Sends same context for every question
❌ **No memory** - Each question is independent

**These limitations are addressed in later phases:**
- Phase 4: Message chunking for longer conversations
- Phase 5: Vector database for semantic search
- Phase 6: RAG system for efficient retrieval
- Phase 11: Conversation memory

---

## Cost Estimation

**Using OpenAI (gpt-4o-mini):**
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

**Typical query:**
- Context: ~2,000 tokens (50 messages)
- Question: ~50 tokens
- Response: ~150 tokens
- **Cost per query: ~$0.0004 (0.04¢)**

**100 queries = ~$0.04 (4 cents)**

**Using Ollama (local):**
- **Cost: $0** (runs on your computer)
- Slower but completely free
- Good for learning and testing

---

## Common Issues & Troubleshooting

### Issue 1: Bot doesn't respond

**Check:**
1. Is bot online in Discord?
2. Does bot have "Read Message History" permission?
3. Are you using correct prefix (`!ask` not `ask`)?

**Fix:**
```bash
# Check bot logs
python bot.py
# Look for "Loaded MVP chatbot cog" message
```

### Issue 2: "Rate limit exceeded"

**Error:** `openai.RateLimitError: Rate limit exceeded`

**Fix:**
- You're making too many requests
- Wait a few seconds between queries
- Or upgrade OpenAI plan for higher limits

### Issue 3: Bot gives generic answers

**Problem:** Bot isn't using chat history

**Fix:**
- Check that messages exist in channel
- Verify bot can read message history
- Add print statement to debug:
  ```python
  print(f"Fetched {len(messages)} messages")
  print(f"Context length: {len(context)} characters")
  ```

### Issue 4: "Context too long" error

**Error:** `This model's maximum context length is X tokens`

**Fix:**
- Reduce `MVP_MESSAGE_LIMIT` in `.env`
- Reduce `MVP_MAX_CONTEXT_TOKENS` in `.env`

---

## Exercises to Extend This MVP

Try these improvements on your own:

### Exercise 1: Add Message Counter
Show how many messages were actually used (after token limit)

```python
# In _build_context, return both context and count
def _build_context(self, messages):
    # ... existing code ...
    return "\n".join(context_parts), len(context_parts)

# Update caller to show count
context, count = self._build_context(messages)
embed.add_field(name="📊 Context", value=f"Analyzed {count} messages")
```

### Exercise 2: Add Source Citations
Show which messages the answer came from

```python
# After getting AI response, add:
system_prompt += "\nInclude timestamps of relevant messages in your answer."
```

### Exercise 3: Support Follow-up Questions
Remember the last question and context

```python
# Add to __init__:
self.last_context = {}  # {user_id: context}

# In ask command:
# Check if follow-up
if question.lower().startswith("follow up:"):
    context = self.last_context.get(ctx.author.id, context)
else:
    self.last_context[ctx.author.id] = context
```

### Exercise 4: Add Cost Tracking
Log and display API costs

```python
# After AI response:
input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens
cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

embed.add_field(name="💰 Cost", value=f"${cost:.6f}")
```

---

## What's Next?

Congratulations! You now have a working chatbot. 🎉

**Current capabilities:**
✅ Answers questions about recent chat history
✅ Handles errors gracefully
✅ Rate-limited to prevent abuse
✅ Works with OpenAI or Ollama

**What you'll learn in upcoming phases:**

- **Phase 3:** Security fundamentals (input validation, safe prompts)
- **Phase 4:** Token-aware chunking (handle longer conversations)
- **Phase 5:** Vector database (ChromaDB for semantic search)
- **Phase 6:** RAG system (efficient retrieval, better accuracy)
- **Phase 7:** Multiple embedding strategies
- **Phase 8:** Hybrid search (combine keyword + semantic)

**Next recommended step:** Phase 3 - Security Fundamentals

---

## Key Takeaways

1. **Start simple** - This MVP proves the concept quickly
2. **Token limits matter** - Always count tokens before sending to AI
3. **Rate limiting is essential** - Protects your API budget
4. **Error handling is important** - Users will try unexpected things
5. **Recent context works** - For many use cases, last 50 messages is enough

You don't need complex RAG systems for every use case. Sometimes simple is better!

---

## Additional Resources

- [Discord.py Message History](https://discordpy.readthedocs.io/en/stable/api.html#discord.TextChannel.history)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/chat-completions)
- [Tiktoken Documentation](https://github.com/openai/tiktoken)
- [Ollama Models](https://ollama.com/library)

---

**Ready to add security?** → Proceed to Phase 3: Security Fundamentals
**Want to see the difference?** → Try Phase 6 RAG and compare results!
