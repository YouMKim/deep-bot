# Chatbot Usage Guide

A comprehensive guide to using the conversational AI chatbot in your Discord server.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [How It Works](#how-it-works)
4. [Features](#features)
5. [Commands](#commands)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Rate Limits](#rate-limits)
9. [FAQ](#faq)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The chatbot provides natural, conversational AI in a dedicated Discord channel. It can:
- **Chat naturally** without command prefixes
- **Answer questions** about past Discord conversations using RAG (Retrieval-Augmented Generation)
- **Remember context** from your conversation session
- **Filter by user** when you mention them in questions
- **Track usage** and costs per user

### Key Features

- 🤖 **Dual-mode intelligence**: Conversational chat + RAG-powered Q&A
- 💬 **Context-aware**: Remembers your conversation history
- 🔍 **Discord memory**: Searches past messages to answer questions
- 👤 **User filtering**: Ask questions about specific users
- ⏰ **Rate limiting**: Fair usage with 10 messages/minute
- 💰 **Cost tracking**: Monitor your AI usage and costs

---

## Quick Start

### 1. Find the Chatbot Channel

The chatbot only responds in a specific channel (configured by admin).

**How to find it:**
- Look for the channel ID in the bot configuration
- Ask your server admin which channel is the "chatbot channel"
- Try sending a message in the suspected channel

### 2. Start Chatting

Just send a message - no command prefix needed!

```
You: Hello!
Bot: Hello! How can I assist you today?
```

### 3. Ask Questions About Past Conversations

```
You: What did Alice say about the database?
Bot: [Searches Discord history]
Based on the conversation history, Alice mentioned that she prefers PostgreSQL
for the new project because of its JSON support and reliability.
```

---

## How It Works

### Intelligent Mode Selection

The chatbot automatically chooses between two modes:

```
Your Message
    ↓
Is it a question? ──No──> Chat Mode (conversational)
    ↓ Yes
Does it need Discord history? ──No──> Chat Mode (conversational)
    ↓ Yes
RAG Mode (search history)
```

### Chat Mode

For general conversation, greetings, and questions about the bot:

```
You: How are you?
Bot: I'm doing great, thanks for asking! How can I help you?
```

**Triggers Chat Mode:**
- Greetings: "hi", "hello", "what's up"
- Questions about the bot: "what can you do?"
- General questions: "how does X work?"

### RAG Mode

For questions about past conversations or specific users:

```
You: @Bob what did you decide about the API?
Bot: [Searches Bob's messages]
Bob decided to use REST instead of GraphQL for simplicity.
```

**Triggers RAG Mode:**
- User mentions: "@Alice what did you say?"
- Temporal references: "what did we discuss yesterday?"
- History keywords: "who mentioned the meeting?"
- Decision keywords: "what was decided?"

📖 **See:** `docs/CHATBOT_RAG_DIFFERENTIATION.md` for full details.

---

## Features

### 1. Session-Based Conversations

Each user gets their own conversation session that:
- Remembers your messages and the bot's responses
- Maintains context for natural follow-up questions
- Expires after 30 minutes of inactivity (configurable)
- Can be reset with `!chatbot_reset`

**Example:**
```
You: What is Python?
Bot: Python is a high-level programming language known for its simplicity...

You: Can you give me an example?  ← Bot remembers we're talking about Python
Bot: Sure! Here's a simple Python example...
```

### 2. Discord History Search (RAG)

The bot can search through past Discord messages to answer questions:

**Search by user:**
```
You: @Alice what did you say about deployment?
Bot: [Searches only Alice's messages]
```

**Search by topic:**
```
You: What did we discuss about the database schema?
Bot: [Searches all relevant messages]
```

**Search by time:**
```
You: What was decided yesterday?
Bot: [Searches messages from yesterday]
```

### 3. Channel Context Awareness

In Chat mode, the bot can see recent channel messages for context:

```
[Recent in channel]
Alice: I'm having trouble with Docker
Bob: Have you tried rebuilding the image?

You: Can you help with this?  ← Bot sees the recent context
Bot: Based on Alice's Docker issue, I recommend checking your Dockerfile...
```

### 4. Usage Tracking

Track your AI usage with `!chatbot_stats`:

```
📊 Chatbot Statistics for YourName

Messages in Session: 12
Session Started: 2025-11-19 14:30:00
Last Activity: 2025-11-19 15:45:00
Total Cost: $0.0234
Total Tokens: 15,432
Social Credit: 98.50
```

---

## Commands

### User Commands

#### `!chatbot_reset`

Reset your conversation history and start fresh.

```
!chatbot_reset
✅ @You Your conversation history has been cleared!
```

**When to use:**
- Starting a new topic
- Bot seems confused by old context
- You want a fresh start

#### `!chatbot_stats`

View your usage statistics.

```
!chatbot_stats
[Shows embed with session info, costs, tokens]
```

**Shows:**
- Messages in current session
- Session start time
- Last activity time
- Lifetime cost
- Lifetime tokens used
- Social credit score

### Admin Commands

#### `!chatbot_mode`

View current chatbot configuration (admin only).

```
!chatbot_mode

🤖 Chatbot Configuration

Channel ID: 1440561306548703314
Max History: 20
Session Timeout: 1800s
Max Tokens: 500
Temperature: 0.7
Use RAG: ✅
RAG Threshold: 0.3
Rate Limit: 10/min
AI Provider: gemini
```

---

## Configuration

### Environment Variables

Configure the chatbot in your `.env` file:

```bash
# Required: Channel where chatbot operates
CHATBOT_CHANNEL_ID=1440561306548703314

# Session settings
CHATBOT_MAX_HISTORY=20              # Max messages to remember
CHATBOT_SESSION_TIMEOUT=1800        # Session timeout (seconds)

# AI settings
CHATBOT_MAX_TOKENS=500              # Max response length
CHATBOT_TEMPERATURE=0.7             # Creativity (0.0-1.0)

# RAG settings
CHATBOT_USE_RAG=true                # Enable RAG mode
CHATBOT_RAG_THRESHOLD=0.3           # Similarity threshold for RAG
CHATBOT_INCLUDE_CONTEXT_MESSAGES=5  # Recent messages for context

# Rate limiting
CHATBOT_RATE_LIMIT_MESSAGES=10      # Max messages per window
CHATBOT_RATE_LIMIT_WINDOW=60        # Window size (seconds)

# System prompt (optional)
CHATBOT_SYSTEM_PROMPT="You are a helpful Discord assistant..."
```

### Configuration Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHATBOT_CHANNEL_ID` | *required* | Discord channel ID where bot responds |
| `CHATBOT_MAX_HISTORY` | `20` | Max conversation messages to remember |
| `CHATBOT_SESSION_TIMEOUT` | `1800` | Session expiry time (30 minutes) |
| `CHATBOT_MAX_TOKENS` | `500` | Max tokens in bot response |
| `CHATBOT_TEMPERATURE` | `0.7` | Response creativity (0=factual, 1=creative) |
| `CHATBOT_USE_RAG` | `true` | Enable RAG for history search |
| `CHATBOT_RAG_THRESHOLD` | `0.3` | Min similarity for RAG results (0-1) |
| `CHATBOT_INCLUDE_CONTEXT_MESSAGES` | `5` | Recent channel messages for context |
| `CHATBOT_RATE_LIMIT_MESSAGES` | `10` | Messages allowed per window |
| `CHATBOT_RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |

---

## Usage Examples

### Example 1: General Conversation

```
You: Hi there!
Bot: Hello! How can I assist you today?

You: What can you do?
Bot: I can help with conversations and answer questions about past Discord
     messages. You can ask me about what people said, decisions that were made,
     or just chat naturally. Try asking me something!

You: How does Python work?
Bot: Python is an interpreted, high-level programming language. It uses an
     interpreter to execute code line by line...
```

**Mode:** Chat (no history search needed)

---

### Example 2: Asking About Past Conversations

```
You: What did we discuss about the database yesterday?
Bot: [RAG Mode - Searches Discord history]

Based on the conversation history, yesterday the team discussed:
- Migrating from SQLite to PostgreSQL
- Alice recommended PostgreSQL for better JSON support
- Bob mentioned concerns about migration complexity
- The team decided to start the migration next week
```

**Mode:** RAG (temporal reference "yesterday" + history keyword "discuss")

---

### Example 3: User-Specific Questions

```
You: @Alice what did you say about Docker?
Bot: [RAG Mode - Searches only Alice's messages]

Alice mentioned that she's been working on containerizing the application
with Docker. She's created Dockerfiles for both the frontend and backend,
and is working on a docker-compose setup for local development.
```

**Mode:** RAG (user mention triggers history search)

---

### Example 4: Decision Tracking

```
You: What was decided about the API design?
Bot: [RAG Mode - Searches for decision keywords]

Based on the conversation history:
- The team decided to use REST instead of GraphQL
- API versioning will use URL paths (e.g., /v1/users)
- Authentication will use JWT tokens
- Bob was assigned to draft the API specification
```

**Mode:** RAG (decision keyword "decided")

---

### Example 5: Follow-up Questions

```
You: Tell me about React hooks
Bot: React hooks are functions that let you use state and lifecycle features
     in functional components. The most common hooks are useState and useEffect...

You: Can you explain useState?  ← Bot remembers we're talking about React
Bot: useState is a hook that lets you add state to functional components.
     It returns a pair: the current state value and a function to update it.
     For example: const [count, setCount] = useState(0);

You: What about useEffect?  ← Still in context
Bot: useEffect lets you perform side effects in functional components...
```

**Mode:** Chat (conversational with session context)

---

## Rate Limits

### User Rate Limits

To prevent abuse and control costs:

- **Limit:** 10 messages per minute per user
- **Window:** Sliding 60-second window
- **Enforcement:** Automatic

### When You Hit the Limit

```
⏰ @You Rate limit exceeded. Please wait 0m 45s before sending another message.
```

**What to do:**
- Wait for the countdown
- Consider if your questions can be combined
- Use `!chatbot_reset` if you need a fresh start (doesn't bypass rate limit)

### Why Rate Limiting?

1. **Cost control:** AI API calls cost money
2. **Fair usage:** Ensures everyone can use the bot
3. **Quality:** Prevents spam and low-quality interactions

---

## FAQ

### Q: Why isn't the bot responding?

**A:** Check:
1. ✅ You're in the correct channel (chatbot channel)
2. ✅ Message doesn't start with `!` (commands bypass chatbot)
3. ✅ You haven't hit rate limit (10/minute)
4. ✅ Bot is online and running

### Q: How do I know if the bot is using RAG or Chat mode?

**A:**
- **RAG mode:** Responses reference "conversation history" or past messages
- **Chat mode:** Responses are general/conversational
- You can check the logs (admin) to see which mode was used

### Q: Can I see the sources the bot used?

**A:**
- **For `!ask` command:** React with 📚 to see sources
- **For chatbot:** Currently no, it's designed for natural conversation
- Consider using `!ask @User question` for source-visible answers

### Q: Does the bot remember everything I've ever said?

**A:** No:
- **Session memory:** Only remembers messages in your current session (max 20 by default)
- **Session timeout:** Expires after 30 minutes of inactivity
- **Reset:** You can reset anytime with `!chatbot_reset`
- **RAG search:** Can search ALL Discord history, not just your session

### Q: What's the difference between chatbot and `!ask`?

| Feature | Chatbot | `!ask` |
|---------|---------|--------|
| **Prefix** | None (natural) | `!ask` required |
| **Mode** | Auto (Chat/RAG) | Always RAG |
| **Context** | Session history | None (stateless) |
| **Sources** | Hidden | Visible (📚 reaction) |
| **Use case** | Conversation | Specific questions |

### Q: How much does it cost to use?

**A:**
- Check your stats with `!chatbot_stats`
- Costs vary by AI provider (OpenAI, Anthropic, Gemini)
- Admins can see aggregate costs with admin commands
- Typical cost: $0.001-0.01 per message

### Q: Can I use the bot in multiple channels?

**A:** No, the chatbot only works in the designated channel. This is intentional to:
- Prevent spam across the server
- Control costs
- Create a dedicated space for AI interaction

Use `!ask` in other channels for one-off questions.

---

## Troubleshooting

### Bot doesn't respond

**Symptoms:** You send a message, nothing happens

**Solutions:**
1. Check you're in the chatbot channel
2. Verify bot is online (check member list)
3. Make sure message doesn't start with `!`
4. Check rate limit (wait 1 minute and try again)

---

### Bot gives generic/wrong answers

**Symptoms:** Bot's answer doesn't match Discord history

**Possible causes:**
1. **Using Chat mode instead of RAG**
   - Solution: Add keywords like "what did we discuss" or mention users
2. **RAG threshold too high**
   - Solution: Admin can adjust `CHATBOT_RAG_THRESHOLD`
3. **Messages not indexed**
   - Solution: Admin should run `!ingest_channel` to index messages

**Workaround:** Use `!ask what did we discuss about X?` for guaranteed RAG search

---

### Rate limit errors

**Symptoms:** "⏰ Rate limit exceeded" message

**Solutions:**
1. Wait the specified time (shown in message)
2. Combine multiple questions into one message
3. If persistent, contact admin (may be misconfigured)

**Note:** Rate limit resets are automatic using a sliding window.

---

### Session seems confused

**Symptoms:** Bot brings up old topics or seems lost

**Solutions:**
1. **Reset session:** `!chatbot_reset`
2. **Be explicit:** Use full context instead of pronouns
3. **Use RAG mode:** Ask explicit questions about past messages

**Example:**
```
Bad:  "What about that?" (unclear context)
Good: "What did Alice say about the database?" (explicit)
```

---

### Bot says "I don't have enough context"

**Symptoms:** Bot can't find relevant information

**Possible causes:**
1. **Messages not indexed:** Use `!ingest_channel` (admin)
2. **Threshold too high:** Adjust `CHATBOT_RAG_THRESHOLD` (admin)
3. **No matching messages:** The information genuinely doesn't exist

**Workaround:** Try rephrasing your question or asking differently

---

## Advanced Usage

### Combining Chat and RAG

Ask a RAG question, then follow up conversationally:

```
You: What did Alice say about Redis?
Bot: [RAG mode] Alice mentioned Redis would be good for caching...

You: How does Redis work?  ← Switches to Chat mode
Bot: [Chat mode] Redis is an in-memory data store...

You: Should we use it?  ← Uses session context (remembers Redis)
Bot: [Chat mode] Based on what Alice mentioned, yes, Redis would be
     beneficial for your caching needs...
```

### Using with Other Commands

Chatbot works alongside other bot features:

```
You: What did we discuss about summaries?
Bot: [RAG mode] The team discussed implementing message summaries...

You: !summarize 50  ← Use summarize command
Bot: [Generates summary of last 50 messages]

You: Thanks, that's helpful!  ← Back to chatbot
Bot: [Chat mode] You're welcome! Let me know if you need anything else!
```

---

## Privacy & Data

### What does the bot remember?

1. **Session data (temporary):**
   - Your messages to the bot
   - Bot's responses to you
   - Expires after 30 minutes
   - Stored in memory (lost on restart)

2. **Usage statistics (permanent):**
   - Total messages sent
   - Tokens used
   - Cost
   - Social credit score
   - Stored in `data/ai_usage.db`

3. **Discord messages (permanent):**
   - All messages ingested with `!ingest_channel`
   - Stored in vector database for RAG
   - Used for answering questions about history

### Can others see my conversations?

- **No:** Each user has their own isolated session
- **Admin:** Can potentially access logs/databases
- **Discord:** Conversations happen in the public chatbot channel (visible to channel members)

---

## Tips & Best Practices

### 💡 Getting Better Answers

1. **Be specific:**
   - Bad: "What was said?"
   - Good: "What did Alice say about the deployment strategy?"

2. **Use mentions for user filtering:**
   - `@Alice what did you recommend?`

3. **Include time context:**
   - "What was decided yesterday?"
   - "What did we discuss last week?"

4. **Use decision keywords:**
   - "What was decided..."
   - "What did we agree on..."
   - "What was planned..."

### 💡 Managing Context

1. **Reset when switching topics:**
   ```
   !chatbot_reset
   What are the best practices for Docker?
   ```

2. **Be explicit, not vague:**
   - Bad: "What about that thing?"
   - Good: "What about the Redis implementation?"

3. **Break up complex questions:**
   ```
   You: What did we decide about the database?
   Bot: [Answers about database decision]

   You: And what about the API design?
   Bot: [Answers about API design]
   ```

### 💡 Cost Optimization

1. **Use `!ask` for one-off questions** (no session overhead)
2. **Reset sessions** when done to free memory
3. **Combine related questions** into one message
4. **Be concise** - shorter questions = fewer tokens

---

## Getting Help

### For Users

1. **Check this guide first**
2. **Try `!chatbot_stats`** to see your usage
3. **Try `!chatbot_reset`** if bot seems confused
4. **Ask admin** for configuration issues

### For Admins

1. **Check logs:** Bot logs all chatbot activity
2. **Validate config:** Ensure `.env` is correct
3. **Check ingestion:** Run `!ingest_channel` to index messages
4. **Adjust settings:** Modify `.env` and restart bot

---

## Related Documentation

- 📖 [Chatbot RAG Differentiation](./CHATBOT_RAG_DIFFERENTIATION.md) - How RAG vs Chat mode works
- 📖 [Chatbot Code Review](./CHATBOT_CODE_REVIEW.md) - Technical implementation details
- 📖 [RAG Pipeline](./RAG_PIPELINE.md) - How RAG search works (if exists)

---

## Changelog

### Current Version (v1.0)

**Features:**
- ✅ Natural conversation without command prefix
- ✅ Dual-mode (Chat/RAG) intelligence
- ✅ Session-based context
- ✅ User mention filtering
- ✅ Rate limiting (10/min)
- ✅ Usage tracking
- ✅ Background cleanup

**Commands:**
- `!chatbot_reset` - Reset session
- `!chatbot_stats` - View statistics
- `!chatbot_mode` - View config (admin)

---

**Last Updated:** 2025-11-19
**Author:** deep-bot development team
