# Chatbot Feature Implementation Guide

## Overview

This document outlines the implementation plan for adding a RAG-enhanced conversational chatbot feature to Deep-Bot. The chatbot will operate in a dedicated Discord channel, maintaining conversation context and leveraging the existing RAG system to answer questions about past messages.

---

## Configuration

### Dedicated Chatbot Channel
**Channel ID:** `1440561306548703314`

### Rate Limiting
**Limit:** 10 messages per minute per user

---

## Current Code State Review

### ✅ Strengths

Your codebase is **production-ready** and well-architected for adding a chatbot feature:

1. **Clean Architecture** - Excellent separation of concerns (bot/, ai/, rag/, storage/)
2. **Multi-Provider AI Service** - Already supports OpenAI and Anthropic with hot-swapping (ai/service.py)
3. **Advanced RAG Pipeline** - 5-stage RAG with hybrid search, multi-query, HyDE, and reranking (rag/pipeline.py)
4. **Message Storage System** - SQLite + ChromaDB with chunking strategies (storage/)
5. **User Tracking** - AI usage tracking with costs, tokens, and "social credit" (ai/tracker.py)
6. **Rate Limiting Infrastructure** - Already implemented for other commands
7. **Discord.py 2.3.2+** - Modern async architecture with proper intents
8. **Configuration Management** - Centralized Config class with runtime updates

### 🎯 Existing Components to Leverage

- **AIService.generate()** (ai/service.py:124-161) - Generic AI completion method
- **RAGPipeline.answer_question()** (rag/pipeline.py) - Question answering with context
- **User Tracker** (ai/tracker.py) - Already tracks user AI usage
- **Message Loader** (bot/loaders/message_loader.py) - Fetches conversation history
- **Cooldown Decorator** - Rate limiting pattern from existing cogs
- **Config System** - Easy to add chatbot-specific settings

---

## What Needs to Be Built

### 1. Conversation Session Manager ⚠️ **HIGH PRIORITY**

**Purpose:** Track ongoing conversations with context history

**Requirements:**
- Store conversation history per user (last N messages)
- Manage context window (stay within token limits)
- Session timeout and cleanup (clear stale conversations)
- Thread-safe in-memory storage (dict-based cache)

**Implementation:** `bot/utils/session_manager.py`

**Features:**
- Per-user message history buffer
- Automatic context window trimming
- Session TTL (time-to-live) management
- Session reset functionality
- Format messages for AI context

### 2. Chatbot Cog ⚠️ **HIGH PRIORITY**

**Purpose:** Main chatbot logic and Discord event handling

**Requirements:**
- Listen to `on_message` events
- Filter messages (channel, bots, commands, blacklist)
- Integrate with SessionManager for context
- Use AIService for basic chat
- Use RAGPipeline for question answering
- Track costs and usage

**Implementation:** `bot/cogs/chatbot.py`

**Features:**
- Natural conversation (no command prefix needed)
- Typing indicator while generating response
- RAG integration for fact-based questions
- Rate limiting (10 messages/minute per user)
- Cost tracking per interaction
- Admin commands for management

### 3. Enhanced Configuration ⚠️ **MEDIUM PRIORITY**

**Purpose:** Centralized chatbot settings

**Requirements:**
- Add chatbot-specific config vars to `config.py`
- Support runtime configuration updates
- Validation for channel IDs

**Configuration Variables:**
```python
# Chatbot Configuration
CHATBOT_CHANNEL_ID: int                  # 1440561306548703314
CHATBOT_MAX_HISTORY: int                 # Messages to remember (default: 15)
CHATBOT_SESSION_TIMEOUT: int             # Session expiry in seconds (default: 1800)
CHATBOT_MAX_TOKENS: int                  # Max response length (default: 400)
CHATBOT_TEMPERATURE: float               # Conversational warmth (default: 0.8)
CHATBOT_SYSTEM_PROMPT: str               # Bot personality/behavior
CHATBOT_USE_RAG: bool                    # Enable RAG for questions (default: True)
CHATBOT_RAG_THRESHOLD: float             # Similarity threshold for RAG (default: 0.3)
CHATBOT_RATE_LIMIT_MESSAGES: int         # Messages per minute (default: 10)
CHATBOT_RATE_LIMIT_WINDOW: int           # Rate limit window in seconds (default: 60)
CHATBOT_INCLUDE_CONTEXT_MESSAGES: int    # Recent channel messages for context (default: 5)
```

### 4. Message Filtering & Preprocessing ⚠️ **MEDIUM PRIORITY**

**Purpose:** Clean and validate messages before processing

**Requirements:**
- Ignore bot messages
- Ignore command messages (starting with `!`)
- Check blacklist
- Sanitize input
- Detect question vs. statement

**Implementation:** Part of `bot/cogs/chatbot.py`

---

## Architecture Overview

### High-Level Flow

```
User sends message in #chatbot channel (1440561306548703314)
        ↓
Chatbot Cog (on_message listener)
        ↓
Filter: Channel check, bot check, command check, blacklist
        ↓
Rate limit check (10 messages/minute)
        ↓
SessionManager: Get/Create user session
        ↓
Detect message type: Question vs. Statement
        ↓
┌─────────────────┬─────────────────┐
│   Question?     │   Statement?    │
│   (RAG Mode)    │  (Chat Mode)    │
├─────────────────┼─────────────────┤
│ RAGPipeline     │ AIService       │
│ Search history  │ Generate reply  │
│ Build context   │ with history    │
│ Generate answer │ context         │
└─────────────────┴─────────────────┘
        ↓
Send reply to Discord (with typing indicator)
        ↓
Update session history (add user message + bot reply)
        ↓
Track usage (tokens, cost, social credit)
```

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────┐
│            Discord (Channel Event)              │
└───────────────────┬─────────────────────────────┘
                    │ on_message
                    ↓
┌─────────────────────────────────────────────────┐
│         Chatbot Cog (bot/cogs/chatbot.py)       │
│  - Message filtering                            │
│  - Rate limiting                                │
│  - Question detection                           │
│  - Response generation                          │
└────┬────────────────────────────────────────┬───┘
     │                                        │
     │ Get/Update Session                     │ Answer Question
     ↓                                        ↓
┌──────────────────────────┐    ┌──────────────────────────┐
│    SessionManager        │    │      RAGPipeline         │
│  (bot/utils/session_     │    │   (rag/pipeline.py)      │
│   manager.py)            │    │  - Hybrid search         │
│  - Conversation history  │    │  - Multi-query           │
│  - Context building      │    │  - HyDE                  │
│  - Session timeout       │    │  - Reranking             │
│  - TTL management        │    │  - Answer generation     │
└──────────────────────────┘    └────────┬─────────────────┘
                                         │
                                         │ Use for completion
     ┌───────────────────────────────────┘
     │ Generate Response
     ↓
┌──────────────────────────────────────────────────┐
│           AIService (ai/service.py)              │
│  - OpenAI/Anthropic provider                     │
│  - Temperature normalization                     │
│  - Cost calculation                              │
│  - Token counting                                │
└────────┬─────────────────────────────────────────┘
         │
         │ Track usage
         ↓
┌──────────────────────────────────────────────────┐
│         AIUsageTracker (ai/tracker.py)           │
│  - Per-user cost tracking                        │
│  - Token usage                                   │
│  - Social credit scores                          │
└──────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core Infrastructure ✅ **FOUNDATION**

#### 1.1 Update Configuration (config.py)

Add chatbot-specific configuration variables:

```python
# Chatbot Configuration
CHATBOT_CHANNEL_ID: int = int(os.getenv("CHATBOT_CHANNEL_ID", "1440561306548703314"))
CHATBOT_MAX_HISTORY: int = int(os.getenv("CHATBOT_MAX_HISTORY", "15"))
CHATBOT_SESSION_TIMEOUT: int = int(os.getenv("CHATBOT_SESSION_TIMEOUT", "1800"))  # 30 minutes
CHATBOT_MAX_TOKENS: int = int(os.getenv("CHATBOT_MAX_TOKENS", "400"))
CHATBOT_TEMPERATURE: float = float(os.getenv("CHATBOT_TEMPERATURE", "0.8"))
CHATBOT_USE_RAG: bool = os.getenv("CHATBOT_USE_RAG", "True").lower() == "true"
CHATBOT_RAG_THRESHOLD: float = float(os.getenv("CHATBOT_RAG_THRESHOLD", "0.3"))
CHATBOT_RATE_LIMIT_MESSAGES: int = int(os.getenv("CHATBOT_RATE_LIMIT_MESSAGES", "10"))
CHATBOT_RATE_LIMIT_WINDOW: int = int(os.getenv("CHATBOT_RATE_LIMIT_WINDOW", "60"))
CHATBOT_INCLUDE_CONTEXT_MESSAGES: int = int(os.getenv("CHATBOT_INCLUDE_CONTEXT_MESSAGES", "5"))

# Chatbot system prompt (defines personality)
CHATBOT_SYSTEM_PROMPT: str = os.getenv(
    "CHATBOT_SYSTEM_PROMPT",
    """You are a helpful and friendly Discord chatbot assistant. You have access to past conversation history
    and can answer questions about what was discussed. Be conversational, concise, and engaging.
    When answering factual questions, rely on the provided context from past messages.
    If you don't have enough context to answer a question, say so politely."""
)
```

#### 1.2 Create Session Manager (bot/utils/session_manager.py)

**Responsibilities:**
- Store conversation sessions (user_id → session data)
- Maintain message history per user
- Manage context window (trim to fit token limits)
- Session TTL and cleanup
- Format messages for AI context

**Data Structure:**
```python
Session = {
    "user_id": int,
    "channel_id": int,
    "messages": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."}
    ],
    "created_at": datetime,
    "last_activity": datetime
}
```

**Key Methods:**
```python
class SessionManager:
    def __init__(self, max_history: int, session_timeout: int):
        """Initialize session manager with config."""

    def get_session(self, user_id: int, channel_id: int) -> dict:
        """Get or create session for user."""

    def add_message(self, user_id: int, role: str, content: str):
        """Add message to user's session history."""

    def get_history(self, user_id: int) -> List[dict]:
        """Get formatted message history for user."""

    def reset_session(self, user_id: int):
        """Clear user's conversation history."""

    def cleanup_expired_sessions(self):
        """Remove sessions older than timeout."""

    def format_for_ai(self, user_id: int, current_message: str) -> str:
        """Format history + current message as AI prompt."""
```

#### 1.3 Create Chatbot Cog (bot/cogs/chatbot.py)

**Core Structure:**

```python
class Chatbot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = Config
        self.session_manager = SessionManager(
            max_history=Config.CHATBOT_MAX_HISTORY,
            session_timeout=Config.CHATBOT_SESSION_TIMEOUT
        )
        self.ai_service = AIService(provider_name=Config.AI_DEFAULT_PROVIDER)
        self.rag_pipeline = RAGPipeline(config=Config)
        self.user_rate_limits = {}  # Track per-user message timestamps

    @commands.Cog.listener()
    async def on_message(self, message):
        """Main chatbot event listener."""
        # 1. Filter: channel, bots, commands, blacklist
        # 2. Rate limit check
        # 3. Get session
        # 4. Detect question vs. statement
        # 5. Generate response (RAG or chat)
        # 6. Send reply
        # 7. Update session

    async def _is_question(self, text: str) -> bool:
        """Detect if message is a question."""

    async def _generate_rag_response(self, question: str, user_id: int) -> str:
        """Use RAG to answer question with context."""

    async def _generate_chat_response(self, message: str, user_id: int) -> str:
        """Generate conversational response with history."""

    def _check_rate_limit(self, user_id: int) -> tuple[bool, float]:
        """Check if user has exceeded rate limit."""

    @commands.command(name='chatbot_reset')
    async def reset_conversation(self, ctx):
        """Reset your conversation history."""

    @commands.command(name='chatbot_stats')
    async def chatbot_stats(self, ctx):
        """View your chatbot usage statistics."""
```

### Phase 2: RAG Integration ✅ **INTELLIGENCE**

#### 2.1 Question Detection

**Method:** Simple heuristics + keyword matching

**Implementation:**
```python
async def _is_question(self, text: str) -> bool:
    """
    Detect if message is a question that should use RAG.

    Criteria:
    - Starts with question word (what, when, where, who, why, how)
    - Ends with '?'
    - Contains question keywords (tell me, explain, what about)
    """
    text_lower = text.lower().strip()

    # Direct question indicators
    question_starters = ['what', 'when', 'where', 'who', 'why', 'how', 'did', 'does', 'is', 'are', 'was', 'were', 'can', 'could', 'would', 'should']
    question_phrases = ['tell me', 'explain', 'what about', 'how about', 'do you know']

    # Check for question mark
    if text_lower.endswith('?'):
        return True

    # Check for question starters
    first_word = text_lower.split()[0] if text_lower.split() else ""
    if first_word in question_starters:
        return True

    # Check for question phrases
    for phrase in question_phrases:
        if phrase in text_lower:
            return True

    return False
```

#### 2.2 RAG Response Generation

**Strategy:** Use existing RAGPipeline with conversation-aware context

**Implementation:**
```python
async def _generate_rag_response(self, question: str, user_id: int, mentioned_users: List[str] = None) -> dict:
    """
    Use RAG pipeline to answer questions with context from past messages.

    Args:
        question: User's question
        user_id: User ID for tracking
        mentioned_users: Optional list of @mentioned users to filter

    Returns:
        dict with answer, sources, cost, etc.
    """
    # Build RAG config
    config = RAGConfig(
        top_k=self.config.RAG_DEFAULT_TOP_K,
        similarity_threshold=self.config.CHATBOT_RAG_THRESHOLD,
        max_context_tokens=self.config.RAG_DEFAULT_MAX_CONTEXT_TOKENS,
        temperature=self.config.CHATBOT_TEMPERATURE,
        strategy=self.config.RAG_DEFAULT_STRATEGY,
        filter_authors=mentioned_users,
        use_hybrid_search=self.config.RAG_USE_HYBRID_SEARCH,
        use_multi_query=self.config.RAG_USE_MULTI_QUERY,
        use_hyde=self.config.RAG_USE_HYDE,
        use_reranking=self.config.RAG_USE_RERANKING,
        max_output_tokens=self.config.CHATBOT_MAX_TOKENS,
    )

    # Get answer from RAG pipeline
    result = await self.rag_pipeline.answer_question(question, config)

    return {
        'content': result.answer,
        'sources': result.sources,
        'cost': result.cost,
        'model': result.model,
        'mode': 'rag'
    }
```

#### 2.3 Conversational Response Generation

**Strategy:** Use AIService.generate() with conversation history context

**Implementation:**
```python
async def _generate_chat_response(self, message: str, user_id: int) -> dict:
    """
    Generate conversational response using chat history.

    Args:
        message: User's message
        user_id: User ID for session tracking

    Returns:
        dict with content, cost, model, etc.
    """
    # Get conversation history
    history = self.session_manager.get_history(user_id)

    # Build prompt with system message + history + current message
    prompt_parts = [self.config.CHATBOT_SYSTEM_PROMPT]

    # Add conversation history
    if history:
        prompt_parts.append("\n\nConversation history:")
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}")

    # Add current message
    prompt_parts.append(f"\n\nUser: {message}\nAssistant:")

    prompt = "\n".join(prompt_parts)

    # Generate response
    result = await self.ai_service.generate(
        prompt=prompt,
        max_tokens=self.config.CHATBOT_MAX_TOKENS,
        temperature=self.config.CHATBOT_TEMPERATURE
    )

    return {
        'content': result['content'],
        'cost': result['cost'],
        'model': result['model'],
        'mode': 'chat'
    }
```

### Phase 3: Advanced Features ✅ **POLISH**

#### 3.1 Channel Context Integration

**Enhancement:** Include recent channel messages for better context

**Implementation:**
```python
async def _get_recent_channel_context(self, channel_id: int, limit: int = 5) -> str:
    """
    Fetch recent messages from channel for additional context.

    Args:
        channel_id: Discord channel ID
        limit: Number of recent messages to include

    Returns:
        Formatted string of recent messages
    """
    channel = self.bot.get_channel(channel_id)
    if not channel:
        return ""

    messages = []
    async for msg in channel.history(limit=limit):
        if not msg.author.bot and not msg.content.startswith('!'):
            messages.append(f"{msg.author.display_name}: {msg.content}")

    messages.reverse()  # Chronological order

    if messages:
        return "Recent channel context:\n" + "\n".join(messages) + "\n\n"
    return ""
```

#### 3.2 Admin Commands

**Commands for managing chatbot:**

```python
@commands.command(name='chatbot_reset', help='Reset your conversation history')
async def reset_conversation(self, ctx):
    """Allow users to reset their conversation history."""
    user_id = ctx.author.id
    self.session_manager.reset_session(user_id)
    await ctx.send(f"✅ {ctx.author.mention} Your conversation history has been cleared!")

@commands.command(name='chatbot_stats', help='View your chatbot usage statistics')
async def chatbot_stats(self, ctx):
    """Show user's chatbot usage statistics."""
    user_id = ctx.author.id
    session = self.session_manager.get_session(user_id, ctx.channel.id)

    embed = discord.Embed(
        title=f"📊 Chatbot Statistics for {ctx.author.display_name}",
        color=discord.Color.blue()
    )

    message_count = len(session.get('messages', []))
    created_at = session.get('created_at', 'N/A')
    last_activity = session.get('last_activity', 'N/A')

    embed.add_field(name="Messages in Session", value=str(message_count), inline=True)
    embed.add_field(name="Session Started", value=str(created_at), inline=True)
    embed.add_field(name="Last Activity", value=str(last_activity), inline=True)

    # Get user's total AI usage from tracker
    from ai.tracker import AIUsageTracker
    tracker = AIUsageTracker()
    stats = tracker.get_user_stats(user_id)

    if stats:
        embed.add_field(name="Total Cost", value=f"${stats['total_cost']:.4f}", inline=True)
        embed.add_field(name="Total Tokens", value=f"{stats['total_tokens']:,}", inline=True)
        embed.add_field(name="Social Credit", value=f"{stats['social_credit']:.2f}", inline=True)

    await ctx.send(embed=embed)

@commands.command(name='chatbot_mode', help='Check current chatbot settings (Admin only)')
async def chatbot_mode(self, ctx):
    """Display current chatbot configuration."""
    if str(ctx.author.id) != str(self.config.BOT_OWNER_ID):
        await ctx.send("🚫 This command is admin-only!")
        return

    embed = discord.Embed(
        title="🤖 Chatbot Configuration",
        color=discord.Color.green()
    )

    embed.add_field(name="Channel ID", value=str(self.config.CHATBOT_CHANNEL_ID), inline=True)
    embed.add_field(name="Max History", value=str(self.config.CHATBOT_MAX_HISTORY), inline=True)
    embed.add_field(name="Session Timeout", value=f"{self.config.CHATBOT_SESSION_TIMEOUT}s", inline=True)
    embed.add_field(name="Max Tokens", value=str(self.config.CHATBOT_MAX_TOKENS), inline=True)
    embed.add_field(name="Temperature", value=str(self.config.CHATBOT_TEMPERATURE), inline=True)
    embed.add_field(name="Use RAG", value="✅" if self.config.CHATBOT_USE_RAG else "❌", inline=True)
    embed.add_field(name="RAG Threshold", value=str(self.config.CHATBOT_RAG_THRESHOLD), inline=True)
    embed.add_field(name="Rate Limit", value=f"{self.config.CHATBOT_RATE_LIMIT_MESSAGES}/min", inline=True)

    await ctx.send(embed=embed)
```

#### 3.3 Enhanced Error Handling

**Graceful degradation and user feedback:**

```python
async def on_message(self, message):
    """Main chatbot listener with comprehensive error handling."""
    try:
        # ... main logic ...
    except Exception as e:
        self.logger.error(f"Chatbot error: {e}", exc_info=True)

        # Send user-friendly error message
        await message.channel.send(
            f"❌ Sorry {message.author.mention}, I encountered an error processing your message. "
            f"Please try again or use `!chatbot_reset` to clear your session."
        )
```

#### 3.4 Typing Indicators & Reactions

**Better user experience:**

```python
async def on_message(self, message):
    # ... filtering logic ...

    # Show typing indicator while processing
    async with message.channel.typing():
        # React to show message received
        await message.add_reaction("👀")

        # Generate response
        response = await self._generate_response(...)

        # Remove "eyes" reaction, add "check" reaction
        await message.remove_reaction("👀", self.bot.user)
        await message.add_reaction("✅")

        # Send response
        await message.channel.send(response['content'])
```

#### 3.5 Session Cleanup Task

**Background task to clean up expired sessions:**

```python
from discord.ext import tasks

class Chatbot(commands.Cog):
    def __init__(self, bot):
        # ... existing init ...
        self.cleanup_sessions.start()  # Start background task

    def cog_unload(self):
        """Clean up when cog is unloaded."""
        self.cleanup_sessions.cancel()

    @tasks.loop(minutes=10)  # Run every 10 minutes
    async def cleanup_sessions(self):
        """Remove expired sessions to free memory."""
        self.session_manager.cleanup_expired_sessions()
        self.logger.info("Cleaned up expired chatbot sessions")

    @cleanup_sessions.before_loop
    async def before_cleanup(self):
        """Wait until bot is ready before starting cleanup task."""
        await self.bot.wait_until_ready()
```

---

## File Structure Summary

```
deep-bot/
├── bot/
│   ├── cogs/
│   │   ├── basic.py           # Existing
│   │   ├── summary.py         # Existing
│   │   ├── admin.py           # Existing
│   │   ├── rag.py             # Existing
│   │   └── chatbot.py         # 🆕 NEW - Conversational chatbot
│   └── utils/
│       ├── discord_utils.py   # Existing
│       └── session_manager.py # 🆕 NEW - Conversation session management
├── config.py                   # 🔧 MODIFY - Add chatbot config vars
├── bot.py                      # 🔧 MODIFY - Load chatbot cog
└── docs/
    └── CHATBOT_IMPLEMENTATION.md  # This file
```

---

## Implementation Checklist

### Core Implementation
- [ ] Add chatbot configuration variables to `config.py`
- [ ] Create `bot/utils/session_manager.py`
  - [ ] Session data structure
  - [ ] Get/create session
  - [ ] Add message to history
  - [ ] Format history for AI
  - [ ] Session timeout cleanup
  - [ ] Reset session
- [ ] Create `bot/cogs/chatbot.py`
  - [ ] `on_message` listener
  - [ ] Message filtering (channel, bots, commands, blacklist)
  - [ ] Rate limiting (10 messages/minute)
  - [ ] Question detection
  - [ ] RAG response generation
  - [ ] Chat response generation
  - [ ] Session updates
  - [ ] Cost tracking
- [ ] Update `bot.py` to load chatbot cog
- [ ] Update `.env.example` with chatbot config

### Advanced Features
- [ ] Channel context integration (recent messages)
- [ ] Admin commands
  - [ ] `!chatbot_reset` - Reset conversation
  - [ ] `!chatbot_stats` - View usage stats
  - [ ] `!chatbot_mode` - View config (admin only)
- [ ] Enhanced error handling
- [ ] Typing indicators and reactions
- [ ] Background session cleanup task
- [ ] @mention support in RAG queries

### Testing
- [ ] Test basic conversation flow
- [ ] Test RAG integration (questions)
- [ ] Test rate limiting
- [ ] Test session management (creation, timeout, reset)
- [ ] Test with different AI providers (OpenAI, Anthropic)
- [ ] Test cost tracking
- [ ] Test error handling
- [ ] Test admin commands
- [ ] Load test (multiple users, concurrent messages)

### Documentation
- [ ] Update README.md with chatbot feature
- [ ] Add .env.example entries for chatbot config
- [ ] Document chatbot commands
- [ ] Add usage examples

---

## Configuration Reference

### Environment Variables (.env)

```bash
# Chatbot Configuration
CHATBOT_CHANNEL_ID=1440561306548703314
CHATBOT_MAX_HISTORY=15
CHATBOT_SESSION_TIMEOUT=1800
CHATBOT_MAX_TOKENS=400
CHATBOT_TEMPERATURE=0.8
CHATBOT_USE_RAG=True
CHATBOT_RAG_THRESHOLD=0.3
CHATBOT_RATE_LIMIT_MESSAGES=10
CHATBOT_RATE_LIMIT_WINDOW=60
CHATBOT_INCLUDE_CONTEXT_MESSAGES=5
CHATBOT_SYSTEM_PROMPT="You are a helpful and friendly Discord chatbot assistant..."
```

### Runtime Adjustments

Settings can be modified at runtime (resets on bot restart):
- Temperature, max tokens, RAG settings already support runtime updates via existing config methods

---

## Usage Examples

### Basic Conversation
```
User: Hey bot, how are you?
Bot: I'm doing great, thanks for asking! How can I help you today?

User: What's the weather like?
Bot: I don't have access to real-time weather data, but I can chat about it! What's your location?
```

### RAG-Enhanced Questions
```
User: What did Alice say about the database yesterday?
Bot: [Searches message history via RAG]
Based on the conversation history, Alice mentioned that the database migration is scheduled for Friday at 5pm and should take about 2 hours.

User: @Bob what did you decide?
Bot: [Filters to Bob's messages and searches]
Bob decided to use PostgreSQL for the new project and will set it up this week.
```

### Session Management
```
User: !chatbot_reset
Bot: ✅ @User Your conversation history has been cleared!

User: !chatbot_stats
Bot: [Shows embed with session info, message count, costs, social credit]
```

---

## Performance Considerations

### Token Usage
- **Session history:** ~100-200 tokens per user
- **Channel context:** ~50-100 tokens for 5 recent messages
- **RAG context:** ~500-2000 tokens depending on chunks retrieved
- **Response:** ~100-400 tokens
- **Average per interaction:** 500-1000 tokens (chat mode), 1000-3000 tokens (RAG mode)

### Cost Estimates (OpenAI GPT-4)
- **Chat mode:** ~$0.01-0.02 per interaction
- **RAG mode:** ~$0.02-0.05 per interaction
- **10 messages/min × 60 min × $0.02 = ~$12/hour worst case per user**

### Rate Limiting Strategy
- 10 messages/minute per user prevents runaway costs
- Typical usage: 5-10 messages/hour per user = ~$0.50/hour per active user

### Memory Usage
- Session storage: ~1KB per user session
- 1000 active users = ~1MB memory (negligible)
- Automatic cleanup of expired sessions keeps memory low

---

## Security Considerations

### Input Validation
- Sanitize user messages before processing
- Limit message length (Discord's 2000 char limit already enforced)
- Check blacklist before processing

### Rate Limiting
- Per-user rate limiting prevents spam
- Channel-specific rate limiting prevents channel abuse
- Exponential backoff for repeated violations (future enhancement)

### Cost Controls
- Max tokens per response (400 default)
- Session timeout prevents stale memory consumption
- RAG similarity threshold prevents irrelevant searches
- Daily/weekly cost caps (future enhancement)

### Privacy
- Sessions stored in-memory only (no persistent storage of conversations)
- Sessions expire after 30 minutes of inactivity
- Users can manually clear history with `!chatbot_reset`

---

## Future Enhancements

### Potential Additions
1. **Multi-channel support** - Allow chatbot in multiple channels with separate sessions
2. **Personality modes** - Switch between casual, professional, funny personalities
3. **Voice integration** - Respond to voice channel transcripts
4. **Image understanding** - Analyze attached images (GPT-4 Vision)
5. **Proactive engagement** - Bot initiates conversations based on triggers
6. **Memory persistence** - Optional long-term memory across sessions
7. **Group context** - Track channel-wide conversation instead of per-user
8. **Custom RAG strategies** - Per-user or per-channel chunking strategies
9. **Cost analytics dashboard** - Web UI for usage monitoring
10. **Fine-tuned responses** - Train custom model on server's conversation style

---

## Troubleshooting

### Common Issues

**Issue:** Bot not responding in chatbot channel
- Check `CHATBOT_CHANNEL_ID` is correct
- Verify bot has `MESSAGE_CONTENT` intent enabled
- Check bot permissions in channel (Read Messages, Send Messages)
- Check logs for errors

**Issue:** Rate limit errors
- Verify `CHATBOT_RATE_LIMIT_MESSAGES` and `CHATBOT_RATE_LIMIT_WINDOW` are set
- Check if user is hitting rate limit (expected behavior)
- Consider increasing limits for trusted users

**Issue:** RAG not finding relevant context
- Lower `CHATBOT_RAG_THRESHOLD` (default: 0.3 → try 0.2)
- Check if messages have been indexed (run `!load_channel` first)
- Verify chunking has completed (`!chunk_status`)
- Try different chunking strategies

**Issue:** High API costs
- Lower `CHATBOT_MAX_TOKENS` (400 → 200)
- Lower `CHATBOT_MAX_HISTORY` (15 → 10)
- Disable RAG temporarily (`CHATBOT_USE_RAG=False`)
- Check `!mystats` to see per-user costs
- Consider switching to cheaper model (GPT-3.5 instead of GPT-4)

**Issue:** Sessions not expiring
- Check `CHATBOT_SESSION_TIMEOUT` is set correctly
- Verify cleanup task is running (check logs every 10 minutes)
- Manually restart bot to clear all sessions

---

## Conclusion

This implementation plan provides a **production-ready, RAG-enhanced conversational chatbot** that seamlessly integrates with your existing Deep-Bot architecture. By leveraging the existing AIService, RAGPipeline, and user tracking systems, we can add sophisticated conversational AI with minimal code duplication.

The chatbot will:
- ✅ Operate in dedicated channel (1440561306548703314)
- ✅ Maintain conversation context per user
- ✅ Answer questions using RAG with message history
- ✅ Track costs and usage
- ✅ Respect rate limits (10 messages/minute)
- ✅ Integrate seamlessly with existing codebase

**Estimated development time:** 4-6 hours for core implementation + 2-3 hours for testing and polish.

Ready to build! 🚀
