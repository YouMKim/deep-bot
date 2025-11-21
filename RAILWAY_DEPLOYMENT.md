# Railway Deployment Guide

This guide explains how to deploy the Deep-Bot Discord bot to Railway.

## Quick Start

1. **Connect your GitHub repository** to Railway
2. **Set required environment variables** (see below)
3. **Deploy** - Railway will automatically build and deploy using the Dockerfile

## Required Environment Variables

### Core Configuration
```bash
# Discord Configuration (Required)
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CLIENT_ID=your_discord_client_id_here
CHATBOT_CHANNEL_ID=your_chatbot_channel_id

# AI Providers (At least one required)
OPENAI_API_KEY=sk-your_openai_api_key_here
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Bot Owner
BOT_OWNER_ID=your_discord_user_id_here
```

### Embedding Configuration
```bash
# Choose embedding provider (recommended: openai for Railway)
EMBEDDING_PROVIDER=openai  # or "sentence-transformers"

# Why openai is recommended for Railway:
# - openai: Small image size (~500MB), faster startup, requires API key
# - sentence-transformers: Large image size (~2GB+), slower, free but resource-intensive
```

### ChromaDB Configuration (Important!)
```bash
# If you encounter "KeyError: frozenset()" errors, set this to true
RESET_CHROMADB=true

# After successful deployment, you can set it back to false
# This will clear the vector database and require re-indexing messages
```

## Common Issues & Solutions

### 1. Tokenizers Package Error
**Error:** `ValueError: The tokenizers python package is not installed`

**Solution:** This is automatically fixed in the latest code. The bot now uses lazy imports for sentence-transformers.

**Alternative:** Set `EMBEDDING_PROVIDER=openai` to use OpenAI embeddings instead of sentence-transformers.

### 2. ChromaDB Frozenset Error
**Error:** `KeyError: frozenset()` or `Failed to load rag cog: Extension 'bot.cogs.rag' raised an error: KeyError: frozenset()`

**Solution:**
1. Set `RESET_CHROMADB=true` in Railway environment variables
2. Redeploy the bot
3. After successful deployment, optionally set it back to `false`

**Why this happens:** ChromaDB stores metadata that can become incompatible between versions. The auto-reset fixes this.

### 3. Docker Image Too Large
**Error:** Build succeeds but image is very large (2GB+)

**Solution:** Use OpenAI embeddings instead of sentence-transformers:
```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

This reduces the image size from ~2GB to ~500MB by excluding PyTorch and sentence-transformers.

## Deployment Checklist

- [ ] GitHub repository connected to Railway
- [ ] `DISCORD_TOKEN` set
- [ ] `DISCORD_CLIENT_ID` set
- [ ] `CHATBOT_CHANNEL_ID` set (if using chatbot feature)
- [ ] At least one AI provider API key set (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- [ ] `BOT_OWNER_ID` set
- [ ] `EMBEDDING_PROVIDER` set (recommended: `openai`)
- [ ] If encountering errors: `RESET_CHROMADB=true`

## Monitoring Deployment

1. Check Railway logs for startup messages
2. Look for successful cog loading:
   ```
   ✅ Loaded 1 blacklisted user IDs
   INFO - Loaded basic cog
   INFO - Loaded summary cog
   INFO - Loaded admin cog
   INFO - Loaded rag cog
   INFO - Loaded chatbot cog
   INFO - Loaded social credit commands cog
   INFO - Bot has connected to Discord
   ```

3. If any cogs fail to load, check the error messages and adjust environment variables accordingly

## Post-Deployment

### First Time Setup
After deployment, you'll need to index your Discord messages:

1. In Discord, run: `!fetch_messages`
2. Wait for the bot to fetch and index messages
3. Test RAG functionality: `!ask What did we discuss about X?`

### Persistent Data
Railway provides persistent storage for:
- `/data/raw_messages` - Stored Discord messages
- `/data/chroma_db` - Vector database (will be reset if `RESET_CHROMADB=true`)

**Note:** If you set `RESET_CHROMADB=true`, you'll need to re-fetch and re-index your messages.

## Troubleshooting

### Bot doesn't respond
- Check that the bot has proper permissions in your Discord server
- Verify `BOT_PREFIX` (default: `!`) in environment variables
- Check Railway logs for errors

### High memory usage
- Use `EMBEDDING_PROVIDER=openai` instead of `sentence-transformers`
- Reduce `RAG_DEFAULT_TOP_K` (default: 10, try: 5)
- Reduce `CHATBOT_MAX_HISTORY` (default: 20, try: 10)

### API cost concerns
- Monitor usage in OpenAI/Anthropic dashboards
- Set rate limits: `CHATBOT_RATE_LIMIT_MESSAGES` (default: 3)
- Disable expensive features:
  ```bash
  RAG_USE_MULTI_QUERY=false
  RAG_USE_HYDE=false
  RAG_USE_RERANKING=false
  ```

## Need Help?

1. Check Railway logs for detailed error messages
2. Review this deployment guide
3. Check the main README.md for general configuration
4. Open an issue on GitHub with:
   - Error messages from Railway logs
   - Environment variables (redact sensitive values)
   - Steps to reproduce the issue
