# Railway Deployment Guide

This guide walks you through deploying Deep-Bot to Railway's free tier using Docker.

## Prerequisites

1. A Railway account (sign up at [railway.app](https://railway.app))
2. A GitHub account with your bot repository
3. Discord bot token and API keys ready

## Step 1: Prepare Your Repository

Ensure your repository has:
- `Dockerfile` (already created)
- `.dockerignore` (already created)
- `railway.json` (already created)
- `requirements.txt` (already exists)
- `bot.py` (main entry point)

## Step 2: Connect to Railway

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Authorize Railway to access your GitHub account
5. Select your `deep-bot` repository
6. Railway will automatically detect the Dockerfile and start building

## Step 3: Configure Environment Variables

In Railway dashboard, go to your project → Variables tab and add the following:

### Required Variables

**Discord Configuration:**
```
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CLIENT_ID=your_discord_client_id
BOT_OWNER_ID=your_discord_user_id
```

**AI Provider Keys (at least one required):**
```
OPENAI_API_KEY=your_openai_api_key
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key
# OR
GEMINI_API_KEY=your_gemini_api_key
```

**Default AI Provider:**
```
AI_DEFAULT_PROVIDER=openai
```

### Optional Configuration Variables

**Bot Settings:**
```
BOT_PREFIX=!
LOG_LEVEL=INFO
DEBUG_MODE=False
TEST_MODE=False
```

**OpenAI Configuration:**
```
OPENAI_DEFAULT_MODEL=gpt-5-mini-2025-08-07
```

**Anthropic Configuration:**
```
ANTHROPIC_DEFAULT_MODEL=claude-haiku-4-5
```

**Gemini Configuration:**
```
GEMINI_DEFAULT_MODEL=gemini-2.5-flash
```

**RAG Configuration:**
```
RAG_DEFAULT_TOP_K=8
RAG_DEFAULT_TEMPERATURE=0.7
RAG_MAX_OUTPUT_TOKENS=1200
RAG_USE_HYBRID_SEARCH=True
RAG_USE_MULTI_QUERY=True
RAG_USE_HYDE=True
RAG_USE_RERANKING=True
```

**Chatbot Configuration:**
```
CHATBOT_CHANNEL_ID=0
CHATBOT_MAX_HISTORY=15
CHATBOT_USE_RAG=True
CHATBOT_TEMPERATURE=0.8
```

**Social Credit Configuration:**
```
SOCIAL_CREDIT_ENABLED=True
SOCIAL_CREDIT_INITIAL_MEAN=0
SOCIAL_CREDIT_INITIAL_STD=200
```

**Evaluate Command:**
```
EVALUATE_ENABLED=True
EVALUATE_MAX_TOKENS=800
EVALUATE_TEMPERATURE=0.3
```

**Blacklist:**
```
BLACKLIST_IDS=123456789,987654321
```

**Advanced Settings (optional, defaults work fine):**
```
MESSAGE_FETCH_DELAY=0.2
MESSAGE_FETCH_BATCH_SIZE=100
CHUNKING_DEFAULT_STRATEGIES=single,tokens,author
VECTOR_STORE_PROVIDER=chroma
EMBEDDING_BATCH_SIZE=100
```

## Step 4: Configure Persistent Storage

Railway provides persistent volumes that survive container restarts. You need to mount volumes for:

1. **SQLite Databases** (`/app/data/raw_messages/`)
2. **ChromaDB Vector Store** (`/app/data/chroma_db/`)
3. **ML Model Cache** (`/root/.cache/huggingface/`)

### Setting Up Volumes in Railway

1. In Railway dashboard, go to your service
2. Click on "Settings" → "Volumes"
3. Add the following volume mounts:

   **Volume 1:**
   - Mount Path: `/app/data`
   - Description: SQLite databases and ChromaDB storage

   **Volume 2:**
   - Mount Path: `/root/.cache/huggingface`
   - Description: ML model cache (sentence-transformers)

**Note:** Railway will create these volumes automatically. The first deployment will download ML models (~100MB+), which may take a few minutes.

## Step 5: Deploy

1. Railway will automatically build and deploy when you push to your connected branch
2. Or manually trigger a deployment from the Railway dashboard
3. Watch the build logs to ensure everything builds correctly
4. Once deployed, check the logs to see if the bot connects to Discord

## Step 6: Verify Deployment

1. Check Railway logs for:
   - "Bot has connected to Discord"
   - "ChromaDB client initialized successfully"
   - No error messages

2. In Discord, test a command like `!help` to verify the bot is responding

3. Check that data persists:
   - Run `!ask` command
   - Restart the service in Railway
   - Verify data still exists (collections persist)

## Troubleshooting

### Bot Won't Start

**Check:**
- All required environment variables are set
- `DISCORD_TOKEN` is valid
- At least one AI provider key is set
- Check Railway logs for specific error messages

### Database/Storage Issues

**Symptoms:** Data disappears after restart

**Solution:**
- Verify volumes are mounted correctly
- Check volume paths match Dockerfile paths
- Ensure volumes have write permissions

### ML Models Not Downloading

**Symptoms:** Slow first startup, embedding errors

**Solution:**
- First deployment downloads models (~100MB+)
- Check `/root/.cache/huggingface/` volume is mounted
- Models cache persists across restarts

### Build Failures

**Common Issues:**
- Missing dependencies in `requirements.txt`
- Dockerfile syntax errors
- Build timeout (increase in Railway settings)

**Solution:**
- Test Docker build locally: `docker build -t deep-bot .`
- Check Railway build logs for specific errors
- Ensure `.dockerignore` excludes unnecessary files

### High Memory Usage

**Symptoms:** Service crashes or restarts frequently

**Solution:**
- Railway free tier has memory limits
- Consider upgrading to paid tier if needed
- Reduce `RAG_DEFAULT_TOP_K` to use less memory
- Disable unused RAG features if needed

### Logs Not Showing

**Note:** File logging (`bot.log`) may not persist in Railway. All logs go to stdout/stderr and are visible in Railway dashboard.

## Monitoring

Railway provides:
- Real-time logs in dashboard
- Deployment history
- Resource usage metrics
- Automatic restarts on failure

## Updating the Bot

1. Push changes to your GitHub repository
2. Railway automatically detects changes and redeploys
3. Or manually trigger redeploy from Railway dashboard
4. Check logs to verify new version is running

## Cost Considerations

**Railway Free Tier:**
- $5 credit per month
- Sufficient for small Discord bots
- Pay-as-you-go after free credit

**Estimated Costs:**
- Small bot: ~$0-5/month (within free tier)
- Medium bot: ~$5-15/month
- Large bot: ~$15+/month

Monitor usage in Railway dashboard to track costs.

## Security Best Practices

1. **Never commit `.env` files** - Use Railway environment variables
2. **Rotate API keys regularly**
3. **Use Railway's secret management** - Variables are encrypted at rest
4. **Limit `BOT_OWNER_ID`** - Only your Discord user ID
5. **Set `DEBUG_MODE=False`** in production

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Check bot logs in Railway dashboard for errors

