# Memory Optimization Guide

This guide explains how to reduce memory usage by disabling PyTorch and using OpenAI embeddings instead of local sentence-transformers models.

## Memory Savings

By disabling PyTorch and using OpenAI embeddings:
- **Saves ~750MB** of memory
- **Reduces Docker image size** by ~500MB
- **Faster startup** (no model loading)

## Configuration Steps

### 1. Set Environment Variables

In your Railway project (or `.env` file), set these environment variables:

```bash
# Use OpenAI embeddings instead of sentence-transformers
EMBEDDING_PROVIDER=openai

# Disable reranking (requires sentence-transformers)
RAG_USE_RERANKING=False
```

### 2. Build Docker Image

**PyTorch is disabled by default** (saves ~750MB memory). No build args needed!

If you want to **enable PyTorch** (for local embeddings or reranking), pass the build argument:

**For Railway CLI:**
```bash
railway up --build-arg INSTALL_PYTORCH=1
```

**For local Docker build:**
```bash
# Default (PyTorch disabled - recommended)
docker build -t deep-bot .

# To enable PyTorch
docker build --build-arg INSTALL_PYTORCH=1 -t deep-bot .
```

**For docker-compose:**
```yaml
services:
  bot:
    build:
      context: .
      # args:
      #   INSTALL_PYTORCH: 1  # Only uncomment if you need PyTorch
```

### 3. Verify Configuration

After deployment, check your logs. You should see:
- `Skipping PyTorch/sentence-transformers installation (using OpenAI embeddings + disabled reranking)` during build
- `PyTorch dependencies skipped (using OpenAI embeddings)` during runtime
- No errors about missing sentence-transformers
- Bot starts successfully

## What Gets Disabled

When `INSTALL_PYTORCH=0`:
- ❌ PyTorch (~500MB)
- ❌ sentence-transformers (~200MB)
- ❌ tokenizers (~50MB)
- ✅ OpenAI embeddings (via API)
- ✅ ChromaDB (still works)
- ✅ All other features

## Requirements

To use this optimization:
1. ✅ Must have `OPENAI_API_KEY` set
2. ✅ Must set `EMBEDDING_PROVIDER=openai`
3. ✅ Must set `RAG_USE_RERANKING=False`
4. ✅ Must build with `INSTALL_PYTORCH=0`

## Trade-offs

**Pros:**
- ~750MB less memory usage
- Smaller Docker image
- Faster startup
- No local model loading

**Cons:**
- Requires OpenAI API key
- API costs for embeddings (~$0.13 per 1M tokens)
- Requires internet connection for embeddings
- No reranking (slightly lower search quality)

## Re-enabling PyTorch

If you need PyTorch (for local embeddings or reranking):
1. Build with `--build-arg INSTALL_PYTORCH=1`
2. Set `EMBEDDING_PROVIDER=sentence-transformers` (optional, if using local embeddings)
3. Set `RAG_USE_RERANKING=True` (optional, if using reranking)

## Memory Usage Comparison

| Configuration | Memory Usage |
|--------------|--------------|
| Full (PyTorch + reranking) | ~2GB |
| OpenAI embeddings + disabled reranking | ~1.2GB |
| **Savings** | **~800MB** |

