# Deployment Guide - Deep Bot

Complete guide to deploying your Discord RAG chatbot in production.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Docker Compose (Recommended)](#docker-compose)
5. [Production Deployment](#production-deployment)
6. [Monitoring & Maintenance](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- Docker & Docker Compose (or alternative deployment platform)
- Discord Bot Token ([create here](https://discord.com/developers/applications))
- OpenAI API Key (or Anthropic for Claude)

### Recommended
- Basic Linux/command line knowledge
- Git for version control
- Domain name (for production)

---

## Local Development

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/deep-bot.git
cd deep-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```bash
# Discord Configuration
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CLIENT_ID=your_client_id_here
DISCORD_GUILD_ID=your_guild_id_here  # Optional
BOT_PREFIX=!
BOT_OWNER_ID=your_discord_user_id

# AI Provider (choose one)
OPENAI_API_KEY=your_openai_key_here
# or
ANTHROPIC_API_KEY=your_anthropic_key_here

# Embedding Configuration
EMBEDDING_PROVIDER=sentence-transformers  # or "openai"
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_TOKENS=512

# Vector Store
VECTOR_STORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=data/chroma

# Chunking Strategy
CHUNKING_WINDOW_SIZE=10
CHUNKING_OVERLAP=2
CHUNKING_MAX_TOKENS=512
CHUNKING_MIN_CHUNK_SIZE=3

# RAG Configuration
RAG_TOP_K=5
RAG_MIN_SIMILARITY=0.7
RAG_DEFAULT_STRATEGY=token_aware
RAG_INCLUDE_SOURCES=True
RAG_MAX_CONTEXT_TOKENS=2000

# Rate Limiting
MESSAGE_FETCH_DELAY=1.0
MESSAGE_FETCH_BATCH_SIZE=100
MESSAGE_FETCH_MAX_RETRIES=5

# Optional: Emulation Mode
EMULATION_ENABLED=False

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=False
```

### 3. Run Locally

```bash
python bot.py
```

---

## Docker Deployment

### Step 1: Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
# Use official Python runtime as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw_messages data/chroma

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run bot
CMD ["python", "bot.py"]
```

### Step 2: Create .dockerignore

Create `.dockerignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data (will be mounted as volumes)
data/
*.db
*.sqlite

# Git
.git/
.gitignore

# IDE
.vscode/
.idea/
*.swp
*.swo

# Env files (use docker-compose env instead)
.env
.env.local

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
```

### Step 3: Build & Run

```bash
# Build image
docker build -t deep-bot:latest .

# Run container
docker run -d \
  --name deep-bot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  deep-bot:latest
```

### View Logs

```bash
# View logs
docker logs -f deep-bot

# Stop container
docker stop deep-bot

# Start container
docker start deep-bot

# Remove container
docker rm deep-bot
```

---

## Docker Compose (Recommended)

Docker Compose simplifies multi-container deployments and volume management.

### Step 1: Create docker-compose.yml

```yaml
version: '3.8'

services:
  bot:
    build: .
    container_name: deep-bot
    restart: unless-stopped

    # Environment variables
    environment:
      # Discord
      DISCORD_TOKEN: ${DISCORD_TOKEN}
      DISCORD_CLIENT_ID: ${DISCORD_CLIENT_ID}
      DISCORD_GUILD_ID: ${DISCORD_GUILD_ID:-}
      BOT_PREFIX: ${BOT_PREFIX:-!}
      BOT_OWNER_ID: ${BOT_OWNER_ID}

      # AI Provider
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}

      # Embedding
      EMBEDDING_PROVIDER: ${EMBEDDING_PROVIDER:-sentence-transformers}
      EMBEDDING_MODEL_NAME: ${EMBEDDING_MODEL_NAME:-all-MiniLM-L6-v2}
      EMBEDDING_BATCH_SIZE: ${EMBEDDING_BATCH_SIZE:-32}

      # Vector Store
      VECTOR_STORE_PROVIDER: ${VECTOR_STORE_PROVIDER:-chroma}
      CHROMA_PERSIST_DIR: /app/data/chroma

      # RAG
      RAG_TOP_K: ${RAG_TOP_K:-5}
      RAG_MIN_SIMILARITY: ${RAG_MIN_SIMILARITY:-0.7}
      RAG_DEFAULT_STRATEGY: ${RAG_DEFAULT_STRATEGY:-token_aware}

      # Logging
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
      DEBUG_MODE: ${DEBUG_MODE:-False}

    # Volumes for persistence
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

    # Health check (optional)
    healthcheck:
      test: ["CMD", "python", "-c", "import os; os.path.exists('/tmp/bot_ready')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

# Optional: Add monitoring
  # watchtower:
  #   image: containrrr/watchtower
  #   container_name: watchtower
  #   restart: unless-stopped
  #   volumes:
  #     - /var/run/docker.sock:/var/run/docker.sock
  #   command: --interval 300  # Check for updates every 5 minutes

volumes:
  data:
  logs:
```

### Step 2: Deploy with Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# View status
docker-compose ps
```

### Step 3: Update Bot

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose up -d --build

# Or use watchtower for automatic updates (see docker-compose.yml)
```

---

## Production Deployment

### Option 1: VPS (DigitalOcean, Linode, etc.)

**1. Create VPS**
- Ubuntu 22.04 LTS
- 2GB RAM minimum (4GB recommended for embeddings)
- 20GB storage

**2. Initial Setup**

```bash
# SSH into VPS
ssh root@your_vps_ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose-plugin -y

# Create user for bot
adduser botuser
usermod -aG docker botuser
su - botuser
```

**3. Deploy Bot**

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/deep-bot.git
cd deep-bot

# Create .env file
nano .env
# (paste your environment variables)

# Start with Docker Compose
docker-compose up -d

# Enable auto-restart on boot
sudo systemctl enable docker
```

**4. Setup Firewall (UFW)**

```bash
sudo ufw allow OpenSSH
sudo ufw enable
```

**5. Setup Process Monitor (Optional)**

```bash
# Install supervisor
sudo apt install supervisor

# Create supervisor config
sudo nano /etc/supervisor/conf.d/deep-bot.conf
```

Add:
```ini
[program:deep-bot]
directory=/home/botuser/deep-bot
command=docker-compose up
autostart=true
autorestart=true
stderr_logfile=/var/log/deep-bot.err.log
stdout_logfile=/var/log/deep-bot.out.log
```

```bash
# Start supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start deep-bot
```

---

### Option 2: Railway

[Railway](https://railway.app) - Simple Platform-as-a-Service

**1. Prepare for Railway**

Create `railway.json`:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python bot.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**2. Deploy**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add environment variables via Railway dashboard

# Deploy
railway up

# View logs
railway logs
```

**Cost**: ~$5-10/month

---

### Option 3: Heroku

**1. Create Heroku App**

```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create your-bot-name

# Add buildpack
heroku buildpacks:add heroku/python
```

**2. Create Procfile**

Create `Procfile`:

```
worker: python bot.py
```

**3. Configure & Deploy**

```bash
# Set config vars
heroku config:set DISCORD_TOKEN=your_token_here
heroku config:set OPENAI_API_KEY=your_key_here
# ... (add all env vars)

# Deploy
git push heroku main

# Scale worker
heroku ps:scale worker=1

# View logs
heroku logs --tail
```

**Cost**: Free tier available, $7/month for basic dyno

---

### Option 4: AWS EC2

**1. Launch EC2 Instance**
- Amazon Linux 2 or Ubuntu
- t3.small (2GB RAM minimum)
- Configure security group (no inbound needed for Discord bot)

**2. Connect & Setup**

```bash
# Connect via SSH
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Docker (Amazon Linux)
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**3. Deploy**

Same as VPS deployment above.

**Cost**: ~$10-20/month (t3.small)

---

## Monitoring & Maintenance

### Logging

**View Logs**
```bash
# Docker
docker logs -f deep-bot

# Docker Compose
docker-compose logs -f bot

# Last 100 lines
docker-compose logs --tail=100 bot
```

**Persistent Logging**

Update `bot.py` to log to file:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler()
    ]
)
```

### Health Monitoring

**Option 1: UptimeRobot**
- Free tier available
- Monitors HTTP endpoints
- Add `/health` endpoint to bot

**Option 2: Healthchecks.io**
- Ping-based monitoring
- Add cron job to ping healthchecks.io

```python
# In bot.py
import requests

async def ping_healthcheck():
    """Ping healthcheck service every 5 minutes"""
    while True:
        try:
            requests.get("https://hc-ping.com/YOUR_UUID")
        except:
            pass
        await asyncio.sleep(300)  # 5 minutes

# Start in bot ready event
@bot.event
async def on_ready():
    bot.loop.create_task(ping_healthcheck())
```

### Backup Strategy

**Automated Backups**

Create `backup.sh`:

```bash
#!/bin/bash

# Backup script for deep-bot data
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/botuser/backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup SQLite databases
cp -r /home/botuser/deep-bot/data/raw_messages $BACKUP_DIR/raw_messages_$DATE

# Backup ChromaDB
cp -r /home/botuser/deep-bot/data/chroma $BACKUP_DIR/chroma_$DATE

# Compress
tar -czf $BACKUP_DIR/backup_$DATE.tar.gz \
  $BACKUP_DIR/raw_messages_$DATE \
  $BACKUP_DIR/chroma_$DATE

# Remove uncompressed backups
rm -rf $BACKUP_DIR/raw_messages_$DATE $BACKUP_DIR/chroma_$DATE

# Keep only last 7 backups
ls -t $BACKUP_DIR/*.tar.gz | tail -n +8 | xargs -r rm

echo "Backup completed: backup_$DATE.tar.gz"
```

**Schedule with Cron**

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /home/botuser/backup.sh
```

---

## Troubleshooting

### Bot Won't Start

**Check logs:**
```bash
docker-compose logs bot
```

**Common issues:**
- Invalid Discord token → Check `.env`
- Missing dependencies → Rebuild: `docker-compose up -d --build`
- Port conflicts → Check if other services using same ports

### High Memory Usage

**Symptoms:** Bot crashes, OOM errors

**Solutions:**
1. Reduce `EMBEDDING_BATCH_SIZE`
2. Lower `RAG_TOP_K`
3. Upgrade server RAM
4. Use cloud embeddings instead of local

### Slow Responses

**Check:**
- Embedding provider (local vs cloud)
- Vector store size (rebuild with better chunking)
- RAG settings (lower `top_k`)

### Database Locked Errors

**Solution:** Check for multiple processes accessing database

```bash
# Find processes
lsof data/raw_messages/messages.db

# Kill if needed
kill -9 PID
```

### Out of Disk Space

**Check usage:**
```bash
df -h
du -sh data/*
```

**Clean up:**
```bash
# Remove old logs
find logs/ -name "*.log" -mtime +30 -delete

# Prune Docker
docker system prune -a
```

---

## Security Best Practices

### 1. Environment Variables
- ✅ **Never commit** `.env` to git
- ✅ Use different tokens for dev/prod
- ✅ Rotate tokens periodically

### 2. Server Security
```bash
# Disable root login
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no

# Enable firewall
sudo ufw enable
sudo ufw allow OpenSSH

# Keep system updated
sudo apt update && sudo apt upgrade -y
```

### 3. Bot Permissions
- Only request **necessary** Discord permissions
- Use **role-based** access control for sensitive commands
- Implement **rate limiting** for commands

### 4. API Keys
- **Limit** API key permissions
- **Monitor** API usage
- **Set** spending limits on OpenAI/Anthropic

---

## Performance Optimization

### 1. Caching

Add Redis for caching (optional):

```yaml
# Add to docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### 2. Database Optimization

```python
# In message_storage.py
# Add connection pooling
# Vacuum database periodically
import sqlite3

def vacuum_database(self):
    """Optimize database"""
    with self._get_connection() as conn:
        conn.execute("VACUUM")
```

### 3. Concurrent Processing

```python
# Process multiple channels in parallel
import asyncio

async def chunk_all_channels(channels):
    tasks = [chunk_channel(ch) for ch in channels]
    await asyncio.gather(*tasks)
```

---

## Scaling Considerations

### When to Scale?

- Serving **> 100 servers**
- **> 10,000** messages/day
- **Response time** > 3 seconds consistently

### Horizontal Scaling

Run multiple bot instances:
```bash
# Bot 1: Handles servers 1-50
# Bot 2: Handles servers 51-100
```

Share data via:
- Centralized PostgreSQL (instead of SQLite)
- Cloud vector store (Pinecone instead of ChromaDB)
- Redis for shared cache

---

## Summary

**Recommended Setup for Beginners:**
- Start with **Docker Compose** locally
- Deploy to **Railway** or **Heroku** for simplicity
- Use **sentence-transformers** (local embeddings) to save costs

**Recommended Setup for Production:**
- **VPS** (DigitalOcean/Linode) with Docker Compose
- **Automated backups** via cron
- **Monitoring** with Healthchecks.io or UptimeRobot
- **Log aggregation** with simple file logging

**Cost Comparison:**

| Platform | Monthly Cost | Pros | Cons |
|----------|--------------|------|------|
| Railway | $5-10 | Easy, auto-deploy | Limited free tier |
| Heroku | $7+ | Simple, reliable | Expensive at scale |
| DigitalOcean | $6-12 | Full control, cheap | Requires setup |
| AWS EC2 | $10-20 | Scalable, flexible | Complex, expensive |
| Local/Home | $0 | Free, full control | Uptime issues, no support |

---

## Next Steps

1. **Test locally** with Docker Compose
2. **Deploy** to chosen platform
3. **Monitor** logs for 24 hours
4. **Set up backups**
5. **Invite bot** to Discord server
6. **Start chunking** messages: `!chunk_channel`
7. **Test RAG**: `!ask what have people discussed?`

Good luck with your deployment! 🚀
