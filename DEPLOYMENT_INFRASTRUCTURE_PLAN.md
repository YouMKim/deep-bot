# 🚀 Deployment & Infrastructure Plan

Complete infrastructure and deployment strategy for Deep Bot in production environments.

---

## 📋 Table of Contents

1. [Infrastructure Overview](#infrastructure-overview)
2. [Folder Structure](#folder-structure)
3. [Docker Setup](#docker-setup)
4. [Cloud Platform Options](#cloud-platform-options)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Monitoring & Logging](#monitoring--logging)
7. [Backup Strategy](#backup-strategy)
8. [Security Hardening](#security-hardening)
9. [Cost Optimization](#cost-optimization)
10. [Disaster Recovery](#disaster-recovery)

---

## 🏗️ Infrastructure Overview

### Architecture Components

```
┌──────────────────────────────────────────────────────────┐
│                     Cloud Platform                        │
│  (Railway / Render / AWS / GCP / Azure / DigitalOcean)   │
└────────────────────┬─────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    ┌────▼─────┐          ┌──────▼──────┐
    │  Docker  │          │   Storage   │
    │Container │          │   Volumes   │
    │          │          │             │
    │ Deep Bot ├─────────▶│ - data/     │
    │          │          │ - logs/     │
    │  + Cogs  │          │ - cache/    │
    │  + RAG   │          └─────────────┘
    │  + AI    │
    └────┬─────┘
         │
    ┌────▼─────────────────────────────────────┐
    │         External Services                │
    ├──────────────────────────────────────────┤
    │ • Discord API                            │
    │ • OpenAI / Anthropic API                 │
    │ • Vector DB (ChromaDB / Pinecone)        │
    │ • Monitoring (Sentry / DataDog)          │
    └──────────────────────────────────────────┘
```

### Core Principles
- **Containerization** - Docker for consistency
- **Stateful Storage** - Persistent volumes for data
- **Environment Config** - .env for all secrets
- **Health Monitoring** - Automated health checks
- **Horizontal Scaling** - Can run multiple instances (future)

---

## 📁 Folder Structure

### Recommended Project Structure with Deployment

```
deep-bot/
├── 📁 Core Application (Phases 1-18)
│   ├── bot.py
│   ├── config.py
│   ├── requirements.txt
│   ├── ai/
│   ├── storage/
│   ├── embedding/
│   ├── chunking/
│   ├── retrieval/
│   ├── rag/
│   ├── evaluation/
│   ├── security/
│   ├── bot/
│   │   ├── cogs/
│   │   └── loaders/
│   └── utils/
│
├── 🐳 Docker & Deployment
│   ├── Dockerfile                    # Production image
│   ├── docker-compose.yml            # Local + production setup
│   ├── .dockerignore                 # Exclude from image
│   │
│   ├── deployment/                   # ⭐ NEW FOLDER
│   │   ├── railway/
│   │   │   ├── railway.json          # Railway config
│   │   │   └── README.md             # Railway deployment guide
│   │   │
│   │   ├── render/
│   │   │   ├── render.yaml           # Render config
│   │   │   └── README.md             # Render deployment guide
│   │   │
│   │   ├── aws/
│   │   │   ├── ecs-task-definition.json  # AWS ECS
│   │   │   ├── cloudformation.yaml       # Infrastructure as Code
│   │   │   └── README.md
│   │   │
│   │   ├── gcp/
│   │   │   ├── cloud-run.yaml        # Google Cloud Run
│   │   │   ├── app.yaml              # App Engine config
│   │   │   └── README.md
│   │   │
│   │   ├── azure/
│   │   │   ├── azure-container-instance.json
│   │   │   └── README.md
│   │   │
│   │   ├── kubernetes/               # K8s deployment (advanced)
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── configmap.yaml
│   │   │   └── secrets.yaml
│   │   │
│   │   └── scripts/
│   │       ├── deploy.sh             # Automated deployment script
│   │       ├── health-check.sh       # Health monitoring
│   │       ├── backup.sh             # Data backup script
│   │       └── rollback.sh           # Rollback script
│   │
│   ├── .github/                      # CI/CD
│   │   └── workflows/
│   │       ├── test.yml              # Run tests on PR
│   │       ├── deploy.yml            # Deploy on merge to main
│   │       └── docker-build.yml      # Build & push Docker image
│   │
│   └── .gitlab-ci.yml                # GitLab CI/CD (alternative)
│
├── 📊 Monitoring & Logging
│   ├── monitoring/
│   │   ├── prometheus.yml            # Prometheus config
│   │   ├── grafana-dashboard.json    # Grafana dashboard
│   │   └── alerts.yml                # Alert rules
│   │
│   └── scripts/
│       ├── log-analyzer.py           # Parse logs for errors
│       └── metrics-exporter.py       # Export custom metrics
│
├── 🔒 Security
│   ├── .env.example                  # Template (committed)
│   ├── .env                          # Actual secrets (gitignored)
│   ├── secrets/                      # Encrypted secrets (advanced)
│   │   └── encrypted-secrets.yml
│   └── security/
│       └── audit-logs.py
│
├── 🗄️ Data (Gitignored)
│   ├── data/
│   │   ├── raw_messages/             # SQLite DBs
│   │   ├── chroma/                   # Vector embeddings
│   │   ├── cache/                    # Smart cache
│   │   ├── bookmarks.db              # User bookmarks
│   │   └── backups/                  # Automated backups
│   │
│   └── logs/
│       ├── bot.log
│       ├── errors.log
│       └── archive/                  # Rotated logs
│
└── 📝 Documentation
    ├── DEPLOYMENT_INFRASTRUCTURE_PLAN.md  # This file
    ├── DEPLOYMENT_GUIDE.md
    ├── IMPLEMENTATION_GUIDE.md
    └── PHASES 1-18.md
```

---

## 🐳 Docker Setup

### Multi-Stage Dockerfile (Already Created)

```dockerfile
# Benefits:
# - Smaller final image (no build tools)
# - Faster builds (layer caching)
# - Security (non-root user)
# - Production-ready
```

### Docker Compose (Already Created)

```yaml
# Features:
# - Persistent volumes (data/ and logs/)
# - Resource limits (2 CPU, 2GB RAM)
# - Health checks
# - Logging configuration
# - Easy local development
```

### Usage

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop bot
docker-compose down

# Restart
docker-compose restart bot

# Update and restart
git pull
docker-compose up -d --build
```

---

## ☁️ Cloud Platform Options

### 1. Railway (⭐ RECOMMENDED for Beginners)

**Pros:**
- ✅ Extremely easy deployment (connect GitHub → deploy)
- ✅ Free $5/month credit
- ✅ Automatic HTTPS
- ✅ Built-in monitoring
- ✅ Great for Discord bots

**Cons:**
- ❌ Can get expensive at scale
- ❌ Limited customization

**Setup:**

Create `deployment/railway/railway.json`:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "numReplicas": 1,
    "sleepApplication": false,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**Deployment:**
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Add environment variables
railway variables set DISCORD_TOKEN=your_token
railway variables set OPENAI_API_KEY=your_key

# 5. Deploy
railway up

# 6. View logs
railway logs
```

**Cost:** ~$5-20/month depending on usage

---

### 2. Render (Great Alternative)

**Pros:**
- ✅ Free tier available
- ✅ Easy deployment from GitHub
- ✅ Good documentation
- ✅ Auto-deploy on git push

**Cons:**
- ❌ Free tier has spin-down after inactivity
- ❌ Limited free hours

**Setup:**

Create `deployment/render/render.yaml`:

```yaml
services:
  - type: worker
    name: deep-bot
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: DISCORD_TOKEN
        sync: false
      - key: OPENAI_API_KEY
        sync: false
      - key: LOG_LEVEL
        value: INFO
    disk:
      name: bot-data
      mountPath: /app/data
      sizeGB: 1
```

**Deployment:**
1. Connect GitHub repo to Render
2. Select "New" → "Background Worker"
3. Point to `render.yaml`
4. Add environment variables in dashboard
5. Deploy!

**Cost:** Free tier or $7/month for always-on

---

### 3. AWS ECS (Production Grade)

**Pros:**
- ✅ Extremely scalable
- ✅ Full control
- ✅ Integrates with other AWS services
- ✅ Battle-tested

**Cons:**
- ❌ Complex setup
- ❌ Steeper learning curve
- ❌ Can be expensive

**Setup:**

Create `deployment/aws/ecs-task-definition.json`:

```json
{
  "family": "deep-bot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "deep-bot",
      "image": "your-ecr-repo/deep-bot:latest",
      "essential": true,
      "environment": [],
      "secrets": [
        {
          "name": "DISCORD_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:discord-token"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/deep-bot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "mountPoints": [
        {
          "sourceVolume": "bot-data",
          "containerPath": "/app/data"
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "bot-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-xxxxx",
        "transitEncryption": "ENABLED"
      }
    }
  ]
}
```

**Deployment:**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker build -t deep-bot .
docker tag deep-bot:latest your-account.dkr.ecr.us-east-1.amazonaws.com/deep-bot:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/deep-bot:latest

# Deploy to ECS
aws ecs update-service --cluster deep-bot-cluster --service deep-bot --task-definition deep-bot:1 --force-new-deployment
```

**Cost:** ~$15-50/month (Fargate + EFS + CloudWatch)

---

### 4. Google Cloud Run

**Pros:**
- ✅ Serverless (pay per use)
- ✅ Auto-scaling
- ✅ Free tier (2 million requests/month)
- ✅ Easy deployment

**Cons:**
- ❌ Not ideal for always-on bots (cold starts)
- ❌ Stateless by default

**Setup:**

Create `deployment/gcp/cloud-run.yaml`:

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: deep-bot
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "1"
    spec:
      containers:
      - image: gcr.io/your-project/deep-bot
        env:
        - name: DISCORD_TOKEN
          valueFrom:
            secretKeyRef:
              name: discord-token
              key: latest
        resources:
          limits:
            memory: 1Gi
            cpu: "1"
        volumeMounts:
        - name: data
          mountPath: /app/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: bot-data
```

**Deployment:**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/deep-bot
gcloud run deploy deep-bot --image gcr.io/your-project/deep-bot --platform managed --region us-central1 --allow-unauthenticated
```

**Cost:** ~$0-10/month (mostly free tier)

---

## 🔄 CI/CD Pipeline

### GitHub Actions (Recommended)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Build Docker image
      run: docker build -t deep-bot:latest .

    - name: Push to Docker Hub
      env:
        DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
        DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
      run: |
        echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
        docker tag deep-bot:latest your-username/deep-bot:latest
        docker push your-username/deep-bot:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest

    steps:
    - name: Deploy to Railway
      env:
        RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
      run: |
        npm install -g @railway/cli
        railway up --service deep-bot
```

---

## 📊 Monitoring & Logging

> **💡 Production Automation**: For complete background task automation (auto-sync, health checks, event listeners), see **[Phase 19: Production Automation & Background Tasks](./PHASE_19.md)**.
> This section covers monitoring infrastructure; Phase 19 covers automated operations.

### Background Task Automation (Phase 19)

Before diving into monitoring infrastructure, ensure you've implemented the automated background tasks from **Phase 19**:

- **Auto-Sync Task** - Automatically syncs vector DBs every 30 minutes (no manual intervention!)
- **Message Listener** - Captures Discord messages in real-time as they're posted
- **Health Monitor** - Runs health checks every 5 minutes and alerts on failures
- **Resource Monitor** - Tracks CPU, memory, disk usage

These tasks ensure your bot runs autonomously in the cloud with minimal manual intervention.

**Quick Setup:**
```python
# In bot.py - these cogs are loaded automatically
background_cogs = [
    'bot.tasks.auto_sync',              # Phase 19.1
    'bot.listeners.message_listener',   # Phase 19.2
    'bot.tasks.health_monitor',         # Phase 19.3
]
```

**Configuration (`.env`):**
```bash
# Background task intervals (Phase 19)
AUTO_SYNC_INTERVAL_MINUTES=30         # How often to sync vector DBs
AUTO_SYNC_STRATEGIES=temporal,conversation  # Which strategies to auto-sync
HEALTH_CHECK_INTERVAL_MINUTES=5       # How often to run health checks
```

**Production Benefits:**
- ✅ Zero manual intervention after deployment
- ✅ Real-time message processing (no lag)
- ✅ Automatic recovery from transient failures
- ✅ Self-monitoring with health checks
- ✅ Efficient incremental updates (Phase 6.8 checkpoints)

---

### 1. Application Logging

Update `bot.py` with structured logging:

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(handlers=[handler], level=logging.INFO)
```

### 2. Metrics Collection

Create `monitoring/metrics.py`:

```python
"""
Metrics collection for monitoring bot health.
"""

from typing import Dict
import time
from collections import defaultdict

class MetricsCollector:
    """Collect and export metrics."""

    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)

    def increment(self, metric: str, value: int = 1):
        """Increment a counter."""
        self.counters[metric] += value

    def gauge(self, metric: str, value: float):
        """Set a gauge value."""
        self.gauges[metric] = value

    def histogram(self, metric: str, value: float):
        """Add value to histogram."""
        self.histograms[metric].append(value)

    def get_stats(self) -> Dict:
        """Get all metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": self.gauges,
            "histograms": {
                k: {
                    "count": len(v),
                    "avg": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0
                }
                for k, v in self.histograms.items()
            }
        }

# Usage in bot
metrics = MetricsCollector()

@bot.event
async def on_command(ctx):
    metrics.increment("commands_total")
    start = time.time()

    # ... command execution ...

    duration = time.time() - start
    metrics.histogram("command_duration_seconds", duration)
```

### 3. External Monitoring (Sentry)

```python
# Add to requirements.txt
# sentry-sdk

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

sentry_sdk.init(
    dsn="https://your-sentry-dsn@sentry.io/project-id",
    traces_sample_rate=0.1,
    integrations=[
        LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR
        )
    ]
)
```

---

## 💾 Backup Strategy

### Automated Backup Script

Create `deployment/scripts/backup.sh`:

```bash
#!/bin/bash

# Backup script for Deep Bot data

BACKUP_DIR="/app/data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${TIMESTAMP}.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup data
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    /app/data/raw_messages \
    /app/data/chroma \
    /app/data/bookmarks.db \
    /app/data/user_ai_stats.json \
    /app/data/cache

# Keep only last 7 days of backups
find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +7 -delete

# Upload to S3 (optional)
if [ ! -z "$AWS_S3_BUCKET" ]; then
    aws s3 cp "$BACKUP_DIR/$BACKUP_FILE" "s3://$AWS_S3_BUCKET/backups/"
fi

echo "Backup completed: $BACKUP_FILE"
```

### Restore Script

Create `deployment/scripts/restore.sh`:

```bash
#!/bin/bash

# Restore from backup

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore.sh <backup_file>"
    exit 1
fi

# Stop bot
docker-compose down

# Restore data
tar -xzf "$BACKUP_FILE" -C /

# Restart bot
docker-compose up -d

echo "Restore completed from $BACKUP_FILE"
```

---

## 🔒 Security Hardening

### 1. Environment Variables

**Never commit secrets!**

```bash
# .env (gitignored)
DISCORD_TOKEN=your_secret_token
OPENAI_API_KEY=your_api_key

# .env.example (committed as template)
DISCORD_TOKEN=your_discord_token_here
OPENAI_API_KEY=your_openai_key_here
```

### 2. Secrets Management (Production)

Use cloud provider secret managers:

```python
# AWS Secrets Manager
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# Usage
DISCORD_TOKEN = get_secret('discord-bot-token')
```

### 3. Rate Limiting

Already in codebase, but ensure it's enabled:

```python
# In cogs/
@commands.cooldown(1, 60, commands.BucketType.user)
async def expensive_command(ctx):
    pass
```

### 4. Input Validation

See `PHASE_03_SECURITY.md` for comprehensive input validation.

---

## 💰 Cost Optimization

### Estimated Monthly Costs

| Platform | Compute | Storage | Egress | Total |
|----------|---------|---------|--------|-------|
| **Railway** | $10-20 | Included | Included | **$10-20** |
| **Render** | $7 (hobby) | $1 | Minimal | **~$8** |
| **AWS ECS** | $15 (Fargate) | $3 (EFS) | $1-5 | **$20-25** |
| **GCP Cloud Run** | $0-5 | $1 | Minimal | **~$5** |

**API Costs (Separate):**
- OpenAI GPT-4o-mini: ~$5-20/month (varies by usage)
- Anthropic Claude: ~$10-30/month
- ChromaDB: Free (self-hosted)

**Optimization Tips:**
1. Use caching (Phase 7.5) → 50-80% API cost reduction
2. Use local embeddings (sentence-transformers) → Free
3. Set token limits on responses
4. Monitor usage with tracking

---

## 🚨 Disaster Recovery

### Checklist

✅ **Automated backups** (daily)
✅ **Offsite storage** (S3 / Cloud Storage)
✅ **Tested restore procedure** (monthly)
✅ **Rollback strategy** (git tags + deployment scripts)
✅ **Health monitoring** (Sentry / DataDog)
✅ **Alerting** (email / Slack / PagerDuty)
✅ **Documentation** (runbooks for common issues)

### Recovery Time Objectives (RTO)

- **Data Loss**: < 24 hours (daily backups)
- **Service Restoration**: < 30 minutes (automated rollback)
- **Full Recovery**: < 2 hours (manual intervention if needed)

---

## 📝 Summary & Recommendations

### Quick Start Path (Recommended)

1. **Development**: Use `docker-compose up -d` locally
2. **Staging**: Deploy to Railway (free tier)
3. **Production**: Railway or Render ($7-15/month)
4. **Scale**: Migrate to AWS/GCP when needed

### Must-Have Features

- ✅ Docker containerization
- ✅ Automated backups
- ✅ Environment-based configuration
- ✅ Structured logging
- ✅ Health checks
- ✅ CI/CD pipeline

### Nice-to-Have Features

- 📊 Prometheus + Grafana monitoring
- 🔔 Slack/Discord alerting
- 🤖 Auto-scaling
- 🌐 Web dashboard (Phase suggestion!)

---

**Next Steps:**

1. Choose cloud platform (Railway recommended)
2. Set up CI/CD (GitHub Actions template provided)
3. Configure monitoring (Sentry free tier)
4. Implement backup automation
5. Test disaster recovery

**Questions? Check the detailed deployment guides in `deployment/<platform>/README.md`**

---

📚 **Related Documentation:**
- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Step-by-step deployment
- [REFACTORING_PLAN.md](./REFACTORING_PLAN.md) - Code organization
- [Phase 18: Security](./PHASE_18.md) - Security best practices
