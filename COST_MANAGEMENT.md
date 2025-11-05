# Cost Management & Budget Planning 💰

**Build a powerful AI chatbot without breaking the bank!**

This guide helps you understand, track, and control costs when building your Discord AI chatbot. Learn how to use free options or manage paid APIs on a budget.

---

## Table of Contents

1. [Cost Overview](#cost-overview)
2. [Free vs Paid Options](#free-vs-paid-options)
3. [Cost Calculator](#cost-calculator)
4. [Budget Planning](#budget-planning)
5. [Cost Optimization](#cost-optimization)
6. [Monitoring & Alerts](#monitoring--alerts)
7. [Cost Tracking Implementation](#cost-tracking-implementation)

---

## Cost Overview

### What Costs Money?

Your Discord chatbot has three potential cost centers:

```
┌─────────────────────────────────────────────────────────────┐
│                     Cost Centers                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. LLM API Calls (answering questions)                     │
│     └─> OpenAI, Anthropic, Cohere                           │
│     └─> Cost: $0.0004 - $0.03 per query                    │
│                                                               │
│  2. Embedding API Calls (converting text to vectors)        │
│     └─> OpenAI embeddings, Cohere                           │
│     └─> Cost: $0.0001 per 1000 tokens                      │
│                                                               │
│  3. Vector Database Hosting (storing embeddings)            │
│     └─> Pinecone, Weaviate, Qdrant                         │
│     └─> Cost: $0 - $70/month                               │
│                                                               │
│  4. Bot Hosting (running the bot 24/7)                     │
│     └─> Railway, Heroku, AWS                                │
│     └─> Cost: $0 - $10/month                               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Good news:** Options 2-4 can be **completely free** with local alternatives!

---

## Free vs Paid Options

### 🆓 Option 1: Completely Free Setup

**Total monthly cost: $0**

| Component | Free Option | Performance |
|-----------|-------------|-------------|
| **LLM** | Ollama (local) | Good, slower |
| **Embeddings** | sentence-transformers (local) | Good |
| **Vector DB** | ChromaDB (local) | Excellent |
| **Hosting** | Run on your computer | Only when computer is on |

**Pros:**
- ✅ Zero cost
- ✅ Full privacy
- ✅ No API limits
- ✅ Great for learning

**Cons:**
- ❌ Slower responses (5-30 seconds)
- ❌ Requires decent computer (8GB+ RAM)
- ❌ Not 24/7 (unless always-on computer)

**Best for:** Learning, testing, personal projects

### 💵 Option 2: Hybrid (Paid LLM, Free Everything Else)

**Monthly cost: ~$5-20**

| Component | Option | Cost |
|-----------|--------|------|
| **LLM** | OpenAI gpt-4o-mini | ~$5-20/month |
| **Embeddings** | sentence-transformers (local) | $0 |
| **Vector DB** | ChromaDB (local) | $0 |
| **Hosting** | Railway free tier | $0 |

**Pros:**
- ✅ Fast AI responses (1-3 seconds)
- ✅ High-quality answers
- ✅ Low cost
- ✅ Can run 24/7

**Cons:**
- ❌ Ongoing monthly cost
- ❌ API rate limits
- ❌ Need to monitor usage

**Best for:** Small communities, production use

### 💰 Option 3: Fully Paid (Best Performance)

**Monthly cost: ~$50-100+**

| Component | Option | Cost |
|-----------|--------|------|
| **LLM** | OpenAI GPT-4 | ~$30-50/month |
| **Embeddings** | OpenAI embeddings | ~$1-2/month |
| **Vector DB** | Pinecone | ~$70/month |
| **Hosting** | AWS/Heroku | ~$10/month |

**Pros:**
- ✅ Fastest responses (<1 second)
- ✅ Best quality
- ✅ Highly scalable
- ✅ Professional reliability

**Cons:**
- ❌ Expensive
- ❌ Overkill for small projects

**Best for:** Large communities, commercial use

---

## Cost Calculator

### API Pricing Reference

#### LLM (Language Models)

| Model | Provider | Input (per 1M tokens) | Output (per 1M tokens) | Cost per Query* |
|-------|----------|----------------------|------------------------|-----------------|
| **gpt-4o-mini** | OpenAI | $0.15 | $0.60 | $0.0004 |
| **gpt-4o** | OpenAI | $2.50 | $10.00 | $0.008 |
| **gpt-4-turbo** | OpenAI | $10.00 | $30.00 | $0.03 |
| **claude-3-haiku** | Anthropic | $0.25 | $1.25 | $0.0008 |
| **claude-3-sonnet** | Anthropic | $3.00 | $15.00 | $0.01 |
| **Llama 3.2 3B** | Ollama (local) | $0 | $0 | $0 |
| **Mistral 7B** | Ollama (local) | $0 | $0 | $0 |

*Typical query: 2,000 input tokens + 150 output tokens

#### Embeddings

| Model | Provider | Cost per 1M tokens | Cost per 1000 messages* |
|-------|----------|-------------------|------------------------|
| **text-embedding-3-small** | OpenAI | $0.02 | $0.002 |
| **text-embedding-3-large** | OpenAI | $0.13 | $0.013 |
| **all-MiniLM-L6-v2** | sentence-transformers | $0 | $0 |
| **all-mpnet-base-v2** | sentence-transformers | $0 | $0 |

*Average message: 100 tokens

#### Vector Databases

| Provider | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| **ChromaDB** | Unlimited (local) | N/A | Learning, small projects |
| **Pinecone** | None | $70/month (1M vectors) | Production |
| **Weaviate** | Cloud trial | $25+/month | Mid-size projects |
| **Qdrant** | 1GB free (cloud) | $0.10/GB/month | Cost-conscious production |

### Interactive Cost Calculator

Use this Python script to estimate your costs:

```python
"""
Cost calculator for Discord AI chatbot.
"""


class CostCalculator:
    """Calculate estimated costs for your chatbot."""

    # Pricing per million tokens
    PRICES = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "ollama": {"input": 0, "output": 0},
    }

    EMBEDDING_PRICES = {
        "openai-small": 0.02,  # per 1M tokens
        "openai-large": 0.13,
        "sentence-transformers": 0,
    }

    def calculate_query_cost(
        self,
        model: str,
        queries_per_day: int,
        avg_context_tokens: int = 2000,
        avg_response_tokens: int = 150,
    ) -> dict:
        """
        Calculate cost for LLM queries.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            queries_per_day: Expected queries per day
            avg_context_tokens: Average tokens in context
            avg_response_tokens: Average tokens in response

        Returns:
            Cost breakdown dict
        """
        if model not in self.PRICES:
            raise ValueError(f"Unknown model: {model}")

        prices = self.PRICES[model]

        # Calculate costs
        daily_input_tokens = queries_per_day * avg_context_tokens
        daily_output_tokens = queries_per_day * avg_response_tokens

        daily_input_cost = (daily_input_tokens / 1_000_000) * prices["input"]
        daily_output_cost = (daily_output_tokens / 1_000_000) * prices["output"]

        daily_cost = daily_input_cost + daily_output_cost
        monthly_cost = daily_cost * 30
        yearly_cost = monthly_cost * 12

        return {
            "model": model,
            "queries_per_day": queries_per_day,
            "daily_cost": round(daily_cost, 4),
            "monthly_cost": round(monthly_cost, 2),
            "yearly_cost": round(yearly_cost, 2),
            "cost_per_query": round(daily_cost / queries_per_day, 6),
        }

    def calculate_embedding_cost(
        self,
        provider: str,
        messages_to_embed: int,
        avg_tokens_per_message: int = 100,
    ) -> dict:
        """
        Calculate cost for embedding messages.

        Args:
            provider: Embedding provider
            messages_to_embed: Number of messages
            avg_tokens_per_message: Average tokens per message

        Returns:
            Cost breakdown dict
        """
        if provider not in self.EMBEDDING_PRICES:
            raise ValueError(f"Unknown provider: {provider}")

        price_per_million = self.EMBEDDING_PRICES[provider]

        total_tokens = messages_to_embed * avg_tokens_per_message
        total_cost = (total_tokens / 1_000_000) * price_per_million

        return {
            "provider": provider,
            "messages": messages_to_embed,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "cost_per_message": round(total_cost / messages_to_embed, 6),
        }

    def print_cost_comparison(self, queries_per_day: int):
        """Print cost comparison for different models."""
        print(f"\n💰 Cost Comparison ({queries_per_day} queries/day)\n")
        print("=" * 70)
        print(f"{'Model':<20} {'Daily':<12} {'Monthly':<12} {'Per Query':<12}")
        print("=" * 70)

        for model in self.PRICES.keys():
            costs = self.calculate_query_cost(model, queries_per_day)
            print(
                f"{model:<20} "
                f"${costs['daily_cost']:<11.4f} "
                f"${costs['monthly_cost']:<11.2f} "
                f"${costs['cost_per_query']:<11.6f}"
            )

        print("=" * 70)


# Example usage
if __name__ == "__main__":
    calc = CostCalculator()

    # Your estimated usage
    queries_per_day = 50  # Adjust this!

    print("🤖 Discord AI Chatbot Cost Estimator")
    print(f"\nAssumptions:")
    print(f"  - {queries_per_day} queries per day")
    print(f"  - 2,000 tokens context per query")
    print(f"  - 150 tokens response per query")

    # Compare all models
    calc.print_cost_comparison(queries_per_day)

    # Specific calculation
    print("\n📊 Recommended: gpt-4o-mini")
    costs = calc.calculate_query_cost("gpt-4o-mini", queries_per_day)
    print(f"  Daily cost:   ${costs['daily_cost']:.4f}")
    print(f"  Monthly cost: ${costs['monthly_cost']:.2f}")
    print(f"  Yearly cost:  ${costs['yearly_cost']:.2f}")

    # Embedding costs
    print("\n📚 One-time embedding cost (10,000 messages):")
    embed_costs = calc.calculate_embedding_cost("openai-small", 10000)
    print(f"  OpenAI: ${embed_costs['total_cost']:.4f}")
    embed_costs = calc.calculate_embedding_cost("sentence-transformers", 10000)
    print(f"  Local:  ${embed_costs['total_cost']:.4f} (FREE)")
```

Run it:
```bash
python cost_calculator.py
```

**Example output:**
```
💰 Cost Comparison (50 queries/day)

======================================================================
Model                Daily        Monthly      Per Query
======================================================================
gpt-4o-mini          $0.0195      $0.59        $0.000390
gpt-4o               $0.3250      $9.75        $0.006500
gpt-4-turbo          $1.3000      $39.00       $0.026000
claude-3-haiku       $0.0406      $1.22        $0.000813
ollama               $0.0000      $0.00        $0.000000
======================================================================
```

---

## Budget Planning

### Setting Your Budget

**Step 1: Estimate your usage**

```
Daily queries = (active users) × (avg queries per user per day)

Example:
- 10 active users
- 5 queries each per day
- = 50 queries/day
```

**Step 2: Choose your tier**

| Budget | Recommended Setup | Expected Performance |
|--------|------------------|---------------------|
| **$0/month** | Ollama + local embeddings + ChromaDB | Good for learning |
| **$1-5/month** | gpt-4o-mini (limited) + local embeddings | 50-300 queries/day |
| **$10-20/month** | gpt-4o-mini + local embeddings | 500-1000 queries/day |
| **$50+/month** | GPT-4 + OpenAI embeddings + Pinecone | Production scale |

**Step 3: Set API spending limits**

```bash
# In OpenAI dashboard:
Settings → Billing → Usage limits
- Set "Hard limit": $20/month
- Set "Email notification": $15/month
```

### Budget Allocation Example

**Monthly budget: $20**

```
┌─────────────────────────────────────────┐
│         Budget Allocation               │
├─────────────────────────────────────────┤
│                                         │
│  LLM queries (gpt-4o-mini)    $15      │
│    └─> ~750 queries                    │
│                                         │
│  Emergency buffer           $5         │
│    └─> Unexpected spikes               │
│                                         │
│  Total:                     $20        │
│                                         │
└─────────────────────────────────────────┘
```

---

## Cost Optimization

### 10 Ways to Reduce Costs

**1. Use local embeddings instead of OpenAI**
```python
# ❌ Expensive: $0.02 per 1M tokens
embeddings = openai.Embedding.create(...)

# ✅ Free: $0
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
```
**Savings: ~$5-10/month**

**2. Use cheaper LLM model**
```python
# ❌ Expensive: $0.026 per query
model = "gpt-4-turbo"

# ✅ Cheap: $0.0004 per query (65x cheaper!)
model = "gpt-4o-mini"
```
**Savings: ~90% on LLM costs**

**3. Limit context window**
```python
# ❌ Expensive: 5000 tokens context
context_tokens = 5000

# ✅ Optimized: 2000 tokens context
context_tokens = 2000  # Usually enough!
```
**Savings: ~60% on input costs**

**4. Cache frequent queries**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_ai_response(query: str, context: str):
    # Only calls API once per unique query
    return openai.ChatCompletion.create(...)
```
**Savings: 30-50% for repeated questions**

**5. Rate limit expensive operations**
```python
@commands.cooldown(1, 30, commands.BucketType.user)  # 1 per 30 seconds
async def ask(self, ctx, *, question: str):
    # Prevents abuse
```
**Savings: Prevents cost spikes**

**6. Use streaming responses** (for user experience, not cost)
```python
# Appears faster to users, same cost
stream = openai.ChatCompletion.create(..., stream=True)
```

**7. Batch embedding operations**
```python
# ❌ One at a time (many API calls)
for message in messages:
    embedding = embed(message)

# ✅ Batch (one API call)
embeddings = embed_batch(messages)  # Much faster!
```

**8. Set max token limits**
```python
response = openai.ChatCompletion.create(
    ...,
    max_tokens=200  # Prevent huge responses
)
```
**Savings: 50% on output costs**

**9. Monitor and analyze usage**
```python
# Log every API call
logger.info(f"API call: {tokens} tokens, ${cost:.4f}")
```

**10. Use tiered responses**
```python
# Quick questions → cheap model
if len(question) < 50:
    model = "gpt-4o-mini"
# Complex questions → better model
else:
    model = "gpt-4o"
```
**Savings: 30-40% overall**

---

## Monitoring & Alerts

### Implementing Cost Tracking

Create `utils/cost_tracker.py`:

```python
"""
Track API costs in real-time.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict


logger = logging.getLogger(__name__)


class CostTracker:
    """Track and log API costs."""

    # Pricing (per 1M tokens)
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "text-embedding-3-small": {"input": 0.02, "output": 0},
    }

    def __init__(self, log_file: str = "data/costs.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.daily_total = 0.0
        self.monthly_total = 0.0

        # Load existing totals
        self._load_totals()

    def log_api_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str = None,
        command: str = None,
    ):
        """
        Log an API call and calculate cost.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            user_id: User who made the request
            command: Command that triggered the call
        """
        if model not in self.PRICING:
            logger.warning(f"Unknown model for pricing: {model}")
            return

        pricing = self.PRICING[model]

        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        # Update totals
        self.daily_total += total_cost
        self.monthly_total += total_cost

        # Log to file
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": round(total_cost, 6),
            "daily_total": round(self.daily_total, 4),
            "monthly_total": round(self.monthly_total, 2),
            "user_id": user_id,
            "command": command,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Log to console
        logger.info(
            f"API call: {model} | "
            f"Tokens: {input_tokens + output_tokens} | "
            f"Cost: ${total_cost:.6f} | "
            f"Daily: ${self.daily_total:.4f} | "
            f"Monthly: ${self.monthly_total:.2f}"
        )

        # Check budget alerts
        self._check_alerts()

    def _load_totals(self):
        """Load daily/monthly totals from log file."""
        if not self.log_file.exists():
            return

        today = datetime.utcnow().date()
        month = datetime.utcnow().month

        with open(self.log_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                entry_date = datetime.fromisoformat(entry["timestamp"]).date()

                # Sum today's costs
                if entry_date == today:
                    self.daily_total += entry["cost"]

                # Sum this month's costs
                if datetime.fromisoformat(entry["timestamp"]).month == month:
                    self.monthly_total += entry["cost"]

    def _check_alerts(self):
        """Check if budget thresholds are exceeded."""
        # Daily alert: $1
        if self.daily_total > 1.0:
            logger.warning(f"⚠️  Daily cost exceeded $1: ${self.daily_total:.2f}")

        # Monthly alert: $20
        if self.monthly_total > 20.0:
            logger.error(f"🚨 Monthly cost exceeded $20: ${self.monthly_total:.2f}")

    def get_stats(self) -> Dict:
        """Get current cost statistics."""
        return {
            "daily_total": round(self.daily_total, 4),
            "monthly_total": round(self.monthly_total, 2),
            "log_file": str(self.log_file),
        }


# Global instance
cost_tracker = CostTracker()
```

### Using Cost Tracker

```python
# In your AI service
from utils.cost_tracker import cost_tracker

response = openai.ChatCompletion.create(...)

# Log the cost
cost_tracker.log_api_call(
    model="gpt-4o-mini",
    input_tokens=response.usage.prompt_tokens,
    output_tokens=response.usage.completion_tokens,
    user_id=str(user_id),
    command="ask"
)
```

### Discord Command to Check Costs

```python
@commands.command(name="costs")
@commands.has_permissions(administrator=True)
async def costs(self, ctx):
    """Show API cost statistics (admin only)."""
    stats = cost_tracker.get_stats()

    embed = discord.Embed(
        title="💰 API Cost Statistics",
        color=discord.Color.gold()
    )

    embed.add_field(
        name="Today",
        value=f"${stats['daily_total']:.4f}",
        inline=True
    )

    embed.add_field(
        name="This Month",
        value=f"${stats['monthly_total']:.2f}",
        inline=True
    )

    await ctx.send(embed=embed)
```

---

## Cost Tracking Implementation

### Full Example: AI Service with Cost Tracking

```python
"""
AI service with built-in cost tracking.
"""

import openai
from utils.cost_tracker import cost_tracker


class CostAwareAIService:
    """AI service that tracks costs automatically."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def query(
        self,
        question: str,
        context: str,
        user_id: str = None
    ) -> dict:
        """
        Query AI with automatic cost tracking.

        Returns:
            {
                "answer": str,
                "cost": float,
                "tokens": int
            }
        """
        # Make API call
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=200
        )

        # Extract usage
        usage = response.usage
        answer = response.choices[0].message.content

        # Track cost
        cost_tracker.log_api_call(
            model=self.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            user_id=user_id,
            command="query"
        )

        # Calculate cost for return
        pricing = cost_tracker.PRICING[self.model]
        cost = (
            (usage.prompt_tokens / 1_000_000) * pricing["input"] +
            (usage.completion_tokens / 1_000_000) * pricing["output"]
        )

        return {
            "answer": answer,
            "cost": round(cost, 6),
            "tokens": usage.total_tokens
        }
```

---

## Summary

### Key Takeaways

1. **Start free** - Use Ollama and local embeddings for learning
2. **Monitor costs** - Implement cost tracking from day one
3. **Set limits** - Configure API spending limits
4. **Optimize** - Use cheaper models, limit context, cache queries
5. **Scale gradually** - Only upgrade when you need to

### Cost Comparison Table

| Setup | Monthly Cost | Performance | Best For |
|-------|--------------|-------------|----------|
| Local-only (Ollama) | $0 | Good | Learning |
| gpt-4o-mini + local | $1-5 | Great | Personal use |
| gpt-4o-mini + local (high volume) | $10-20 | Great | Small communities |
| GPT-4 + OpenAI embeddings | $50+ | Excellent | Production |

### Recommended Path

1. **Phase 0-6:** Use Ollama (free) while learning
2. **Phase 7-10:** Switch to gpt-4o-mini ($5/month) for better quality
3. **Phase 11+:** Add cost tracking and optimization
4. **Production:** Evaluate based on actual usage

---

## Additional Resources

- [OpenAI Pricing](https://openai.com/api/pricing/)
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [Ollama Models](https://ollama.com/library)
- [Token Calculator](https://platform.openai.com/tokenizer)

---

**Ready to build cost-effectively?** → Start with [Phase 0](PHASE_00.md) using Ollama!
