# Multi-stage build for Discord bot
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.12-slim

# Install runtime dependencies (for sentence-transformers and ChromaDB)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create data directories with proper permissions
RUN mkdir -p data/raw_messages data/chroma_db && \
    chmod -R 755 data

# Create cache directory for ML models
RUN mkdir -p /root/.cache/huggingface && \
    chmod -R 755 /root/.cache

# Expose port (Railway will auto-assign, but good practice)
EXPOSE 8000

# Run the bot
CMD ["python", "bot.py"]

