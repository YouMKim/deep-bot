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
# Exclude testing and dev dependencies to reduce size
COPY requirements.txt .
# Create production requirements without test/dev dependencies
RUN grep -vE "(pytest|black)" requirements.txt > requirements-prod.txt || cp requirements.txt requirements-prod.txt

# Always install sentence-transformers and tokenizers (needed for reranking)
# Even when using OpenAI embeddings, reranking still requires sentence-transformers
# Install CPU-only PyTorch first (much smaller than GPU version)
# This reduces PyTorch size from ~2GB to ~500MB
RUN echo "Installing CPU-only PyTorch for sentence-transformers (required for reranking)..." && \
    pip install --no-cache-dir --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    echo "PyTorch CPU-only installed"

# Install other dependencies
RUN pip install --no-cache-dir --user -r requirements-prod.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/* && \
    find /root/.local -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /root/.local -name "*.pyc" -delete 2>/dev/null || true && \
    find /root/.local -name "*.pyo" -delete 2>/dev/null || true && \
    find /root/.local -type d -name "tests" -exec rm -r {} + 2>/dev/null || true && \
    find /root/.local -type d -name "test" -exec rm -r {} + 2>/dev/null || true && \
    find /root/.local -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true

# Final stage - minimal runtime image
FROM python:3.12-slim

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Copy Python packages from builder (exclude model cache and pip cache)
COPY --from=builder /root/.local /root/.local

# Clean up any cached models, test files, and documentation (keep .dist-info for Python)
RUN rm -rf /root/.local/lib/python*/site-packages/*/models/* 2>/dev/null || true && \
    rm -rf /root/.local/lib/python*/site-packages/*/cache/* 2>/dev/null || true && \
    find /root/.local -type d -name "tests" -exec rm -r {} + 2>/dev/null || true && \
    find /root/.local -type d -name "test" -exec rm -r {} + 2>/dev/null || true && \
    find /root/.local -name "*.md" -not -path "*/.dist-info/*" -delete 2>/dev/null || true && \
    find /root/.local -name "*.txt" -path "*/test*" -not -path "*/.dist-info/*" -delete 2>/dev/null || true && \
    find /root/.local -name "*.rst" -not -path "*/.dist-info/*" -delete 2>/dev/null || true

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set HuggingFace cache location (will be mounted as volume)
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create data directories with proper permissions
RUN mkdir -p data/raw_messages data/chroma_db && \
    chmod -R 755 data

# Create cache directory for ML models (will be mounted as volume)
RUN mkdir -p /root/.cache/huggingface && \
    chmod -R 755 /root/.cache

# Clean up any accidentally downloaded models
RUN rm -rf /root/.cache/huggingface/* 2>/dev/null || true

# Expose port (Railway will auto-assign, but good practice)
EXPOSE 8000

# Run the bot
CMD ["python", "bot.py"]

