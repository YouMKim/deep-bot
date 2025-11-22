# Multi-stage build for Discord bot
# Updated to ensure tokenizers compiled extensions are properly copied
FROM python:3.12-slim as builder

# Install build dependencies including Rust (required for tokenizers)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env \
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
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    echo "PyTorch CPU-only installed"

# Install tokenizers explicitly before sentence-transformers to avoid dependency issues
# Ensure Rust is in PATH for compilation
RUN echo "Installing tokenizers (required by sentence-transformers)..." && \
    . $HOME/.cargo/env && \
    pip install --no-cache-dir tokenizers>=0.15.0 && \
    echo "Tokenizers installed"

# Install other dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/* && \
    find /usr/local/lib/python3.12/site-packages -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.pyc" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.pyo" -delete 2>/dev/null || true

# Final stage - minimal runtime image
FROM python:3.12-slim

# Install only essential runtime dependencies
# libgcc-s1 may be needed for tokenizers compiled extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Copy Python packages from builder (system-wide installation)
# Ensure directories exist first and copy ALL files including compiled extensions
RUN mkdir -p /usr/local/lib/python3.12/site-packages && \
    mkdir -p /usr/local/bin

# Copy site-packages preserving all files including .so files (compiled extensions)
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Reinstall tokenizers to ensure compiled extensions work correctly
# This fixes issues where .so files might not be compatible or missing
RUN pip install --no-cache-dir --force-reinstall --no-deps tokenizers>=0.15.0 || \
    pip install --no-cache-dir tokenizers>=0.15.0

# Verify critical packages are installed and importable BEFORE cleanup
RUN python -c "import tokenizers; print(f'✓ tokenizers {tokenizers.__version__} installed')" && \
    python -c "import sentence_transformers; print(f'✓ sentence-transformers installed')" && \
    echo "All required packages verified"

# Clean up ONLY documentation and test files (preserve all .so, .py, and .dist-info files)
RUN find /usr/local/lib/python3.12/site-packages -name "*.md" -not -path "*/.dist-info/*" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.rst" -not -path "*/.dist-info/*" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.txt" -path "*/test*" -not -path "*/.dist-info/*" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "tests" -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "test" -exec rm -r {} + 2>/dev/null || true

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

