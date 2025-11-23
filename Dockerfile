# Multi-stage build for Discord bot
# Updated to conditionally install PyTorch/sentence-transformers based on build args
FROM python:3.12-slim as builder

# Build arguments to control optional dependencies
# Set INSTALL_PYTORCH=1 to install PyTorch and sentence-transformers (for reranking/local embeddings)
# Set INSTALL_PYTORCH=0 to skip (saves ~750MB, use with OpenAI embeddings + disabled reranking)
# Default: 0 (disabled) for memory optimization
ARG INSTALL_PYTORCH=0

# Install build dependencies (only if PyTorch is needed)
RUN if [ "$INSTALL_PYTORCH" = "1" ]; then \
        apt-get update && apt-get install -y \
        build-essential \
        gcc \
        g++ \
        curl \
        && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
        && . $HOME/.cargo/env \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
# Exclude testing and dev dependencies to reduce size
COPY requirements.txt .
# Create production requirements without test/dev dependencies
RUN grep -vE "(pytest|black)" requirements.txt > requirements-prod.txt || cp requirements.txt requirements-prod.txt

# Conditionally install PyTorch and sentence-transformers
# Only install if INSTALL_PYTORCH=1 (default is 0 for memory optimization)
RUN if [ "$INSTALL_PYTORCH" = "1" ]; then \
        echo "Installing CPU-only PyTorch for sentence-transformers (required for reranking/local embeddings)..." && \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
        echo "PyTorch CPU-only installed" && \
        echo "Installing tokenizers (required by sentence-transformers)..." && \
        . $HOME/.cargo/env && \
        pip install --no-cache-dir tokenizers>=0.15.0 && \
        echo "Tokenizers installed" && \
        echo "Installing sentence-transformers..." && \
        pip install --no-cache-dir sentence-transformers>=2.2.2 && \
        echo "sentence-transformers installed"; \
    else \
        echo "Skipping PyTorch/sentence-transformers installation (using OpenAI embeddings + disabled reranking)"; \
    fi

# Install other dependencies
# If PyTorch is disabled, exclude sentence-transformers and tokenizers from requirements
RUN if [ "$INSTALL_PYTORCH" = "1" ]; then \
        echo "Installing all dependencies (including PyTorch)..." && \
        pip install --no-cache-dir -r requirements-prod.txt; \
    else \
        echo "Filtering out PyTorch dependencies from requirements..." && \
        grep -vE "(sentence-transformers|tokenizers)" requirements-prod.txt > requirements-no-pytorch.txt || cp requirements-prod.txt requirements-no-pytorch.txt && \
        echo "Installing dependencies without PyTorch..." && \
        echo "Filtered requirements file:" && \
        head -20 requirements-no-pytorch.txt && \
        echo "..." && \
        pip install --no-cache-dir -r requirements-no-pytorch.txt || (echo "ERROR: pip install failed!" && cat requirements-no-pytorch.txt && exit 1) && \
        echo "Verifying core dependencies are installed..." && \
        python -c "import discord; print('✓ discord.py installed')" || (echo "ERROR: discord.py not installed!" && pip list | grep -i discord && exit 1) && \
        python -c "import openai; print('✓ openai installed')" || (echo "ERROR: openai not installed!" && exit 1) && \
        python -c "import chromadb; print('✓ chromadb installed')" || (echo "WARNING: chromadb not installed (may be optional)" && pip list | grep -i chroma) && \
        echo "Verifying sentence-transformers is NOT installed..." && \
        (python -c "import sentence_transformers" 2>/dev/null && echo "WARNING: sentence-transformers was installed (unexpected)" || echo "✓ Confirmed: sentence-transformers is NOT installed"); \
    fi && \
    pip cache purge && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/* && \
    find /usr/local/lib/python3.12/site-packages -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.pyc" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.pyo" -delete 2>/dev/null || true

# Final stage - minimal runtime image
FROM python:3.12-slim

# Install only essential runtime dependencies
# libgomp1 and libgcc-s1 may be needed for tokenizers/PyTorch compiled extensions
# Only install if PyTorch is being used
ARG INSTALL_PYTORCH=0
RUN if [ "$INSTALL_PYTORCH" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgcc-s1 \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean \
        && rm -rf /var/cache/apt/archives/*; \
    fi

# Copy Python packages from builder (system-wide installation)
# Ensure directories exist first and copy ALL files including compiled extensions
RUN mkdir -p /usr/local/lib/python3.12/site-packages && \
    mkdir -p /usr/local/bin

# Copy site-packages preserving all files including .so files (compiled extensions)
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Conditionally reinstall tokenizers (only if PyTorch is installed)
ARG INSTALL_PYTORCH=0
RUN if [ "$INSTALL_PYTORCH" = "1" ]; then \
        pip install --no-cache-dir --force-reinstall --no-deps tokenizers>=0.15.0 || \
        pip install --no-cache-dir tokenizers>=0.15.0 && \
        python -c "import tokenizers; print(f'✓ tokenizers {tokenizers.__version__} installed')" && \
        python -c "import sentence_transformers; print(f'✓ sentence-transformers installed')" && \
        echo "PyTorch dependencies verified"; \
    else \
        echo "PyTorch dependencies skipped (using OpenAI embeddings)"; \
    fi

# Clean up ONLY documentation and test files (preserve all .so, .py, and .dist-info files)
RUN find /usr/local/lib/python3.12/site-packages -name "*.md" -not -path "*/.dist-info/*" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.rst" -not -path "*/.dist-info/*" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name "*.txt" -path "*/test*" -not -path "*/.dist-info/*" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "tests" -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name "test" -exec rm -r {} + 2>/dev/null || true

# Set HuggingFace cache location (will be mounted as volume)
# Only needed if using sentence-transformers
ARG INSTALL_PYTORCH=0
RUN if [ "$INSTALL_PYTORCH" = "1" ]; then \
        mkdir -p /root/.cache/huggingface && \
        chmod -R 755 /root/.cache && \
        rm -rf /root/.cache/huggingface/* 2>/dev/null || true; \
    fi
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create data directories with proper permissions
RUN mkdir -p data/raw_messages data/chroma_db && \
    chmod -R 755 data

# Expose port (Railway will auto-assign, but good practice)
EXPOSE 8000

# Run the bot
CMD ["python", "bot.py"]

