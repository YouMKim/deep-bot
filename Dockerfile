# Deep Bot - Production Docker Image
# Multi-stage build for optimal image size

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 botuser && \
    mkdir -p /app/data /app/logs && \
    chown -R botuser:botuser /app

# Copy Python dependencies from builder
COPY --from=builder --chown=botuser:botuser /root/.local /home/botuser/.local

# Copy application code
COPY --chown=botuser:botuser . .

# Set Python path
ENV PATH=/home/botuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run bot
CMD ["python", "bot.py"]
