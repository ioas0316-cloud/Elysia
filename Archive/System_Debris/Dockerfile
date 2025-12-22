# Elysia Consciousness Engine - Docker Configuration
# ==================================================
# Multi-stage build for optimized production deployment

# Stage 1: Base image with dependencies
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs saves data

# Expose ports (API and Dashboard)
EXPOSE 8000 8080

# Default command for development
CMD ["python", "Core/Interface/api_server.py"]

# Stage 3: Production image
FROM base as production

# Create non-root user for security
RUN useradd -m -u 1000 elysia && \
    mkdir -p /app/logs /app/data /app/saves && \
    chown -R elysia:elysia /app

# Copy application code with correct ownership
COPY --chown=elysia:elysia . .

# Switch to non-root user
USER elysia

# Expose API port
EXPOSE 8000

# Health check using FastAPI endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys, requests; r = requests.get('http://localhost:8000/health', timeout=5); sys.exit(0 if r.status_code == 200 else 1)" || exit 1

# Production command with configurable workers
CMD uvicorn Core.Interface.api_server:app --host 0.0.0.0 --port 8000 --workers ${ELYSIA_WORKERS:-4} --log-level ${ELYSIA_LOG_LEVEL:-info}
