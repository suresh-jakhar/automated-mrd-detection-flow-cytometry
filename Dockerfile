# Multi-stage build to reduce image size
FROM python:3.11-slim AS builder

WORKDIR /tmp

# Install build dependencies (only needed during build)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install to a custom directory (CPU-only PyTorch for smaller image)
COPY requirements-docker.txt .
RUN pip install --user --no-cache-dir -r requirements-docker.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to use local pip packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Copy application code
COPY model_wrapper.py .
COPY inference.py .
COPY app.py .
COPY model/ ./model/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
