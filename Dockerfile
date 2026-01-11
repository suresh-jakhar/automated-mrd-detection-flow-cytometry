FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir gunicorn

# Copy application
COPY . /app

# Expose the port used by the Flask app
EXPOSE 5000

# Run with gunicorn for production
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
