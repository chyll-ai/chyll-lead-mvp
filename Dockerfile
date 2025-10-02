FROM python:3.11-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install only basic dependencies first
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    python-dotenv==1.0.0 \
    httpx==0.25.2 \
    tldextract==5.1.1

# Copy application code
COPY services/fastapi/ ./services/fastapi/

# Set working directory
WORKDIR /app/services/fastapi

# Expose port
EXPOSE 8000

# Start command - use the minimal app that doesn't need ML libraries
CMD ["python", "-m", "uvicorn", "app-minimal:app", "--host", "0.0.0.0", "--port", "8000"]