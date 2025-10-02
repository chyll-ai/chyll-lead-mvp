FROM python:3.11.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/fastapi/ ./services/fastapi/

# Set working directory
WORKDIR /app/services/fastapi

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "app-production:app", "--host", "0.0.0.0", "--port", "8000"]
