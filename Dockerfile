# Use Python 3.10 to support newer numpy/pandas versions
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        gcc \
        python3-dev \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directory for credentials
RUN mkdir -p /app/credentials

# Expose port 5000 (Flask default)
EXPOSE 5000

# Run with Gunicorn (production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
