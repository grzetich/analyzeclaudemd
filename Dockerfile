# Use Python 3.11 to match production (Render.com uses Python 3.11)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p cache logs/analysis_logs templates static

# Set environment variables
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Use gunicorn like production, but with debug settings
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--reload", "--log-level", "debug", "app:app"]