FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    # For PDF processing
    poppler-utils \
    # For OCR (optional)
    tesseract-ocr \
    tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY extracteur_devis_simplifie.py .

# Expose port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
# Note: Set GOOGLE_API_KEY environment variable in Render
ENTRYPOINT ["streamlit", "run", "extracteur_devis_simplifie.py", "--server.port=8501", "--server.address=0.0.0.0"]