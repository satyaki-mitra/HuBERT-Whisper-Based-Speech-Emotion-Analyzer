FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploaded_files data/recorded_files data/temp_files logs exports/visualizations models/hubert models/whisper

# Download models from HuggingFace
RUN python -c "from transformers import AutoModel, AutoFeatureExtractor; \
    AutoModel.from_pretrained('facebook/hubert-base-ls960', cache_dir='models/hubert'); \
    AutoFeatureExtractor.from_pretrained('facebook/hubert-base-ls960', cache_dir='models/hubert')"

# Download Whisper model
RUN python -c "import whisper; whisper.load_model('large-v3', download_root='models/whisper')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860
ENV DEVICE=cpu

# Expose port for HuggingFace Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/api/v1/health')" || exit 1

# Start application
CMD ["python", "app.py"]