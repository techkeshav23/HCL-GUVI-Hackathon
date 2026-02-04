# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for librosa/soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model during build to bake it into the image
# This ensures we don't depend on external APIs/Internet at runtime
RUN python utils/download_model.py

# Create temp directory for audio files and set permissions
RUN mkdir -p audio/temp && chmod 777 audio/temp

# Create a non-root user (security best practice required by some hosts)
RUN useradd -m -u 1000 user
USER user

# Expose port (Hugging Face Spaces uses 7860 by default)
ENV PORT=7860
EXPOSE 7860

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
