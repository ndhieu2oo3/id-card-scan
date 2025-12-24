FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update --fix-missing -y && \
    apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix numpy version for PaddleOCR compatibility
RUN pip install --no-cache-dir 'numpy<2.0'

# Ensure gunicorn is available in the container
RUN pip install --no-cache-dir gunicorn

# Copy application code
COPY . .

# Expose ports (8000 for Flask API - TGI runs in separate Docker container)
EXPOSE 8000

# Health check (calls the registered health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python3.10 -c "import sys,requests; r=requests.get('http://localhost:8000/id_card_scan/health'); sys.exit(0 if r.status_code==200 else 1)" || exit 1

# Run the application with Gunicorn (production)
# Note: the LLM API should be provided externally (TGI or other service).
ENV HOST 0.0.0.0
ENV PORT 8000
ENV LLM_MODEL Qwen/Qwen3-1.7B-Base
# Default LLM API URL (external, typical TGI Docker maps 80->8000 on host)
ENV LLM_API_URL http://localhost:8000/v1/chat/completions
CMD ["bash", "-c", "gunicorn -w 4 -b 0.0.0.0:8000 'app:create_app()'"]