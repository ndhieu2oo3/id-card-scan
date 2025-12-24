#!/bin/bash

# ID Card OCR API startup script

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
if [ ! -f ".requirements_installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    
    # Fix numpy version for PaddleOCR compatibility
    pip install 'numpy<2.0'
    
    touch .requirements_installed
fi

# Load environment variables from .env
export $(cat .env | grep -v '#' | xargs)

# Suppress PaddleOCR warnings
export GLOG_minloglevel=2  # Suppress INFO and WARNING logs from glog
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow logs if present

# Display configuration
echo "=================================================="
echo "ID Card OCR API Configuration:"
echo "  Model: ${LLM_MODEL:-Qwen/Qwen3-1.7B-Base}"
echo "  LLM API: ${LLM_API_URL:-http://localhost:8000/v1/chat/completions}"
echo "  Flask API: http://${HOST:-0.0.0.0}:${PORT:-8000}"
echo "=================================================="
echo ""
echo "Note: This application expects an external OpenAI-compatible LLM API"
echo "configured at LLM_API_URL (see .env for details)."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down Flask API..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Flask OCR API
echo "Starting Flask OCR API on http://${HOST:-0.0.0.0}:${PORT:-8000}"

DEBUG_VAL="${DEBUG:-False}"
if [ "$(echo "$DEBUG_VAL" | tr '[:upper:]' '[:lower:]')" = "true" ] ; then
    echo "DEBUG=true -> starting Flask development server"
    python3 run.py
else
    # Ensure gunicorn installed
    if ! command -v gunicorn >/dev/null 2>&1; then
        echo "Installing gunicorn..."
        pip install gunicorn
    fi

    WORKERS=${GUNICORN_WORKERS:-4}
    HOST_BIND=${HOST:-0.0.0.0}
    PORT_BIND=${PORT:-8000}

    echo "Starting Gunicorn with ${WORKERS} workers on ${HOST_BIND}:${PORT_BIND}"
    exec gunicorn -w ${WORKERS} -b ${HOST_BIND}:${PORT_BIND} 'app:create_app()'
fi
