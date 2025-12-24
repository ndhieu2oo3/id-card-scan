# ID Card OCR API with vLLM

A Flask API for ID card text recognition using PaddleOCR for OCR and vLLM for local LLM-based field extraction.

## Features

- **Local OCR Processing**: PaddleOCR for Vietnamese ID card text detection and recognition
- **Local LLM Serving**: Qwen2.5-7B-Instruct via vLLM (no external API dependency)
- **Structured Extraction**: Extract ID card fields with Vietnamese field descriptions
- **Parallel Processing**: Multi-threaded LLM queries for fast field extraction
- **Production Ready**: Docker support with Gunicorn + vLLM

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (for Qwen2.5-7B-Instruct)
  - RTX 3090, RTX 4090, A100, H100, etc.
  - RTX 4080 (12GB) - marginal, may require optimization
- **CPU**: 8+ cores for parallel processing
- **RAM**: 32GB+ system RAM
- **Storage**: 20GB+ for model weights and dependencies

### Recommended Setup
- **GPU**: NVIDIA A100 (40GB) or H100 (80GB) for production
- **CPU**: 16+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 64GB+ for stable concurrent processing
- **Storage**: 100GB+ fast SSD (NVMe recommended)

### Software Requirements
- **CUDA**: 11.8+ (tested with 11.8, 12.0, 13.0)
- **cuDNN**: 8.x
- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA support

## Quick Start

### Local Development

1. **Install CUDA & cuDNN** (if not already installed)
   ```bash
   # Check CUDA version
   nvcc --version
   ```

2. **Setup vLLM** (first time only)
   ```bash
   # Install vLLM with your CUDA version (e.g., 11.8)
   export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
   export CUDA_VERSION=118  # For CUDA 11.8
   
   # Using pip (requires PyTorch installed first)
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
   
   # Then install vLLM
   pip install vllm
   ```

3. **Run Application**
   ```bash
   bash run.sh
   # Starts vLLM server on port 8001
   # Starts Flask API on port 8000
   # Visit http://localhost:8000/id_card_scan/health
   ```

### Production (Docker)

```bash
docker build -t id-card-ocr .

# GPU support required
docker run --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -e LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e LLM_GPU_MEMORY_UTIL=0.8 \
  id-card-ocr
```

## Configuration

Edit `.env` to customize:

```env
# OCR Configuration
DEVICE_GPU=-1                    # -1=CPU, >=0=GPU ID (for OCR)
HARD_TOKEN=your-token            # Leave empty to disable auth

# LLM Configuration (vLLM Local Serving)
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct     # HuggingFace model ID
LLM_API_URL=http://localhost:8001/v1/chat/completions
LLM_SERVER_PORT=8001              # vLLM API port
LLM_GPU_MEMORY_UTIL=0.8           # GPU memory allocation (0.0-1.0)
LLM_MAX_MODEL_LEN=5000            # Maximum sequence length
LLM_ENFORCE_EAGER=True            # Use eager execution (faster, more memory)

# Flask API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

### Tuning for Your Hardware

**For RTX 4090 or similar (24GB VRAM)**:
```env
LLM_GPU_MEMORY_UTIL=0.85
LLM_MAX_MODEL_LEN=6000
```

**For A100 (40GB VRAM)**:
```env
LLM_GPU_MEMORY_UTIL=0.9
LLM_MAX_MODEL_LEN=8000
```

**For Limited Memory (RTX 3070, 8GB)**:
- Use quantized model: `Qwen/Qwen2.5-7B-Instruct-GPTQ` (requires `auto-gptq`)
- Or use smaller model: `Qwen/Qwen2.5-3B-Instruct`
- Reduce `LLM_GPU_MEMORY_UTIL` to 0.6-0.7

## Project Structure

```
app/
├── __init__.py              # Flask factory
├── extentions.py            # Config & settings
├── validator.py             # Auth decorator
├── api/id_card_scan/
│   ├── routes.py           # Endpoints
│   ├── algorithm.py        # OCR + LLM pipeline
│   └── utils.py            # Image preprocessing
└── vision/
    ├── weights/            # OCR Model files
    └── ocr/
        ├── ocr_service.py  # OCR service
        └── llm_extractor.py # vLLM client
```
TEXT_REC_DICT_PATH=./app/vision/weights/vn_dict.txt
TEXT_DET_PDPARAM_MODEL_PATH=./app/vision/weights/en_PP-OCRv3_det_infer
TEXT_REC_CONF_THRESHOLD=0.7
TEXT_DET_CONF_THRESHOLD=0.3
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/id_card_scan/health
```

**Response:**
```json
{
  "success": true,
  "message": "OCR API is running",
  "status": "healthy"
}
```

### OCR Processing

**Single image:**
```bash
curl -X POST http://localhost:8000/id_card_scan/ocr \
  -F "image_file=@id_card.jpg" \
  -F "hard_token=your-token"
```

**Multiple images:**
```bash
curl -X POST http://localhost:8000/id_card_scan/ocr \
  -F "image_file=@image1.jpg" \
  -F "image_file=@image2.jpg" \
  -F "hard_token=your-token"
```


**With header authentication:**
```bash
curl -X POST http://localhost:8000/id_card_scan/ocr \
  -H "X-API-Token: your-token" \
  -F "image_file=@id_card.jpg"
```

**Without text boxes (exclude ocr_data):**
```bash
curl -X POST http://localhost:8000/id_card_scan/ocr \
  -F "image_file=@id_card.jpg" \
  -F "hard_token=your-token" \
  -F "display_text_box=false"
```
**Success Response (HTTP 200):**
```json
{
  "success": true,
  "message": "OCR processing completed",
  "trace_id": "uuid",
  "data": {
    "results": [{
      "filename": "id_card.jpg",
      "ocr_data": [
        {
          "text": "NGUYEN VAN A",
          "confidence": 0.9542,
          "box": [[10, 20], [150, 20], [150, 40], [10, 40]]
        }
      ],
      "processing_time": {
        "detection_ms": 145.32,
        "recognition_ms": 234.56
      }
    }],
    "processed_count": 1,
    "error_count": 0
  }
}
```

**Error Response (HTTP 400/401/500):**
```json
{
  "success": false,
  "message": "Invalid or missing authentication token",
  "error_code": "UNAUTHORIZED",
  "trace_id": "uuid"
}
```

### OCR + LLM Extraction (Structured Field Extraction)

**Config Parameters:**
- `info` (required): Dict mapping field_key -> field_description_in_vietnamese
  - field_key: The name of the field in output JSON
  - field_description_in_vietnamese: Vietnamese description for LLM to understand what to extract
- `parallel` (optional, default: true): Extract fields in parallel using ThreadPoolExecutor
- `batch_size` (optional, default: 3): Max concurrent LLM queries
- `llm_timeout` (optional, default: 30): Timeout per field in seconds
- `display_text_box` (optional, default: false): Include ocr_data with text boxes in response

Extract specific fields from ID card using OCR + LLM intelligence.

**Single image:**
```bash
curl -X POST http://localhost:8000/id_card_scan/extract \
  -F "image_file=@id_card.jpg" \
  -F "hard_token=your-token" \
  -F 'config={"info": {"id_number": "Số căn cước", "full_name": "Tên đầy đủ", "date_of_birth": "Ngày sinh"}, "parallel": true, "batch_size": 3, "llm_timeout": 30}'
```

**Multiple images:**
```bash
curl -X POST http://localhost:8000/id_card_scan/extract \
  -F "image_file=@image1.jpg" \
  -F "image_file=@image2.jpg" \
  -F "hard_token=your-token" \
  -F 'config={"info": {"id_number": "Số căn cước", "full_name": "Tên đầy đủ", "address": "Địa chỉ"}, "parallel": true}'
```

**With text boxes included:**
```bash
curl -X POST http://localhost:8000/id_card_scan/extract \
  -F "image_file=@id_card.jpg" \
  -F "hard_token=your-token" \
  -F 'config={"info": {"id_number": "Số căn cước", "full_name": "Tên đầy đủ"}, "display_text_box": true}'
```

**Success Response (HTTP 200):**
```json
{
  "success": true,
  "message": "Extraction completed successfully",
  "trace_id": "uuid",
  "data": {
    "results": [{
      "filename": "id_card.jpg",
      "extracted_fields": {
        "full_name": "Nguyen Van A",
        "id_number": "123456789",
        "date_of_birth": "01/01/1990"
      },
      "ocr_data": [
        {
          "text": "NGUYEN VAN A",
          "confidence": 0.9542,
          "box": [[10, 20], [150, 20], [150, 40], [10, 40]]
        }
      ],
      "processing_time": {
        "ocr_ms": 234.56,
        "extraction_ms": 1250.45
      }
    }],
    "processed_count": 1,
    "error_count": 0
  }
}
```

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `UNAUTHORIZED` | 401 | Invalid/missing token |
| `MISSING_FILE` | 400 | No image file provided |
| `MISSING_CONFIG` | 400 | No config JSON provided (extract endpoint) |
| `INVALID_CONFIG` | 400 | Invalid config JSON format |
| `SYSTEM_ERROR` | 500 | OCR system not initialized |

## Installation (Manual)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 run.py
```

## Testing with Python

```python
import requests

# OCR endpoint
url = "http://localhost:8000/id_card_scan/ocr"
files = {'image_file': open('id_card.jpg', 'rb')}
data = {'hard_token': 'your-token'}

response = requests.post(url, files=files, data=data)
print(response.json())

# Extract endpoint
url = "http://localhost:8000/id_card_scan/extract"
files = {'image_file': open('id_card.jpg', 'rb')}
data = {
    'hard_token': 'your-token',
    'config': '{"info": {"id_number": "Số căn cước (12 số)", "full_name": "Họ và tên", "address":"Địa chỉ thường trú", "date_of_birth": "Ngày sinh"}, "parallel": true, "display_text_box":false}'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Docker Deployment

### Build and Run with GPU Support

```bash
docker build -t id-card-ocr .

# Run with GPU (requires nvidia-docker or Docker 20.10+ with --gpus support)
docker run --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -e LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e LLM_GPU_MEMORY_UTIL=0.8 \
  -e HARD_TOKEN=your-token \
  id-card-ocr
```

### Key Points
- Container exposes **two ports**:
  - `8000`: Flask OCR API endpoint
  - `8001`: vLLM API server (internal)
- GPU support is **required** for vLLM
- First run downloads models (~20GB) - this may take 10-30 minutes
- Healthcheck monitors Flask API on port 8000
- vLLM server automatically starts in background before Flask API

### Docker Compose Example

```yaml
version: '3.8'

services:
  id-card-ocr:
    build: .
    image: id-card-ocr:latest
    container_name: id-card-ocr-api
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
      - LLM_GPU_MEMORY_UTIL=0.8
      - LLM_MAX_MODEL_LEN=5000
      - HARD_TOKEN=your-secret-token
      - DEBUG=False
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./app/vision/weights:/app/app/vision/weights
      - ~/.cache/huggingface:/root/.cache/huggingface
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/id_card_scan/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
```

Run with:
```bash
docker-compose up --build
```

## Production Deployment (Systemd)

### Setup with vLLM + Gunicorn

Create `/etc/systemd/system/id-card-ocr.service`:

```ini
[Unit]
Description=ID Card OCR API with vLLM
After=network.target
Wants=id-card-ocr-vllm.service

[Service]
Type=notify
User=id-card
Group=id-card
WorkingDirectory=/home/id-card/id-card-ocr
Environment="PATH=/home/id-card/id-card-ocr/venv/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="LLM_MODEL=Qwen/Qwen2.5-7B-Instruct"
Environment="LLM_GPU_MEMORY_UTIL=0.8"

# Wait for vLLM server
ExecStartPre=/bin/sleep 15

# Start Flask API via Gunicorn
ExecStart=/home/id-card/id-card-ocr/venv/bin/gunicorn \
    --workers 4 \
    --worker-class sync \
    --bind 127.0.0.1:8000 \
    --timeout 120 \
    --access-logfile /var/log/id-card-ocr/access.log \
    --error-logfile /var/log/id-card-ocr/error.log \
    'app:create_app()'

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/id-card-ocr-vllm.service` for vLLM:

```ini
[Unit]
Description=ID Card OCR - vLLM Server
After=network.target

[Service]
Type=simple
User=id-card
Group=id-card
WorkingDirectory=/home/id-card/id-card-ocr
Environment="PATH=/home/id-card/id-card-ocr/venv/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="LLM_MODEL=Qwen/Qwen2.5-7B-Instruct"
Environment="LLM_API_URL=http://localhost:8001/v1/chat/completions"
Environment="LLM_SERVER_PORT=8001"
Environment="LLM_GPU_MEMORY_UTIL=0.8"
Environment="LLM_MAX_MODEL_LEN=5000"

ExecStart=/home/id-card/id-card-ocr/venv/bin/python3 \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 5000 \
    --enforce-eager

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable id-card-ocr-vllm id-card-ocr
sudo systemctl start id-card-ocr-vllm
sleep 15
sudo systemctl start id-card-ocr

# View logs
sudo journalctl -u id-card-ocr-vllm -f
sudo journalctl -u id-card-ocr -f

# Check status
sudo systemctl status id-card-ocr-vllm id-card-ocr
```

## Troubleshooting

### vLLM Server Issues

```bash
# Check if vLLM server is running
curl http://localhost:8001/v1/models

# Check GPU availability
nvidia-smi

# Check vLLM logs
tail -f /var/log/id-card-ocr/vllm.log
```

### High Memory Usage
- Reduce `LLM_GPU_MEMORY_UTIL` to 0.7 or lower
- Reduce `LLM_MAX_MODEL_LEN` to 3000-4000
- Use smaller model: `Qwen/Qwen2.5-3B-Instruct`

### Slow Responses
- Increase `LLM_GPU_MEMORY_UTIL` and `LLM_MAX_MODEL_LEN`
- Add more Gunicorn workers (4-8 for 16+ CPU cores)
- Use NVMe SSD for model cache (`~/.cache/huggingface`)

### Model Download Fails
- Check internet connection
- Set `HF_TOKEN` if using private models
- Increase HuggingFace timeout: `HF_HUB_DOWNLOAD_TIMEOUT=300`

## License

Uses PaddleOCR models and Qwen2.5-7B-Instruct. See LICENSE for details.
# id-card-scan
