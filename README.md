# ID Card OCR API

Vietnamese ID card text recognition using PaddleOCR + OpenAI-compatible LLM API for structured field extraction.

## Features

- **OCR**: PaddleOCR for Vietnamese ID card text detection and recognition
- **LLM Integration**: OpenAI-compatible API (vLLM, OpenAI, Ollama, etc.)
- **Field Extraction**: Structured field extraction with Vietnamese descriptions
- **Parallel Processing**: Multi-threaded LLM queries
- **Production Ready**: Docker support with Gunicorn

## Requirements

- Python 3.10+
- External LLM API (OpenAI-compatible endpoint)
- Optional: NVIDIA GPU for faster OCR

## Quick Start

### Local Development

```bash
git clone <repo>
cd id-card-scan
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure .env
export LLM_API_URL=http://localhost:8000/v1/chat/completions
export LLM_MODEL=Qwen/Qwen3-1.7B-Base
export HARD_TOKEN=your-token

# Start server
bash run.sh
```

Visit: `http://localhost:8000/id_card_scan/health`

### Docker

```bash
docker build -t id-card-ocr .

docker run -p 8000:8000 \
  -e LLM_API_URL=http://localhost:8000/v1/chat/completions \
  -e LLM_MODEL=Qwen/Qwen3-1.7B-Base \
  id-card-ocr
```

## Configuration

Edit `.env`:

```env
# OCR
DEVICE_GPU=-1                    # -1=CPU, >=0=GPU ID
HARD_TOKEN=your-token            # Auth token (optional)

# LLM API
LLM_MODEL=Qwen/Qwen3-1.7B-Base
LLM_API_URL=http://localhost:8000/v1/chat/completions

# Flask
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

### Supported LLM Providers

**vLLM**
```env
LLM_API_URL=http://localhost:8000/v1/chat/completions
LLM_MODEL=Qwen/Qwen3-1.7B-Base
```

**OpenAI API**
```env
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-3.5-turbo
# Set OPENAI_API_KEY environment variable
```

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
        └── llm_extractor.py # LLM client
```

## API Examples

### Health Check

```bash
curl http://localhost:8000/id_card_scan/health
```

Response:
```json
{
  "success": true,
  "message": "OCR API is running",
  "status": "healthy"
}
```

### OCR Only

```bash
curl -X POST http://localhost:8000/id_card_scan/ocr \
  -F "image_file=@id_card.jpg" \
  -F "hard_token=your-token"
```

Response:
```json
{
  "success": true,
  "data": {
    "results": [{
      "filename": "id_card.jpg",
      "ocr_data": [
        {
          "text": "NGUYEN VAN A",
          "confidence": 0.95,
          "box": [[10, 20], [150, 20], [150, 40], [10, 40]]
        }
      ],
      "processing_time": {
        "detection_ms": 145.32,
        "recognition_ms": 234.56
      }
    }]
  }
}
```

### OCR + LLM Extraction

```bash
curl -X POST http://localhost:8000/id_card_scan/extract \
  -F "image_file=@id_card.jpg" \
  -F "hard_token=your-token" \
  -F 'config={
    "info": {
      "id_number": "Số căn cước",
      "full_name": "Tên đầy đủ",
      "date_of_birth": "Ngày sinh"
    },
    "parallel": true
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "results": [{
      "filename": "id_card.jpg",
      "extracted_fields": {
        "full_name": "NGUYEN VAN A",
        "id_number": "123456789",
        "date_of_birth": "01/01/1990"
      },
      "processing_time": {
        "ocr_ms": 234.56,
        "extraction_ms": 1250.45
      }
    }]
  }
}
```

### Multiple Images

```bash
curl -X POST http://localhost:8000/id_card_scan/ocr \
  -F "image_file=@image1.jpg" \
  -F "image_file=@image2.jpg" \
  -F "hard_token=your-token"
```

### With Header Auth

```bash
curl -X POST http://localhost:8000/id_card_scan/ocr \
  -H "X-API-Token: your-token" \
  -F "image_file=@id_card.jpg"
```

## Docker Compose

```yaml
version: '3.8'

services:
  id-card-ocr:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_API_URL=http://your-llm-service:8000/v1/chat/completions
      - LLM_MODEL=Qwen/Qwen3-1.7B-Base
      - HARD_TOKEN=your-token
    restart: unless-stopped
```

Run:
```bash
docker-compose up --build
```

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `UNAUTHORIZED` | 401 | Invalid/missing token |
| `MISSING_FILE` | 400 | No image file |
| `MISSING_CONFIG` | 400 | No config (extract endpoint) |
| `INVALID_CONFIG` | 400 | Invalid config JSON |
| `SYSTEM_ERROR` | 500 | OCR system error |

## License

Uses PaddleOCR models and OpenAI-compatible LLM APIs. See LICENSE for details.
