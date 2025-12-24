from typing import List, Optional
from pydantic import BaseModel


class OCRBox(BaseModel):
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float


class OCRResult(BaseModel):
    """Single OCR text result."""
    text: str
    confidence: float
    box: List[List[float]]


class CCCDResponse(BaseModel):
    """ID Card OCR response model."""
    success: bool
    message: str
    trace_id: str
    data: Optional[dict] = None
    error_code: Optional[str] = None
    results: Optional[List[dict]] = None


class CCCDRequest:
    """ID Card OCR request - handled as multipart form."""
    pass
