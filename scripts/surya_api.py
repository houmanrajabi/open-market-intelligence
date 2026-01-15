"""
Surya API Server - OpenAI-compatible API for Surya OCR and Layout Detection

Deploy this on VastAI for GPU-accelerated document processing.

Usage:
    pip install fastapi uvicorn surya-ocr pillow python-multipart
    uvicorn surya_api:app --host 0.0.0.0 --port 8002 --workers 1

Endpoints:
    GET  /v1/models                    - List available models
    POST /v1/layout/detect              - Layout detection
    POST /v1/ocr/recognize              - OCR text recognition
    POST /v1/document/process           - Combined layout + OCR
    GET  /health                        - Health check
"""

import io
import base64
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# Surya imports
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Surya API Server",
    description="OpenAI-compatible API for Surya OCR and Layout Detection",
    version="1.0.0"
)

# Global model instances (loaded once at startup)
foundation_predictor = None
layout_predictor = None
recognition_predictor = None

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class ModelInfo(BaseModel):
    """Model information response"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str = "surya"

class ModelsResponse(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[ModelInfo]

class LayoutDetectionRequest(BaseModel):
    """Request for layout detection"""
    image: str = Field(..., description="Base64 encoded image")
    return_boxes: bool = Field(default=True, description="Return bounding boxes")
    return_polygons: bool = Field(default=False, description="Return polygon coordinates")

class BBoxInfo(BaseModel):
    """Bounding box information"""
    label: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: Optional[float] = None
    polygon: Optional[List[List[float]]] = None

class LayoutDetectionResponse(BaseModel):
    """Response from layout detection"""
    model: str = "surya-layout"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    elements: List[BBoxInfo]
    image_size: List[int]  # [width, height]

class OCRRequest(BaseModel):
    """Request for OCR"""
    image: str = Field(..., description="Base64 encoded image")
    languages: List[str] = Field(default=["en"], description="Language codes")

class TextLine(BaseModel):
    """OCR text line"""
    text: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: Optional[float] = None

class OCRResponse(BaseModel):
    """Response from OCR"""
    model: str = "surya-ocr"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    text_lines: List[TextLine]
    full_text: str
    image_size: List[int]  # [width, height]

class DocumentProcessRequest(BaseModel):
    """Request for full document processing (layout + OCR)"""
    image: str = Field(..., description="Base64 encoded image")
    languages: List[str] = Field(default=["en"], description="Language codes")
    return_polygons: bool = Field(default=False, description="Return polygon coordinates")

class DocumentElement(BaseModel):
    """Combined layout and OCR element"""
    element_id: str
    type: str  # Layout label (Table, Figure, Text, etc.)
    bbox: List[float]
    text: Optional[str] = None  # OCR text if applicable
    confidence: Optional[float] = None

class DocumentProcessResponse(BaseModel):
    """Response from full document processing"""
    model: str = "surya-document"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    elements: List[DocumentElement]
    image_size: List[int]

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load Surya models at startup"""
    global foundation_predictor, layout_predictor, recognition_predictor

    try:
        logger.info("Loading Surya models...")

        # Load foundation model (shared encoder)
        logger.info("Loading Foundation Predictor...")
        foundation_predictor = FoundationPredictor()

        # Load layout model
        logger.info("Loading Layout Predictor...")
        layout_predictor = LayoutPredictor(foundation_predictor)

        # Load OCR model
        logger.info("Loading Recognition Predictor...")
        recognition_predictor = RecognitionPredictor(foundation_predictor)

        logger.info("✅ All Surya models loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Surya API server...")

# ============================================================================
# Helper Functions
# ============================================================================

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def process_uploaded_image(file: UploadFile) -> Image.Image:
    """Process uploaded file to PIL Image"""
    try:
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Surya API Server",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "layout": "/v1/layout/detect",
            "ocr": "/v1/ocr/recognize",
            "document": "/v1/document/process",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "foundation": foundation_predictor is not None,
            "layout": layout_predictor is not None,
            "recognition": recognition_predictor is not None
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)"""
    models = [
        ModelInfo(id="surya-layout", owned_by="surya"),
        ModelInfo(id="surya-ocr", owned_by="surya"),
        ModelInfo(id="surya-document", owned_by="surya")
    ]
    return ModelsResponse(object="list", data=models)

@app.post("/v1/layout/detect", response_model=LayoutDetectionResponse)
async def detect_layout(request: LayoutDetectionRequest):
    """
    Detect layout elements in an image

    Args:
        request: LayoutDetectionRequest with base64 image

    Returns:
        LayoutDetectionResponse with detected elements
    """
    try:
        # Decode image
        image = decode_base64_image(request.image)
        img_width, img_height = image.size

        # Run layout detection
        logger.info(f"Running layout detection on image ({img_width}x{img_height})")
        layout_result = layout_predictor([image])[0]

        # Process results
        elements = []
        for bbox_item in layout_result.bboxes:
            bbox_info = BBoxInfo(
                label=bbox_item.label,
                bbox=bbox_item.bbox,  # [x1, y1, x2, y2]
                confidence=getattr(bbox_item, 'confidence', None)
            )

            # Add polygon if requested
            if request.return_polygons and hasattr(bbox_item, 'polygon'):
                bbox_info.polygon = bbox_item.polygon

            elements.append(bbox_info)

        logger.info(f"Detected {len(elements)} layout elements")

        return LayoutDetectionResponse(
            model="surya-layout",
            elements=elements,
            image_size=[img_width, img_height]
        )

    except Exception as e:
        logger.error(f"Layout detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Layout detection failed: {str(e)}")

@app.post("/v1/layout/detect/upload")
async def detect_layout_upload(file: UploadFile = File(...)):
    """
    Detect layout elements from uploaded file

    Args:
        file: Image file upload

    Returns:
        LayoutDetectionResponse with detected elements
    """
    try:
        # Process uploaded image
        image = process_uploaded_image(file)
        img_width, img_height = image.size

        # Run layout detection
        logger.info(f"Running layout detection on uploaded image ({img_width}x{img_height})")
        layout_result = layout_predictor([image])[0]

        # Process results
        elements = []
        for bbox_item in layout_result.bboxes:
            bbox_info = BBoxInfo(
                label=bbox_item.label,
                bbox=bbox_item.bbox,
                confidence=getattr(bbox_item, 'confidence', None)
            )
            elements.append(bbox_info)

        logger.info(f"Detected {len(elements)} layout elements")

        return LayoutDetectionResponse(
            model="surya-layout",
            elements=elements,
            image_size=[img_width, img_height]
        )

    except Exception as e:
        logger.error(f"Layout detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Layout detection failed: {str(e)}")

@app.post("/v1/ocr/recognize", response_model=OCRResponse)
async def recognize_text(request: OCRRequest):
    """
    Perform OCR on an image

    Args:
        request: OCRRequest with base64 image and languages

    Returns:
        OCRResponse with recognized text
    """
    try:
        # Decode image
        image = decode_base64_image(request.image)
        img_width, img_height = image.size

        # Run OCR
        logger.info(f"Running OCR on image ({img_width}x{img_height}) with languages: {request.languages}")
        ocr_result = recognition_predictor([image], [request.languages])[0]

        # Process results
        text_lines = []
        full_text_parts = []

        for line in ocr_result.text_lines:
            text_line = TextLine(
                text=line.text,
                bbox=line.bbox,
                confidence=getattr(line, 'confidence', None)
            )
            text_lines.append(text_line)
            full_text_parts.append(line.text)

        full_text = "\n".join(full_text_parts)

        logger.info(f"Recognized {len(text_lines)} text lines")

        return OCRResponse(
            model="surya-ocr",
            text_lines=text_lines,
            full_text=full_text,
            image_size=[img_width, img_height]
        )

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@app.post("/v1/ocr/recognize/upload")
async def recognize_text_upload(
    file: UploadFile = File(...),
    languages: List[str] = ["en"]
):
    """
    Perform OCR on uploaded file

    Args:
        file: Image file upload
        languages: Language codes

    Returns:
        OCRResponse with recognized text
    """
    try:
        # Process uploaded image
        image = process_uploaded_image(file)
        img_width, img_height = image.size

        # Run OCR
        logger.info(f"Running OCR on uploaded image ({img_width}x{img_height}) with languages: {languages}")
        ocr_result = recognition_predictor([image], [languages])[0]

        # Process results
        text_lines = []
        full_text_parts = []

        for line in ocr_result.text_lines:
            text_line = TextLine(
                text=line.text,
                bbox=line.bbox,
                confidence=getattr(line, 'confidence', None)
            )
            text_lines.append(text_line)
            full_text_parts.append(line.text)

        full_text = "\n".join(full_text_parts)

        logger.info(f"Recognized {len(text_lines)} text lines")

        return OCRResponse(
            model="surya-ocr",
            text_lines=text_lines,
            full_text=full_text,
            image_size=[img_width, img_height]
        )

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@app.post("/v1/document/process", response_model=DocumentProcessResponse)
async def process_document(request: DocumentProcessRequest):
    """
    Full document processing: Layout detection + OCR

    Args:
        request: DocumentProcessRequest with base64 image

    Returns:
        DocumentProcessResponse with combined results
    """
    try:
        # Decode image
        image = decode_base64_image(request.image)
        img_width, img_height = image.size

        # Run layout detection
        logger.info(f"Running full document processing on image ({img_width}x{img_height})")
        layout_result = layout_predictor([image])[0]

        # Run OCR
        ocr_result = recognition_predictor([image], [request.languages])[0]

        # Combine results: Map OCR lines to layout regions
        elements = []

        for idx, layout_box in enumerate(layout_result.bboxes):
            element = DocumentElement(
                element_id=f"elem_{idx}",
                type=layout_box.label,
                bbox=layout_box.bbox,
                confidence=getattr(layout_box, 'confidence', None)
            )

            # Find OCR text within this layout box
            text_in_box = []
            lx1, ly1, lx2, ly2 = layout_box.bbox

            for ocr_line in ocr_result.text_lines:
                ox1, oy1, ox2, oy2 = ocr_line.bbox

                # Check if OCR line is inside layout box (simple overlap check)
                if (ox1 >= lx1 and ox2 <= lx2 and oy1 >= ly1 and oy2 <= ly2):
                    text_in_box.append(ocr_line.text)

            if text_in_box:
                element.text = "\n".join(text_in_box)

            elements.append(element)

        logger.info(f"Processed document: {len(elements)} elements")

        return DocumentProcessResponse(
            model="surya-document",
            elements=elements,
            image_size=[img_width, img_height]
        )

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/v1/document/process/upload")
async def process_document_upload(
    file: UploadFile = File(...),
    languages: List[str] = ["en"]
):
    """
    Full document processing from uploaded file

    Args:
        file: Image file upload
        languages: Language codes

    Returns:
        DocumentProcessResponse with combined results
    """
    try:
        # Process uploaded image
        image = process_uploaded_image(file)
        img_width, img_height = image.size

        # Run layout detection
        logger.info(f"Running full document processing on uploaded image ({img_width}x{img_height})")
        layout_result = layout_predictor([image])[0]

        # Run OCR
        ocr_result = recognition_predictor([image], [languages])[0]

        # Combine results
        elements = []

        for idx, layout_box in enumerate(layout_result.bboxes):
            element = DocumentElement(
                element_id=f"elem_{idx}",
                type=layout_box.label,
                bbox=layout_box.bbox,
                confidence=getattr(layout_box, 'confidence', None)
            )

            # Find OCR text within this layout box
            text_in_box = []
            lx1, ly1, lx2, ly2 = layout_box.bbox

            for ocr_line in ocr_result.text_lines:
                ox1, oy1, ox2, oy2 = ocr_line.bbox

                if (ox1 >= lx1 and ox2 <= lx2 and oy1 >= ly1 and oy2 <= ly2):
                    text_in_box.append(ocr_line.text)

            if text_in_box:
                element.text = "\n".join(text_in_box)

            elements.append(element)

        logger.info(f"Processed document: {len(elements)} elements")

        return DocumentProcessResponse(
            model="surya-document",
            elements=elements,
            image_size=[img_width, img_height]
        )

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        workers=1,  # Surya models are not multi-process safe, use 1 worker
        log_level="info"
    )
