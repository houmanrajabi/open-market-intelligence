import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import io
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.ordering import load_model as load_order_model

app = FastAPI(title="Surya OCR API")

# Global models (load once at startup)
det_processor, det_model = segformer.load_processor(), segformer.load_model()
rec_model, rec_processor = load_model()
# order_model = load_order_model() # Optional: if you need reading order

@app.get("/health")
def health_check():
    return {"status": "ok", "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"}

@app.post("/v1/ocr")
async def process_document(file: UploadFile = File(...)):
    try:
        # 1. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. Run OCR (Detection + Recognition)
        predictions = run_ocr([image], [image], det_model, det_processor, rec_model, rec_processor)
        
        # 3. Format Output to match your pipeline expectation
        # Extract text blocks
        text_lines = []
        for result in predictions:
            for line in result.text_lines:
                text_lines.append({
                    "text": line.text,
                    "bbox": line.bbox,
                    "confidence": line.confidence
                })
                
        return {"text_lines": text_lines}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)