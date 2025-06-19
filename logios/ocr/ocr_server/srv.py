from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import shutil
import os
from ocr.predict import TextRegognizer
from segmenter import segmentation_and_recognition, segmentation_and_recognition_ii
from pydantic import BaseModel
from loguru import logger
from PIL import Image
import numpy as np
from segmenter import Segmenter

app = FastAPI()

segmenter = Segmenter()

# Directory to save uploaded images
# UPLOAD_DIRECTORY = "uploads"
# /app/data/pages/TODO/Kostas/TODO/2/000_1_cropped.png
# Ensure the upload directory exists
# os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


class StringRequest(BaseModel):
    value: str


class OCRSegment(BaseModel):
    path: str
    x1: int
    y1: int
    x2: int
    y2: int


@app.post("/process_image/")
async def process_image_string(request: StringRequest):
    logger.info(f"Received file {request.value}")
    # ret = segmentation_and_recognition(src_page=request.value)
    ret = segmenter.segment(request.value)
    logger.info(f"Ocr returned {ret} for file {request.value}")
    return {"received_value": request.value, "ret": ret}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
