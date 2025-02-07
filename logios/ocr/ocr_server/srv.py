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


@app.post("/ocr_image_segment")
async def ocr_image_segment(request: OCRSegment):
    logger.info(f"Doing ocr segment for file {request.path}")

    filename = request.path
    x1 = request.x1
    x2 = request.x2
    y1 = request.y1
    y2 = request.y2
    image = np.array(Image.open(filename))
    sub_image = image[y1:y2, x1:x2]
    sub_image_pil = Image.fromarray(sub_image)
    sub_image_pil.save("test_img.png")
    size = sub_image_pil.size
    logger.info(f"Chopping subimage with size {size}")

    # ret = segmentation_and_recognition_ii(src_page="test_img.png")

    ret = segmenter.segment("test_img.png")
    return {"received_value": request.path, "ret": ret}


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
