from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from loguru import logger
from doclayout_yolo import YOLOv10
import pandas as pd
import numpy as np
from PIL import Image

app = FastAPI()

# Directory to save uploaded images
# /app/data/pages/TODO/Kostas/TODO/2/000_1_cropped.png


model = YOLOv10("src/model/doclayout_yolo_docstructbench_imgsz1024.pt")


class StringRequest(BaseModel):
    value: str
    threshold: float


@app.post("/process_image")
async def process_image_string(
    request: StringRequest,
):
    logger.info(f"Received file {request.value}")

    file = request.value
    conf = (
        request.threshold if request.threshold is not None else 0.5
    )  # Default threshold value
    logger.info(f"file = {file}, threshold = {conf}")

    img = Image.open(file)
    image_width, image_height = img.size
    logger.info(f"Image width: {image_width}, Image height: {image_height}")

    det_res = model.predict(
        file,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=conf,  # Confidence threshold
        device="cpu",  # Device to use (e.g., 'cuda:0' or 'cpu')
    )
    logger.info(f"Segmentation: {det_res}")

    class_names = det_res[0].names
    coords = []
    logger.info(f"Layout boxes : {len(det_res[0].boxes)}")
    for box_idx in range(len(det_res[0].boxes)):
        box = det_res[0].boxes[box_idx]
        x1, y1, x2, y2 = [int(_) for _ in box.xyxy[0].cpu().numpy().tolist()]
        cls_name = class_names[box.cls.item()]  #
        coords.append((y1, x1, y2, x2, cls_name))

    return {
        "received_value": request.value,
        "class_names": class_names,
        "block_coords": coords,
        "ret": "success",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
