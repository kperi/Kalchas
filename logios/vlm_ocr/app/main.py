import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.model import QwenOCRModel
from app.response_parser import InvalidModelResponse


logger = logging.getLogger(__name__)


class StringRequest(BaseModel):
    value: str


def _resolve_image_path(value: str) -> Path:
    image_path = Path(value).resolve()
    allowed_root = Path(os.environ.get("OCR_IMAGE_ROOT", "/app/uploads")).resolve()
    if not image_path.is_relative_to(allowed_root):
        raise HTTPException(
            status_code=400,
            detail=f"Image path must be under {allowed_root}",
        )
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return image_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Qwen OCR model")
    app.state.ocr_model = QwenOCRModel()
    logger.info("Qwen OCR model loaded")
    yield


app = FastAPI(title="Logios Qwen VLM OCR", lifespan=lifespan)


@app.get("/health")
async def health():
    model = getattr(app.state, "ocr_model", None)
    return {
        "status": "ok" if model is not None else "loading",
        "model": model.model_id if model is not None else None,
    }


@app.post("/process_image/")
async def process_image(request: StringRequest):
    image_path = _resolve_image_path(request.value)
    logger.info("Processing OCR request for %s", image_path)

    try:
        segments = await run_in_threadpool(
            app.state.ocr_model.process,
            image_path,
        )
    except InvalidModelResponse as exc:
        logger.error("Qwen returned an invalid OCR response for %s: %s", image_path, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Qwen returned an invalid OCR response: {exc}",
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("Qwen OCR failed for %s", image_path)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {exc}") from exc

    recognized_text = "\n".join(segment["text"] for segment in segments)
    return {
        "received_value": request.value,
        "ret": {
            "status": "ok",
            "file": request.value,
            "message": "Qwen VLM recognition completed",
            "recognized_text": recognized_text,
            "segments": segments,
        },
    }

