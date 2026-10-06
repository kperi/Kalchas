# Qwen VLM OCR service

This service mirrors the existing OCR service's `POST /process_image/` contract
and runs alongside it. It loads `Qwen/Qwen2.5-VL-3B-Instruct` by default and
returns line-level text and bounding boxes in the response's `ret.segments`
array.

The first startup downloads the model into the `vlm_model_cache` Docker volume.
An NVIDIA GPU and NVIDIA Container Toolkit are expected for practical inference.

## API

```bash
curl -X POST http://localhost:8001/process_image/ \
  -H "Content-Type: application/json" \
  -d '{"value":"/app/uploads/user/document/TOOCR/PNG/page_001.png"}'
```

Health and loaded model information are available from `GET /health`.

## Configuration

- `VLM_MODEL_ID`: Hugging Face model ID; defaults to
  `Qwen/Qwen2.5-VL-3B-Instruct`.
- `VLM_MAX_NEW_TOKENS`: Maximum generated tokens; defaults to `2048`.
- `VLM_MIN_PIXELS`: Processor minimum image pixels; defaults to `200704`.
- `VLM_MAX_PIXELS`: Processor maximum image pixels; defaults to `1003520`.
- `VLM_OCR_PROMPT`: Complete OCR instruction override.
- `OCR_IMAGE_ROOT`: Allowed image root; defaults to `/app/uploads`.

The Flask UI continues to use the existing `ocr` service. To test this service
as a drop-in replacement, set `OCR_SERVICE_URL=http://vlm_ocr:8000` for
`flaskui`.

